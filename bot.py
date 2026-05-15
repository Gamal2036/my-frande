"""
Advanced Multi-Modal AI Telegram Bot
=====================================
Architecture:
  - Aiogram 3.x for Telegram interaction
  - Groq (Llama-3.3-70b) as primary LLM + Whisper-large-v3 for STT
  - Gemini-1.5-Flash as fallback LLM and vision model
  - Motor (async MongoDB) for persistent per-user conversation memory
  - Pollinations AI for image generation (tool-use)
  - aiohttp + BeautifulSoup for web scraping
  - aiohttp web server on :8080 for Render health-checks

All secrets are loaded exclusively from environment variables.
"""

import asyncio
import base64
import json
import logging
import os
import re
import tempfile
from typing import Any

import aiohttp
import google.generativeai as genai
from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import (
    BufferedInputFile,
    Message,
    Voice,
)
from aiohttp import web
from bs4 import BeautifulSoup
from groq import AsyncGroq
from motor.motor_asyncio import AsyncIOMotorClient

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Environment Configuration
# ──────────────────────────────────────────────
def _require_env(name: str) -> str:
    """Retrieve a required environment variable or abort early with a clear error."""
    value = os.getenv(name)
    if not value:
        raise EnvironmentError(
            f"Required environment variable '{name}' is not set. "
            "Please add it to your Render environment configuration."
        )
    return value


BOT_TOKEN: str = _require_env("BOT_TOKEN")
GROQ_API_KEY: str = _require_env("GROQ_API_KEY")
GEMINI_API_KEY: str = _require_env("GEMINI_API_KEY")
MONGO_URL: str = _require_env("MONGO_URL")

GROQ_TEXT_MODEL = "llama-3.3-70b-versatile"
GROQ_WHISPER_MODEL = "whisper-large-v3"
GEMINI_MODEL = "gemini-1.5-flash"

MAX_HISTORY = 15  # messages kept per user
HEALTH_PORT = 8080

# ──────────────────────────────────────────────
# System Prompt
# ──────────────────────────────────────────────
SYSTEM_PROMPT = """أنت مساعد ذكاء اصطناعي متقدم واسمك "زكي". تتحدث بالعربية الفصحى بأسلوب احترافي وودي.
قواعدك:
١. تجيب دائماً بالعربية ما لم يطلب المستخدم لغة أخرى صراحةً.
٢. عندما يطلب المستخدم رسم أو توليد صورة، استخدم أداة generate_image فوراً.
٣. إذا أرسل المستخدم رابطاً، لخّص محتواه بإيجاز وأضف تحليلك.
٤. كن دقيقاً، مختصراً، ومفيداً. تجنب الإسهاب غير الضروري.
٥. لا تكشف عن مفاتيح API أو أي بيانات حساسة أبداً.
"""

# ──────────────────────────────────────────────
# Tool Definitions (Function Calling)
# ──────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "توليد صورة بناءً على وصف نصي. "
                "استخدم هذه الأداة عندما يطلب المستخدم رسم أو تخيل أو إنشاء صورة."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "الوصف الإنجليزي التفصيلي للصورة المراد توليدها.",
                    }
                },
                "required": ["prompt"],
            },
        },
    }
]


# ──────────────────────────────────────────────
# Database Manager
# ──────────────────────────────────────────────
class DatabaseManager:
    """Handles persistent conversation history via MongoDB (motor)."""

    def __init__(self, mongo_url: str) -> None:
        self._client = AsyncIOMotorClient(mongo_url)
        self._db = self._client["telegram_bot"]
        self._collection = self._db["conversations"]

    async def get_history(self, user_id: int) -> list[dict]:
        doc = await self._collection.find_one({"user_id": user_id})
        if doc:
            return doc.get("messages", [])
        return []

    async def save_history(self, user_id: int, messages: list[dict]) -> None:
        # Keep only the last MAX_HISTORY messages
        trimmed = messages[-MAX_HISTORY:]
        await self._collection.update_one(
            {"user_id": user_id},
            {"$set": {"messages": trimmed}},
            upsert=True,
        )

    async def append_message(self, user_id: int, role: str, content: str) -> list[dict]:
        history = await self.get_history(user_id)
        history.append({"role": role, "content": content})
        await self.save_history(user_id, history)
        return history

    async def close(self) -> None:
        self._client.close()


# ──────────────────────────────────────────────
# Image Generator (Pollinations AI)
# ──────────────────────────────────────────────
class ImageGenerator:
    """Generates images using the free Pollinations.ai API."""

    BASE_URL = "https://image.pollinations.ai/prompt/{prompt}"

    async def generate(self, prompt: str) -> bytes:
        encoded = aiohttp.helpers.quote(prompt, safe="")
        url = self.BASE_URL.format(prompt=encoded)
        params = {"width": 1024, "height": 1024, "nologo": "true", "enhance": "true"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                resp.raise_for_status()
                return await resp.read()


# ──────────────────────────────────────────────
# Web Scraper
# ──────────────────────────────────────────────
class WebScraper:
    """Fetches and cleans textual content from URLs."""

    MAX_CHARS = 4000

    async def fetch(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; TelegramBot/1.0)"}
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                resp.raise_for_status()
                html = await resp.text()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        # Collapse blank lines
        lines = [ln for ln in text.splitlines() if ln.strip()]
        return "\n".join(lines)[: self.MAX_CHARS]


# ──────────────────────────────────────────────
# AI Engine
# ──────────────────────────────────────────────
class AIEngine:
    """Orchestrates LLM calls: Groq primary → Gemini fallback + vision."""

    def __init__(self) -> None:
        self._groq = AsyncGroq(api_key=GROQ_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        self._gemini = genai.GenerativeModel(GEMINI_MODEL)
        self._image_gen = ImageGenerator()

    # ── Text via Groq ──────────────────────────
    async def chat_groq(
        self,
        history: list[dict],
        user_message: str,
    ) -> tuple[str, dict | None]:
        """
        Returns (text_reply, tool_call_dict | None).
        If the model requests a tool, text_reply will be empty and tool_call_dict populated.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        messages.append({"role": "user", "content": user_message})

        response = await self._groq.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.7,
        )
        choice = response.choices[0]
        finish_reason = choice.finish_reason

        if finish_reason == "tool_calls" and choice.message.tool_calls:
            tool_call = choice.message.tool_calls[0]
            return "", {
                "name": tool_call.function.name,
                "arguments": json.loads(tool_call.function.arguments),
            }

        return choice.message.content or "", None

    # ── Text via Gemini (fallback) ─────────────
    async def chat_gemini(self, history: list[dict], user_message: str) -> str:
        # Convert history to Gemini format
        gemini_history = []
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [msg["content"]]})

        chat = self._gemini.start_chat(history=gemini_history)
        full_prompt = f"{SYSTEM_PROMPT}\n\n{user_message}"
        response = await asyncio.to_thread(chat.send_message, full_prompt)
        return response.text

    # ── Vision via Gemini ──────────────────────
    async def analyze_image(self, image_bytes: bytes, caption: str | None) -> str:
        question = caption or "صف هذه الصورة بالتفصيل بالعربية."
        prompt_parts = [
            question,
            {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()},
        ]
        response = await asyncio.to_thread(self._gemini.generate_content, prompt_parts)
        return response.text

    # ── Voice via Groq Whisper ─────────────────
    async def transcribe_voice(self, audio_bytes: bytes, filename: str = "voice.ogg") -> str:
        transcription = await self._groq.audio.transcriptions.create(
            model=GROQ_WHISPER_MODEL,
            file=(filename, audio_bytes),
            response_format="text",
        )
        return transcription

    # ── Image Generation ───────────────────────
    async def generate_image(self, prompt: str) -> bytes:
        return await self._image_gen.generate(prompt)

    # ── Master reply method ────────────────────
    async def get_reply(
        self,
        history: list[dict],
        user_message: str,
    ) -> tuple[str, bytes | None]:
        """
        High-level method that attempts Groq first, falls back to Gemini.
        Returns (text_reply, image_bytes | None).
        """
        image_bytes: bytes | None = None

        try:
            text, tool_call = await self.chat_groq(history, user_message)
        except Exception as exc:
            logger.warning("Groq failed (%s), falling back to Gemini.", exc)
            try:
                text = await self.chat_gemini(history, user_message)
                tool_call = None
            except Exception as exc2:
                logger.error("Gemini also failed: %s", exc2)
                return "عذراً، واجهت مشكلة تقنية. حاول مجدداً لاحقاً.", None

        if tool_call and tool_call["name"] == "generate_image":
            prompt = tool_call["arguments"].get("prompt", user_message)
            try:
                image_bytes = await self.generate_image(prompt)
                text = f"✅ تم توليد الصورة بناءً على: _{prompt}_"
            except Exception as exc:
                logger.error("Image generation failed: %s", exc)
                text = "عذراً، فشل توليد الصورة. حاول مرة أخرى."

        return text, image_bytes


# ──────────────────────────────────────────────
# Bot Handler (Router)
# ──────────────────────────────────────────────
class BotHandler:
    """Registers all Telegram message handlers."""

    URL_RE = re.compile(r"https?://[^\s]+")

    def __init__(
        self,
        db: DatabaseManager,
        ai: AIEngine,
        scraper: WebScraper,
    ) -> None:
        self._db = db
        self._ai = ai
        self._scraper = scraper
        self.router = Router()
        self._register_handlers()

    def _register_handlers(self) -> None:
        self.router.message(CommandStart())(self.handle_start)
        self.router.message(F.voice)(self.handle_voice)
        self.router.message(F.photo)(self.handle_photo)
        self.router.message(F.text)(self.handle_text)

    # ── /start ─────────────────────────────────
    async def handle_start(self, message: Message) -> None:
        await message.answer(
            "👋 أهلاً! أنا *زكي*، مساعدك الذكي.\n\n"
            "يمكنك:\n"
            "• 💬 مراسلتي بالنص\n"
            "• 🎤 إرسال رسائل صوتية\n"
            "• 🖼️ إرسال صور لتحليلها\n"
            "• 🎨 طلب توليد صور بالكتابة\n"
            "• 🔗 إرسال روابط لتلخيصها",
            parse_mode=ParseMode.MARKDOWN,
        )

    # ── Voice ──────────────────────────────────
    async def handle_voice(self, message: Message, bot: Bot) -> None:
        voice: Voice = message.voice  # type: ignore[assignment]
        await message.answer("🎧 جارٍ تحويل الصوت إلى نص...")

        file = await bot.get_file(voice.file_id)
        buf = await bot.download_file(file.file_path)  # type: ignore[arg-type]
        audio_bytes = buf.read()  # type: ignore[union-attr]

        try:
            transcribed = await self._ai.transcribe_voice(audio_bytes)
        except Exception as exc:
            logger.error("Transcription error: %s", exc)
            await message.answer("❌ تعذّر تحويل الصوت. حاول مرة أخرى.")
            return

        await message.answer(f"📝 *النص المُستخرج:*\n{transcribed}", parse_mode=ParseMode.MARKDOWN)
        await self._process_text(message, transcribed)

    # ── Photo ──────────────────────────────────
    async def handle_photo(self, message: Message, bot: Bot) -> None:
        await message.answer("🔍 جارٍ تحليل الصورة...")

        # Use the highest-resolution variant
        photo = message.photo[-1]  # type: ignore[index]
        file = await bot.get_file(photo.file_id)
        buf = await bot.download_file(file.file_path)  # type: ignore[arg-type]
        image_bytes = buf.read()  # type: ignore[union-attr]

        caption = message.caption

        try:
            analysis = await self._ai.analyze_image(image_bytes, caption)
        except Exception as exc:
            logger.error("Vision error: %s", exc)
            await message.answer("❌ تعذّر تحليل الصورة. حاول مرة أخرى.")
            return

        user_id = message.from_user.id  # type: ignore[union-attr]
        await self._db.append_message(user_id, "user", f"[صورة] {caption or ''}")
        await self._db.append_message(user_id, "assistant", analysis)

        await message.answer(analysis)

    # ── Text ───────────────────────────────────
    async def handle_text(self, message: Message) -> None:
        text: str = message.text or ""
        await self._process_text(message, text)

    # ── Core processing pipeline ───────────────
    async def _process_text(self, message: Message, user_input: str) -> None:
        user_id = message.from_user.id  # type: ignore[union-attr]

        # Enrich input with scraped content if URLs are present
        enriched_input = await self._enrich_with_urls(user_input)

        # Retrieve history and get AI reply
        history = await self._db.get_history(user_id)

        typing_task = asyncio.create_task(self._keep_typing(message))
        try:
            reply_text, image_bytes = await self._ai.get_reply(history, enriched_input)
        finally:
            typing_task.cancel()

        # Persist conversation
        await self._db.append_message(user_id, "user", user_input)
        await self._db.append_message(user_id, "assistant", reply_text)

        # Send response
        if image_bytes:
            await message.answer_photo(
                BufferedInputFile(image_bytes, filename="image.jpg"),
                caption=reply_text,
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await message.answer(reply_text, parse_mode=ParseMode.MARKDOWN)

    async def _enrich_with_urls(self, text: str) -> str:
        urls = self.URL_RE.findall(text)
        if not urls:
            return text
        parts = [text]
        for url in urls[:2]:  # limit to 2 URLs per message
            try:
                content = await self._scraper.fetch(url)
                parts.append(f"\n\n[محتوى الرابط {url}]:\n{content}")
            except Exception as exc:
                logger.warning("Scraping %s failed: %s", url, exc)
        return "".join(parts)

    @staticmethod
    async def _keep_typing(message: Message) -> None:
        """Sends 'typing' action repeatedly while AI is processing."""
        try:
            while True:
                await message.answer_chat_action("typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass


# ──────────────────────────────────────────────
# Health Check Web Server
# ──────────────────────────────────────────────
class HealthServer:
    """Minimal aiohttp server for Render uptime health checks."""

    def __init__(self, port: int = HEALTH_PORT) -> None:
        self._port = port
        self._app = web.Application()
        self._app.router.add_get("/", self._handle)
        self._app.router.add_get("/health", self._handle)

    @staticmethod
    async def _handle(request: web.Request) -> web.Response:  # noqa: ARG004
        return web.json_response({"status": "ok", "bot": "زكي"})

    async def start(self) -> None:
        runner = web.AppRunner(self._app)
        await runner.setup()
        site = web.TCPSite(runner, "0.0.0.0", self._port)
        await site.start()
        logger.info("Health server listening on port %d", self._port)


# ──────────────────────────────────────────────
# Application Bootstrap
# ──────────────────────────────────────────────
class Application:
    """Top-level orchestrator that wires all components together."""

    def __init__(self) -> None:
        self._db = DatabaseManager(MONGO_URL)
        self._ai = AIEngine()
        self._scraper = WebScraper()
        self._handler = BotHandler(self._db, self._ai, self._scraper)

        self._bot = Bot(token=BOT_TOKEN)
        self._dp = Dispatcher()
        self._dp.include_router(self._handler.router)

        self._health = HealthServer()

    async def run(self) -> None:
        logger.info("Starting زكي bot...")
        await self._health.start()

        # Drop pending updates accumulated while offline
        await self._bot.delete_webhook(drop_pending_updates=True)

        try:
            await self._dp.start_polling(self._bot, allowed_updates=["message"])
        finally:
            await self._db.close()
            await self._bot.session.close()


# ──────────────────────────────────────────────
# Entry Point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(Application().run())
