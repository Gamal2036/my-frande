"""
زكي — Advanced Multi-Modal AI Telegram Bot
===========================================
Stack:
  • Aiogram 3.x           — Telegram long-polling
  • Groq Llama-3.3-70b    — primary text LLM + function-calling
  • Groq Whisper-large-v3 — speech-to-text
  • Gemini-1.5-Flash      — vision + LLM fallback
  • Motor (async MongoDB)  — per-user conversation memory (last 15 msgs)
  • Pollinations AI        — image generation (tool-use)
  • aiohttp + BS4          — web scraping
  • aiohttp web server     — health-check endpoint required by Render

All credentials come exclusively from environment variables.
Nothing is hard-coded.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import signal
import sys
import traceback
from typing import Optional
from urllib.parse import quote as url_quote

import aiohttp
import google.generativeai as genai
from aiogram import Bot, Dispatcher, F, Router
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import BufferedInputFile, Message, Voice
from aiohttp import web
from bs4 import BeautifulSoup
from groq import AsyncGroq
from motor.motor_asyncio import AsyncIOMotorClient


# ═══════════════════════════════════════════════════════════
#  LOGGING  — stdout so Render captures every line
# ═══════════════════════════════════════════════════════════
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
#  ENVIRONMENT  — crash immediately with a clear message if missing
# ═══════════════════════════════════════════════════════════
def _require(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        logger.critical(
            "❌ Required environment variable '%s' is not set. "
            "Go to Render → Your Service → Environment → Add Environment Variable.",
            name,
        )
        sys.exit(1)
    return value


BOT_TOKEN      = _require("BOT_TOKEN")
GROQ_API_KEY   = _require("GROQ_API_KEY")
GEMINI_API_KEY = _require("GEMINI_API_KEY")
MONGO_URL      = _require("MONGO_URL")

# Render injects $PORT at runtime.  We MUST bind to it — any other port
# will be unreachable and Render will kill the service (exit status 1).
PORT: int = int(os.getenv("PORT", "8080"))

GROQ_TEXT_MODEL    = "llama-3.3-70b-versatile"
GROQ_WHISPER_MODEL = "whisper-large-v3"
GEMINI_MODEL       = "gemini-1.5-flash"
MAX_HISTORY        = 15


# ═══════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════
SYSTEM_PROMPT = (
    'أنت مساعد ذكاء اصطناعي متقدم واسمك "زكي". تتحدث بالعربية الفصحى بأسلوب احترافي وودي.\n'
    "قواعدك:\n"
    "١. تجيب دائماً بالعربية ما لم يطلب المستخدم لغة أخرى صراحةً.\n"
    "٢. عندما يطلب المستخدم رسم أو توليد صورة، استخدم أداة generate_image فوراً.\n"
    "٣. إذا أرسل المستخدم رابطاً، لخّص محتواه بإيجاز وأضف تحليلك.\n"
    "٤. كن دقيقاً، مختصراً، ومفيداً. تجنب الإسهاب غير الضروري.\n"
    "٥. لا تكشف عن مفاتيح API أو أي بيانات حساسة أبداً.\n"
)


# ═══════════════════════════════════════════════════════════
#  TOOL DEFINITIONS  (Groq function-calling schema)
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
#  HEALTH SERVER
#  Binds first so Render's probe sees a 200 before anything else starts.
# ═══════════════════════════════════════════════════════════
class HealthServer:
    """
    Minimal aiohttp HTTP server for Render uptime health checks.

    Design:
    - `start()` binds the port and returns immediately (non-blocking).
      The AppRunner is stored on `self._runner` to prevent garbage-collection.
    - `stop()` drains and cleans up the runner on graceful shutdown.
    - Responds 200 OK to GET / and GET /health.
    """

    def __init__(self) -> None:
        self._runner: Optional[web.AppRunner] = None
        app = web.Application()
        app.router.add_get("/", self._handle)
        app.router.add_get("/health", self._handle)
        self._app = app

    @staticmethod
    async def _handle(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok", "bot": "زكي"})

    async def start(self) -> None:
        """Bind $PORT and return.  Does NOT block the event loop."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "0.0.0.0", PORT)
        await site.start()
        logger.info("✅ Health server listening on 0.0.0.0:%d", PORT)

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
            logger.info("⛔ Health server stopped.")


# ═══════════════════════════════════════════════════════════
#  DATABASE MANAGER
# ═══════════════════════════════════════════════════════════
class DatabaseManager:
    """
    Async MongoDB via Motor.

    - Constructor is synchronous (Motor is lazy, never blocks on __init__).
    - `connect()` issues an async ping; on failure it prints the *exact*
      exception (not a summary) and exits so the Render log shows the cause.
    - Every query method catches exceptions independently so one bad read
      never takes down the bot.
    """

    def __init__(self) -> None:
        try:
            self._client = AsyncIOMotorClient(
                MONGO_URL,
                serverSelectionTimeoutMS=8_000,
                connectTimeoutMS=8_000,
                socketTimeoutMS=10_000,
            )
            self._col = self._client["telegram_bot"]["conversations"]
            logger.info("✅ MongoDB client created (lazy — not yet connected).")
        except Exception:
            logger.critical(
                "❌ Could not instantiate MongoDB client.\n%s",
                traceback.format_exc(),
            )
            sys.exit(1)

    async def connect(self) -> None:
        """Verify connectivity at startup.  Exits on failure with full traceback."""
        logger.info("⏳ Pinging MongoDB at %s ...", MONGO_URL.split("@")[-1])
        try:
            await self._client.admin.command("ping")
            logger.info("✅ MongoDB ping succeeded — database is reachable.")
        except Exception:
            logger.critical(
                "❌ MongoDB ping FAILED. Full traceback:\n%s\n"
                "→ Check MONGO_URL in Render Environment Variables and redeploy.",
                traceback.format_exc(),
            )
            sys.exit(1)

    async def get_history(self, user_id: int) -> list[dict]:
        try:
            doc = await self._col.find_one({"user_id": user_id})
            return doc.get("messages", []) if doc else []
        except Exception:
            logger.error("MongoDB get_history error:\n%s", traceback.format_exc())
            return []

    async def append_message(self, user_id: int, role: str, content: str) -> None:
        try:
            history = await self.get_history(user_id)
            history.append({"role": role, "content": content})
            await self._col.update_one(
                {"user_id": user_id},
                {"$set": {"messages": history[-MAX_HISTORY:]}},
                upsert=True,
            )
        except Exception:
            logger.error("MongoDB append_message error:\n%s", traceback.format_exc())

    async def close(self) -> None:
        self._client.close()
        logger.info("⛔ MongoDB connection closed.")


# ═══════════════════════════════════════════════════════════
#  IMAGE GENERATOR  (Pollinations AI — free, no key needed)
# ═══════════════════════════════════════════════════════════
class ImageGenerator:
    """
    Uses stdlib urllib.parse.quote instead of aiohttp's private
    helpers.quote, which was removed in aiohttp ≥ 3.9.
    """

    _BASE = "https://image.pollinations.ai/prompt/{prompt}"

    async def generate(self, prompt: str) -> bytes:
        url = self._BASE.format(prompt=url_quote(prompt, safe=""))
        params = {"width": 1024, "height": 1024, "nologo": "true", "enhance": "true"}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=90)) as s:
            async with s.get(url, params=params) as resp:
                resp.raise_for_status()
                return await resp.read()


# ═══════════════════════════════════════════════════════════
#  WEB SCRAPER
# ═══════════════════════════════════════════════════════════
class WebScraper:
    _MAX = 4_000

    async def fetch(self, url: str) -> str:
        headers = {"User-Agent": "Mozilla/5.0 (compatible; ZakiBot/2.0)"}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as s:
            async with s.get(url, headers=headers) as resp:
                resp.raise_for_status()
                html = await resp.text()
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        lines = [ln.strip() for ln in soup.get_text("\n").splitlines() if ln.strip()]
        return "\n".join(lines)[: self._MAX]


# ═══════════════════════════════════════════════════════════
#  AI ENGINE
# ═══════════════════════════════════════════════════════════
class AIEngine:
    """
    Groq (primary text + Whisper STT) → Gemini (fallback text + vision).
    Image generation delegated to ImageGenerator via Groq tool-calling.
    """

    def __init__(self) -> None:
        self._groq    = AsyncGroq(api_key=GROQ_API_KEY)
        genai.configure(api_key=GEMINI_API_KEY)
        self._gemini  = genai.GenerativeModel(GEMINI_MODEL)
        self._img_gen = ImageGenerator()

    # ── Groq text + tool-calling ───────────────────────────
    async def _groq_chat(
        self, history: list[dict], user_msg: str
    ) -> tuple[str, Optional[dict]]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + history
        messages.append({"role": "user", "content": user_msg})
        resp = await self._groq.chat.completions.create(
            model=GROQ_TEXT_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            max_tokens=1024,
            temperature=0.7,
        )
        choice = resp.choices[0]
        if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
            tc = choice.message.tool_calls[0]
            return "", {
                "name": tc.function.name,
                "args": json.loads(tc.function.arguments),
            }
        return choice.message.content or "", None

    # ── Gemini text (fallback) ─────────────────────────────
    async def _gemini_chat(self, history: list[dict], user_msg: str) -> str:
        gemini_hist = [
            {
                "role": "user" if m["role"] == "user" else "model",
                "parts": [m["content"]],
            }
            for m in history
        ]
        chat = self._gemini.start_chat(history=gemini_hist)
        resp = await asyncio.to_thread(
            chat.send_message, f"{SYSTEM_PROMPT}\n\n{user_msg}"
        )
        return resp.text

    # ── Gemini vision ──────────────────────────────────────
    async def analyze_image(self, image_bytes: bytes, caption: Optional[str]) -> str:
        question = caption or "صف هذه الصورة بالتفصيل باللغة العربية."
        parts = [
            question,
            {"mime_type": "image/jpeg", "data": base64.b64encode(image_bytes).decode()},
        ]
        resp = await asyncio.to_thread(self._gemini.generate_content, parts)
        return resp.text

    # ── Groq Whisper STT ───────────────────────────────────
    async def transcribe(self, audio: bytes, filename: str = "voice.ogg") -> str:
        result = await self._groq.audio.transcriptions.create(
            model=GROQ_WHISPER_MODEL,
            file=(filename, audio),
            response_format="text",
        )
        return result  # type: ignore[return-value]

    # ── Master reply (Groq → Gemini fallback) ──────────────
    async def reply(
        self, history: list[dict], user_msg: str
    ) -> tuple[str, Optional[bytes]]:
        tool_call: Optional[dict] = None
        image_bytes: Optional[bytes] = None

        try:
            text, tool_call = await self._groq_chat(history, user_msg)
        except Exception as exc:
            logger.warning("Groq failed (%s) — trying Gemini.", exc)
            try:
                text = await self._gemini_chat(history, user_msg)
            except Exception as exc2:
                logger.error("Gemini fallback also failed: %s", exc2)
                return "عذراً، واجهت مشكلة تقنية. حاول مجدداً لاحقاً.", None

        if tool_call and tool_call["name"] == "generate_image":
            prompt = tool_call["args"].get("prompt", user_msg)
            try:
                image_bytes = await self._img_gen.generate(prompt)
                text = f"✅ تم توليد الصورة بناءً على: _{prompt}_"
            except Exception as exc:
                logger.error("Image generation failed: %s", exc)
                text = "عذراً، فشل توليد الصورة. حاول مرة أخرى."

        return text, image_bytes


# ═══════════════════════════════════════════════════════════
#  BOT HANDLER  (Aiogram 3.x Router)
# ═══════════════════════════════════════════════════════════
class BotHandler:
    _URL_RE = re.compile(r"https?://\S+")

    def __init__(
        self, db: DatabaseManager, ai: AIEngine, scraper: WebScraper
    ) -> None:
        self._db      = db
        self._ai      = ai
        self._scraper = scraper
        self.router   = Router()
        self._register()

    def _register(self) -> None:
        self.router.message(CommandStart())(self._start)
        self.router.message(F.voice)(self._voice)
        self.router.message(F.photo)(self._photo)
        self.router.message(F.text)(self._text)

    # /start ───────────────────────────────────────────────
    async def _start(self, msg: Message) -> None:
        await msg.answer(
            "👋 أهلاً! أنا *زكي*، مساعدك الذكي.\n\n"
            "يمكنك:\n"
            "• 💬 مراسلتي بالنص\n"
            "• 🎤 إرسال رسائل صوتية\n"
            "• 🖼️ إرسال صور لتحليلها\n"
            "• 🎨 طلب توليد صور (مثال: ارسم قطة)\n"
            "• 🔗 إرسال روابط لتلخيصها",
            parse_mode=ParseMode.MARKDOWN,
        )

    # Voice ────────────────────────────────────────────────
    async def _voice(self, msg: Message, bot: Bot) -> None:
        voice: Voice = msg.voice  # type: ignore[assignment]
        await msg.answer("🎧 جارٍ تحويل الصوت إلى نص...")
        file = await bot.get_file(voice.file_id)
        buf  = await bot.download_file(file.file_path)  # type: ignore[arg-type]
        audio_bytes = buf.read()  # type: ignore[union-attr]
        try:
            text = await self._ai.transcribe(audio_bytes)
        except Exception:
            logger.error("Transcription error:\n%s", traceback.format_exc())
            await msg.answer("❌ تعذّر تحويل الصوت. حاول مرة أخرى.")
            return
        await msg.answer(
            f"📝 *النص المُستخرج:*\n{text}", parse_mode=ParseMode.MARKDOWN
        )
        await self._process(msg, text)

    # Photo ────────────────────────────────────────────────
    async def _photo(self, msg: Message, bot: Bot) -> None:
        await msg.answer("🔍 جارٍ تحليل الصورة...")
        photo = msg.photo[-1]  # type: ignore[index]
        file  = await bot.get_file(photo.file_id)
        buf   = await bot.download_file(file.file_path)  # type: ignore[arg-type]
        image_bytes = buf.read()  # type: ignore[union-attr]
        try:
            analysis = await self._ai.analyze_image(image_bytes, msg.caption)
        except Exception:
            logger.error("Vision error:\n%s", traceback.format_exc())
            await msg.answer("❌ تعذّر تحليل الصورة. حاول مرة أخرى.")
            return
        uid = msg.from_user.id  # type: ignore[union-attr]
        await self._db.append_message(uid, "user", f"[صورة] {msg.caption or ''}")
        await self._db.append_message(uid, "assistant", analysis)
        await msg.answer(analysis)

    # Text ─────────────────────────────────────────────────
    async def _text(self, msg: Message) -> None:
        await self._process(msg, msg.text or "")

    # Core pipeline ────────────────────────────────────────
    async def _process(self, msg: Message, user_input: str) -> None:
        uid      = msg.from_user.id  # type: ignore[union-attr]
        enriched = await self._enrich(user_input)
        history  = await self._db.get_history(uid)

        typing = asyncio.create_task(self._keep_typing(msg))
        try:
            reply_text, image_bytes = await self._ai.reply(history, enriched)
        finally:
            typing.cancel()
            try:
                await typing
            except asyncio.CancelledError:
                pass

        await self._db.append_message(uid, "user", user_input)
        await self._db.append_message(uid, "assistant", reply_text)

        if image_bytes:
            await msg.answer_photo(
                BufferedInputFile(image_bytes, "image.jpg"),
                caption=reply_text,
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await msg.answer(reply_text, parse_mode=ParseMode.MARKDOWN)

    async def _enrich(self, text: str) -> str:
        urls = self._URL_RE.findall(text)
        if not urls:
            return text
        parts = [text]
        for url in urls[:2]:
            try:
                content = await self._scraper.fetch(url)
                parts.append(f"\n\n[محتوى الرابط {url}]:\n{content}")
            except Exception as exc:
                logger.warning("Scraping %s failed: %s", url, exc)
        return "".join(parts)

    @staticmethod
    async def _keep_typing(msg: Message) -> None:
        try:
            while True:
                await msg.answer_chat_action("typing")
                await asyncio.sleep(4)
        except asyncio.CancelledError:
            pass


# ═══════════════════════════════════════════════════════════
#  APPLICATION  — startup sequencer + graceful shutdown
# ═══════════════════════════════════════════════════════════
class Application:
    """
    Startup sequence (order matters for Render):
    ┌─────────────────────────────────────────────────────┐
    │ 1. HealthServer.start()  → port bound, probe → 200  │
    │ 2. DatabaseManager.connect() → ping, exit on fail   │
    │ 3. delete_webhook()      → clean long-poll baseline  │
    │ 4. start_polling()       → bot accepts messages      │
    └─────────────────────────────────────────────────────┘

    SIGTERM/SIGINT cancels the polling task and runs cleanup
    so the process exits with code 0, not 1.
    """

    def __init__(self) -> None:
        self._health  = HealthServer()
        self._db      = DatabaseManager()
        self._ai      = AIEngine()
        self._scraper = WebScraper()
        handler       = BotHandler(self._db, self._ai, self._scraper)
        self._bot     = Bot(token=BOT_TOKEN)
        self._dp      = Dispatcher()
        self._dp.include_router(handler.router)
        self._poll_task: Optional[asyncio.Task] = None

    # ── Signal handling ────────────────────────────────────
    def _attach_signals(self, loop: asyncio.AbstractEventLoop) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, lambda s=sig: self._request_shutdown(s))
            except NotImplementedError:
                pass  # Windows — handled by KeyboardInterrupt at __main__

    def _request_shutdown(self, sig: signal.Signals) -> None:
        logger.info("📶 Signal %s received — initiating graceful shutdown.", sig.name)
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()

    # ── Cleanup ────────────────────────────────────────────
    async def _cleanup(self) -> None:
        logger.info("🧹 Cleaning up resources...")
        try:
            await self._dp.stop_polling()
        except Exception:
            pass
        try:
            await self._bot.session.close()
        except Exception:
            pass
        await self._db.close()
        await self._health.stop()
        logger.info("✅ Shutdown complete.")

    # ── Main coroutine ─────────────────────────────────────
    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._attach_signals(loop)

        # 1. Health server binds FIRST so Render's probe gets 200 immediately
        await self._health.start()

        # 2. MongoDB: verify with full traceback on failure, then exit cleanly
        await self._db.connect()

        # 3. Clear stale Telegram updates from offline period
        await self._bot.delete_webhook(drop_pending_updates=True)
        logger.info("✅ Telegram webhook cleared. Bot is live — waiting for messages.")

        # 4. Start polling as a cancellable task
        #    Health server stays alive via self._health._runner (not GC'd)
        self._poll_task = asyncio.create_task(
            self._dp.start_polling(self._bot, allowed_updates=["message"])
        )
        try:
            await self._poll_task
        except asyncio.CancelledError:
            logger.info("⛔ Polling task cancelled.")
        except Exception:
            logger.critical(
                "💥 Unhandled exception in polling loop:\n%s",
                traceback.format_exc(),
            )
        finally:
            await self._cleanup()


# ═══════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    try:
        asyncio.run(Application().run())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Process exited cleanly.")
