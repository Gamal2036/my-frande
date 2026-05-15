import asyncio
import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import defaultdict
from typing import List, Dict

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from aiogram.utils.chat_action import ChatActionSender
from groq import Groq

# --- الإعدادات الأساسية (Configuration) ---
API_TOKEN = "8842786012:AAH4VmyHqjjJu_Hh0ZiAf1musjvYQRa8xP8"
GROQ_KEY = "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"
MODEL_NAME = "llama-3.3-70b-versatile"

# إعداد السجلات (Logging) لمراقبة البوت في Render
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- إدارة الذاكرة (Memory Management) ---
class ConversationManager:
    def __init__(self, max_history: int = 10):
        self.history: Dict[int, List[Dict]] = defaultdict(list)
        self.max_history = max_history

    def add_message(self, user_id: int, role: str, content: str):
        self.history[user_id].append({"role": role, "content": content})
        if len(self.history[user_id]) > self.max_history:
            self.history[user_id].pop(0)

    def get_history(self, user_id: int) -> List[Dict]:
        system_prompt = {
            "role": "system", 
            "content": "أنت مساعد ذكي محترف. تحدث بالعربية بأسلوب مهذب. تذكر سياق الحوار دائماً ولا تخرج عن اللغة العربية إلا إذا طُلب منك ذلك."
        }
        return [system_prompt] + self.history[user_id]

chat_manager = ConversationManager(max_history=8)

# --- سيرفر الصحة (Health Check Server) لضمان بقاء السحابة ---
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot Status: Online")

def start_health_server():
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    logger.info(f"Health server started on port {port}")
    server.serve_forever()

# --- المنطق البرمجي للبوت (Bot Logic) ---
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
groq_client = Groq(api_key=GROQ_KEY)

async def call_ai(user_id: int, user_text: str) -> str:
    try:
        chat_manager.add_message(user_id, "user", user_text)
        
        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model=MODEL_NAME,
            messages=chat_manager.get_history(user_id),
            temperature=0.7,
            max_tokens=1024
        )
        
        ai_text = response.choices[0].message.content
        chat_manager.add_message(user_id, "assistant", ai_text)
        return ai_text
    except Exception as e:
        logger.error(f"AI Error: {e}")
        return "⚠️ عذراً، واجهت مشكلة فنية. هل يمكنك إعادة إرسال رسالتك؟"

@dp.message(F.text)
async def message_handler(message: types.Message):
    async with ChatActionSender.typing(bot=bot, chat_id=message.chat.id):
        reply = await call_ai(message.from_user.id, message.text)
        await message.answer(reply, parse_mode=ParseMode.MARKDOWN)

async def main():
    # تشغيل سيرفر الصحة في الخلفية
    threading.Thread(target=start_health_server, daemon=True).start()
    
    # تنظيف التحديثات القديمة وتشغيل البوت
    await bot.delete_webhook(drop_pending_updates=True)
    logger.info("Bot is starting...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot stopped.")
