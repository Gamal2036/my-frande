import asyncio
import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from groq import Groq
import google.generativeai as genai

# --- 1. جلب المفاتيح من البيئة السحابية ---
API_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# --- 2. سيرفر وهمي لإبقاء Render مجانياً ---
def run_health_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Bot is Live!")
    server = HTTPServer(('0.0.0.0', int(os.environ.get("PORT", 8080))), Handler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# --- 3. إعداد المحركات ---
groq_client = Groq(api_key=GROQ_KEY)
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

async def get_smart_reply(text):
    # محاولة Groq (الأسرع)
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "system", "content": "أنت مساعد ذكي محترف."}, {"role": "user", "content": text}]
        )
        return completion.choices[0].message.content
    except:
        # محاولة Gemini (البديل)
        try:
            res = gemini_model.generate_content(text)
            return res.text
        except:
            return "⚠️ المحركات مشغولة، جرب بعد ثوانٍ."

@dp.message(F.text)
async def handle(message: types.Message):
    await bot.send_chat_action(message.chat.id, "typing")
    reply = await get_smart_reply(message.text)
    await message.answer(reply, parse_mode=ParseMode.MARKDOWN)

async def main():
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
