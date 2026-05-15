import asyncio
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from aiogram import Bot, Dispatcher, types, F
from groq import Groq
import google.generativeai as genai

# جلب المفاتيح من بيئة Render
API_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# سيرفر وهمي لإبقاء البوت حياً في Render
def run_health_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"AI Bot is Live")
    server = HTTPServer(('0.0.0.0', int(os.environ.get("PORT", 8080))), Handler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# إعداد المحركات
groq_client = Groq(api_key=GROQ_KEY)
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

async def get_response(text):
    try:
        # محاولة Groq أولاً
        completion = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": text}]
        )
        return completion.choices[0].message.content
    except:
        try:
            # محاولة Gemini ثانياً
            res = gemini_model.generate_content(text)
            return res.text
        except:
            return "⚠️ عذراً، لم أتمكن من معالجة الطلب حالياً."

@dp.message(F.text)
async def handle(message: types.Message):
    await bot.send_chat_action(message.chat.id, "typing")
    reply = await get_response(message.text)
    await message.answer(reply)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
