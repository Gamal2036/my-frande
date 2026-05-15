import asyncio
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from aiogram import Bot, Dispatcher, types, F
from groq import Groq
import google.generativeai as genai

# --- وضع المفاتيح مباشرة لحل المشكلة نهائياً ---
API_TOKEN = "8842786012:AAH4VmyHqjjJu_Hh0ZiAf1musjvYQRa8xP8"
GROQ_KEY = "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"
GEMINI_KEY = "AIzaSyDVAtKS53h8--6dCjSygv64SVmVyRO2gEg"

# سيرفر الصحة لمنع توقف Render
def run_health_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Bot is Healthy")
    server = HTTPServer(('0.0.0.0', int(os.environ.get("PORT", 8080))), Handler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

async def get_ai_response(user_text):
    # 1. محاولة Groq (استخدام أحدث نموذج متاح Llama 3.3)
    try:
        client = Groq(api_key=GROQ_KEY)
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # هذا هو الأحدث والأكثر استقراراً الآن
            messages=[{"role": "user", "content": user_text}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Groq Error: {e}")
        # 2. البديل المستقر جداً من Gemini
        try:
            genai.configure(api_key=GEMINI_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(user_text)
            return response.text
        except Exception as e2:
            print(f"Gemini Error: {e2}")
            return "⚠️ السيرفرات قيد التحديث، جرب بعد ثوانٍ قليلة."

@dp.message(F.text)
async def handle_msg(message: types.Message):
    await bot.send_chat_action(message.chat.id, "typing")
    response = await get_ai_response(message.text)
    await message.answer(response)

async def main():
    # سطر هام جداً لحل مشكلة Conflict التي تظهر في صورتك
    await bot.delete_webhook(drop_pending_updates=True)
    print("🚀 البوت انطلق بالنماذج المحدثة والمفاتيح المباشرة!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
