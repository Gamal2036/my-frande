import asyncio
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from aiogram import Bot, Dispatcher, types, F
from groq import Groq
import google.generativeai as genai

# جلب المفاتيح من السيرفر
API_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# سيرفر الصحة (ضروري جداً لـ Render)
def run_health_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"AI System is Online")
    server = HTTPServer(('0.0.0.0', int(os.environ.get("PORT", 8080))), Handler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

async def get_ai_response(user_text):
    # محاولة Groq (Llama 3.1)
    try:
        client = Groq(api_key=GROQ_KEY)
        completion = client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": user_text}]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Groq Error: {e}")
        # محاولة Gemini كبديل
        try:
            genai.configure(api_key=GEMINI_KEY)
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(user_text)
            return response.text
        except Exception as e2:
            print(f"Gemini Error: {e2}")
            return "⚠️ عذراً، هناك مشكلة في الربط مع السيرفرات العالمية. جرب المحاولة بعد دقيقة."

@dp.message(F.text)
async def handle_message(message: types.Message):
    await bot.send_chat_action(message.chat.id, "typing")
    response = await get_ai_response(message.text)
    await message.answer(response)

async def main():
    print("🚀 البوت يعمل الآن بكامل طاقته...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
