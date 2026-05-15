import asyncio
import os
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from aiogram import Bot, Dispatcher, types, F
from groq import Groq
import google.generativeai as genai

# جلب المفاتيح من إعدادات Render
API_TOKEN = os.getenv("BOT_TOKEN")
GROQ_KEY = os.getenv("GROQ_API_KEY")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")

# --- سيرفر وهمي لمنع Render من إغلاق البوت في الخطة المجانية ---
def run_health_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Bot is Live and Active")
    server = HTTPServer(('0.0.0.0', int(os.environ.get("PORT", 8080))), Handler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# إعداد المحركات الذكية
groq_client = Groq(api_key=GROQ_KEY)
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

async def get_ai_reply(text):
    # محاولة استخدام Groq أولاً (الأسرع عالمياً)
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[{"role": "user", "content": text}]
        )
        return completion.choices[0].message.content
    except:
        # محاولة Gemini ثانياً (كبديل قوي)
        try:
            res = gemini_model.generate_content(text)
            return res.text
        except:
            return "⚠️ عذراً، المحركات تعيد الاتصال الآن. جرب مرة أخرى."

@dp.message(F.text)
async def handle_msg(message: types.Message):
    await bot.send_chat_action(message.chat.id, "typing")
    answer = await get_ai_reply(message.text)
    await message.answer(answer)

async def main():
    print("🚀 البوت الجديد انطلق سحابياً بنجاح!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
