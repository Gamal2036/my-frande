import os
import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
from aiohttp import web
import motor.motor_asyncio
from groq import Groq
import google.generativeai as genai

# --- الإعدادات من Render ---
TOKEN = os.getenv("BOT_TOKEN")
MONGO_URL = os.getenv("MONGO_URL")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PORT = int(os.getenv("PORT", 8080))

# --- تهيئة المكاتب ---
bot = Bot(token=TOKEN)
dp = Dispatcher()
groq_client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
mongo_client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URL)
db = mongo_client.get_database("ai_bot_db")
chats_col = db.get_collection("conversations")

# --- سيرفر الويب لـ Render (Health Check) ---
async def handle_health_check(request):
    return web.Response(text="Bot is Running!", status=200)

async def start_web_server():
    app = web.Application()
    app.router.add_get("/", handle_health_check)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", PORT)
    print(f"✅ Web Server started on port {PORT}")
    await site.start()

# --- منطق البوت ---
@dp.message(Command("start"))
async def start_cmd(message: types.Message):
    await message.answer("أهلاً بك! أنا بوتك الذكي، أستطيع التحدث، رؤية الصور، وسماع صوتك. كيف أساعدك اليوم؟")

@dp.message(F.text)
async def chat_handler(message: types.Message):
    # حفظ الذاكرة ومعالجة النص عبر Groq
    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": message.text}],
            model="llama-3.3-70b-versatile",
        )
        await message.answer(chat_completion.choices[0].message.content)
    except Exception as e:
        print(f"Error: {e}")
        await message.answer("عذراً، حدث خطأ في معالجة طلبك.")

# --- تشغيل كل شيء معاً ---
async def main():
    # تشغيل سيرفر الويب والبوت في نفس الوقت
    await start_web_server()
    print("🚀 Starting Bot Polling...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Bot Stopped.")
