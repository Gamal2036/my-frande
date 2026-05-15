import asyncio
import os
import threading
import logging
import aiohttp
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from groq import Groq
import google.generativeai as genai
from collections import defaultdict
from bs4 import BeautifulSoup

# --- الإعدادات (المفاتيح المباشرة) ---
API_TOKEN = "8842786012:AAH4VmyHqjjJu_Hh0ZiAf1musjvYQRa8xP8"
GROQ_KEY = "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"
GEMINI_KEY = "AIzaSyAzxdrFEN-WeIrYGD9xh9swCh2coo-wsTc"

# إعداد الذاكرة (آخر 6 رسائل)
user_memory = defaultdict(list)

# تهيئة المحركات
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
groq_client = Groq(api_key=GROQ_KEY)
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# --- وظيفة قراءة الروابط ---
async def fetch_url_content(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=10) as response:
                if response.status == 200:
                    html = await response.text()
                    soup = BeautifulSoup(html, 'html.parser')
                    for s in soup(["script", "style"]): s.extract()
                    return soup.get_text()[:3000]
    except: return None

# --- وظيفة توليد الصور ---
async def generate_image(prompt):
    encoded = prompt.replace(" ", "%20")
    return f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&nologo=true"

# --- سيرفر الصحة (Health Server) ---
def run_health_server():
    from http.server import HTTPServer, BaseHTTPRequestHandler
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"AI Multi-Model Bot is Live")
    server = HTTPServer(('0.0.0.0', int(os.environ.get("PORT", 8080))), Handler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# --- المعالج الرئيسي ---
@dp.message(F.text)
async def handle_all(message: types.Message):
    user_id = message.from_user.id
    text = message.text
    
    # 1. طلب صورة
    if any(word in text for word in ["ارسم", "صورة", "تخيل", "draw", "imagine"]):
        await message.answer("🎨 جاري توليد الصورة باستخدام الذكاء الاصطناعي...")
        url = await generate_image(text)
        await message.answer_photo(url, caption="✨ تم التوليد بنجاح!")
        return

    # 2. معالجة الروابط
    context_text = text
    if "http" in text:
        url = "http" + text.split("http")[-1].split(" ")[0]
        await message.answer("🔍 جاري قراءة محتوى الرابط وتلخيصه لك...")
        content = await fetch_url_content(url)
        if content: context_text = f"لخص هذا الرابط بدقة: {content}"

    # 3. الذاكرة والذكاء الاصطناعي (Groq كخيار أول، Gemini كبديل)
    user_memory[user_id].append({"role": "user", "content": context_text})
    user_memory[user_id] = user_memory[user_id][-6:]
    
    await bot.send_chat_action(message.chat.id, "typing")
    
    try:
        # محاولة Groq
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "أنت مساعد محترف ثنائي اللغة، تجيب بالعربية بوضوح."}] + user_memory[user_id]
        )
        ans = response.choices[0].message.content
    except:
        # البديل Gemini إذا فشل Groq
        res = gemini_model.generate_content(context_text)
        ans = res.text

    user_memory[user_id].append({"role": "assistant", "content": ans})
    await message.answer(ans, parse_mode=ParseMode.MARKDOWN)

async def main():
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
