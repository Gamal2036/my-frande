import asyncio
import os
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from collections import defaultdict

from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from groq import Groq
import google.generativeai as genai

# --- 1. الإعدادات والمفاتيح ---
API_TOKEN = "8842786012:AAH4VmyHqjjJu_Hh0ZiAf1musjvYQRa8xP8"
GROQ_KEY = "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"
GEMINI_KEY = "AIzaSyAzxdrFEN-WeIrYGD9xh9swCh2coo-wsTc"

# إعداد السجلات لمتابعة البوت على Render
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. نظام الذاكرة الذكية (تذكر آخر 10 رسائل) ---
user_contexts = defaultdict(list)

def get_messages_with_context(user_id, new_text):
    # إضافة الرسالة الجديدة للذاكرة
    user_contexts[user_id].append({"role": "user", "content": new_text})
    # الاحتفاظ بآخر 10 رسائل فقط لضمان سرعة السيرفر المجاني
    if len(user_contexts[user_id]) > 10:
        user_contexts[user_id].pop(0)
    
    system_prompt = {"role": "system", "content": "أنت مساعد ذكي ومحترف، تجيب دائماً باللغة العربية بأسلوب مهذب وتتذكر سياق الحوار."}
    return [system_prompt] + user_contexts[user_id]

# --- 3. سيرفر الصحة (لبقاء البوت مستيقظاً في Render) ---
def run_health_server():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Bot is Active")
    server = HTTPServer(('0.0.0.0', int(os.environ.get("PORT", 8080))), Handler)
    server.serve_forever()

threading.Thread(target=run_health_server, daemon=True).start()

# --- 4. تهيئة المحركات ---
bot = Bot(token=API_TOKEN)
dp = Dispatcher()
groq_client = Groq(api_key=GROQ_KEY)
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

async def ask_ai(user_id, text):
    messages = get_messages_with_context(user_id, text)
    
    # محاولة Groq أولاً (السرعة القصوى)
    try:
        response = await asyncio.to_thread(
            groq_client.chat.completions.create,
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        answer = response.choices[0].message.content
        user_contexts[user_id].append({"role": "assistant", "content": answer})
        return answer
    except Exception as e:
        logger.error(f"Groq Error: {e}")
        # محاولة Gemini كبديل (الذكاء الفائق)
        try:
            res = await asyncio.to_thread(gemini_model.generate_content, text)
            user_contexts[user_id].append({"role": "assistant", "content": res.text})
            return res.text
        except Exception as e2:
            logger.error(f"Gemini Error: {e2}")
            return "⚠️ السيرفرات مشغولة جداً، سأكون معك خلال لحظات."

# --- 5. التعامل مع الرسائل ---
@dp.message(F.text)
async def handle_message(message: types.Message):
    # إظهار حالة "يكتب الآن..."
    await bot.send_chat_action(message.chat.id, "typing")
    reply = await ask_ai(message.from_user.id, message.text)
    await message.answer(reply, parse_mode=ParseMode.MARKDOWN)

async def main():
    # حذف الويب هوك القديم لتجنب تعارض Conflict
    await bot.delete_webhook(drop_pending_updates=True)
    logger.info("🚀 البوت الاحترافي انطلق!")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
