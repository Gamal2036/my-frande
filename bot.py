import asyncio
import logging
from aiogram import Bot, Dispatcher, types, F
from aiogram.enums import ParseMode
from groq import Groq
import google.generativeai as genai

# --- الإعدادات الفائقة ---
API_TOKEN = "8704241166:AAGBm2zdldKzZUoO3aCdhRhBMab2dU6xlvA"
GROQ_KEY = "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"
GEMINI_KEY = "AIzaSyDVAtKS53h8--6dCjSygv64SVmVyRO2gEg"

# تعليمات النظام (تجعل البوت بمستوى خبير بشري)
AI_INSTRUCTIONS = (
    "أنت نظام ذكاء اصطناعي متطور (Master AI). "
    "واجبك تقديم إجابات دقيقة، تقنية، ومنسقة بأعلى معايير الجودة. "
    "استخدم لغة Markdown المتقدمة: الجداول، كتل البرمجية، والقوائم. "
    "كن سريع البديهة، وإذا كان السؤال غامضاً، اطلب توضيحاً بذكاء."
)

# تهيئة المحركات مع إعدادات تخطي البطء
# تم استخدام محرك Llama 3.1 70B و Gemini 1.5 Flash
groq_client = Groq(api_key=GROQ_KEY, timeout=60.0) 
genai.configure(api_key=GEMINI_KEY)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

async def fetch_ai_response(prompt):
    """نظام التبديل التلقائي الاحترافي"""
    # الخيار الأول: Groq Llama 3.1 (الأقوى والأسرع)
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-70b-versatile",
            messages=[
                {"role": "system", "content": AI_INSTRUCTIONS},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=2048
        )
        return completion.choices[0].message.content
    except Exception as e:
        logging.error(f"Groq failed: {e}")
        
        # الخيار الثاني: Gemini 1.5 Flash (البديل الذكي)
        try:
            model = genai.GenerativeModel(
                model_name='gemini-1.5-flash',
                system_instruction=AI_INSTRUCTIONS
            )
            response = model.generate_content(prompt)
            return response.text
        except Exception as ge:
            logging.error(f"Gemini failed: {ge}")
            return "⚠️ **عذراً، هناك اضطراب شديد في الشبكة المحلية.**\nيرجى التحقق من اتصال الـ VPN أو المحاولة لاحقاً."

@dp.message(F.text)
async def pro_handler(message: types.Message):
    # إرسال رسالة انتظار احترافية
    waiting_msg = await message.answer("🔄 **جاري المعالجة عبر النظام السحابي...**")
    await bot.send_chat_action(message.chat.id, "typing")
    
    answer = await fetch_ai_response(message.text)
    
    try:
        # محاولة تحديث الرسالة بالرد النهائي (تنسيق Markdown)
        await waiting_msg.edit_text(answer, parse_mode=ParseMode.MARKDOWN)
    except:
        # في حال كان الرد طويلاً جداً (أكثر من 4000 حرف) أو خطأ تنسيق
        await waiting_msg.delete()
        if len(answer) > 4000:
            for i in range(0, len(answer), 4000):
                await message.answer(answer[i:i+4000])
        else:
            await message.answer(answer)

async def main():
    logging.basicConfig(level=logging.INFO)
    print("🚀 البوت الخارق يعمل الآن بنظام التبادل التلقائي...")
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
