import os
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "🔍 Диагностика API\n\n"
    text += f"BOT_TOKEN: {'OK' if BOT_TOKEN else 'MISSING'}\n"
    text += f"FOOTBALL_API_KEY: {'OK' if FOOTBALL_API_KEY else 'MISSING'}"
    await update.message.reply_text(text)

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.run_polling()

if __name__ == "__main__":
    main()


