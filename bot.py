import os
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")
SPORTMONKS_API_KEY = os.getenv("SPORTMONKS_API_KEY")

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = "🔍 SportMonks диагностика\n\n"
    text += f"SPORTMONKS_API_KEY: {'OK' if SPORTMONKS_API_KEY else 'MISSING'}"
    await update.message.reply_text(text)

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.run_polling()

if __name__ == "__main__":
    main()

