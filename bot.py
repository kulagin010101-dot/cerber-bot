import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")
THESPORTSDB_API_KEY = os.getenv("THESPORTSDB_API_KEY")

if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN не задан! Проверь переменные окружения Railway.")
if not THESPORTSDB_API_KEY:
    raise ValueError("❌ THESPORTSDB_API_KEY не задан! Проверь переменные окружения Railway.")

# Ссылки на топ-лиги TheSportsDB
LEAGUES = {
    "Англия — Премьер-лига": "4328",
    "Испания — Ла Лига": "4335",
    "Италия — Серия A": "4332",
    "Германия — Бундеслига": "4331",
    "Россия — РПЛ": "4394"
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован!\n\n"
        "Команды:\n"
        "/today — ближайшие матчи топ-лиг"
    )

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = "⚽️ Ближайшие матчи:\n\n"
    found = False

    try:
        for league_name, league_id in LEAGUES.items():
            url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/eventsnextleague.php?id={league_id}"
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()

            if "events" in data and data["events"]:
                message += f"{league_name}:\n"
                for match in data["events"][:10]:  # ближайшие 10 матчей
                    date = match.get("dateEvent", "")
                    time = match.get("strTime", "")
                    home = match.get("strHomeTeam", "")
                    away = match.get("strAwayTeam", "")
                    message += f"{date} {time} — {home} vs {away}\n"
                message += "\n"
                found = True
            else:
                message += f"{league_name}: матчи не найдены\n\n"

        if not found:
            message += "Матчи не найдены."

    except Exception as e:
        message = f"❌ Ошибка при получении матчей: {e}"

    await update.message.reply_text(message)

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("today", today))
    app.run_polling()

if __name__ == "__main__":
    main()

