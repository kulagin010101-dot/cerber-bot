import os
import requests
from datetime import datetime
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("FOOTBALL_API_KEY")

LEAGUES = {
    "England": 39,     # Premier League
    "Spain": 140,      # La Liga
    "Italy": 135,      # Serie A
    "Germany": 78,     # Bundesliga
    "Russia": 235      # RPL
}

HEADERS = {
    "x-apisports-key": API_KEY
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован.\n\n"
        "Команды:\n"
        "/today — матчи на сегодня\n"
    )

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    today_date = datetime.utcnow().strftime("%Y-%m-%d")
    message = "⚽ Матчи на сегодня:\n\n"

    for league_name, league_id in LEAGUES.items():
        url = "https://v3.football.api-sports.io/fixtures"
        params = {
            "league": league_id,
            "date": today_date
        }

        response = requests.get(url, headers=HEADERS, params=params)
        data = response.json()

        if data.get("response"):
            message += f"🏆 {league_name}\n"
            for match in data["response"]:
                home = match["teams"]["home"]["name"]
                away = match["teams"]["away"]["name"]
                time = match["fixture"]["date"][11:16]
                message += f"{time} — {home} vs {away}\n"
            message += "\n"

    if message == "⚽ Матчи на сегодня:\n\n":
        message += "Сегодня матчей нет."

    await update.message.reply_text(message)

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("today", today))
    app.run_polling()

if __name__ == "__main__":
    main()

