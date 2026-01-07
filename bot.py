import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")
SPORTMONKS_API_KEY = os.getenv("SPORTMONKS_API_KEY")

LEAGUES = {
    39: "🇬🇧 Англия — Премьер-лига",
    140: "🇪🇸 Испания — Ла Лига",
    135: "🇮🇹 Италия — Серия A",
    78: "🇩🇪 Германия — Бундеслига",
    235: "🇷🇺 Россия — РПЛ",
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован!\n\n"
        "Команды:\n"
        "/today — ближайшие матчи топ-лиг\n"
    )

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = "⚽ *Ближайшие матчи:*\n\n"
    found = False

    try:
        url = "https://soccer.sportmonks.com/api/v3/fixtures"
        league_ids = ",".join(str(x) for x in LEAGUES.keys())
        params = {
            "api_token": SPORTMONKS_API_KEY,
            "include": "localTeam,visitorTeam,league",
            "sort": "starting_at",
            "filter[league_id]": league_ids,
            "per_page": 10,
            "page": 1
        }

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "data" not in data or not data["data"]:
            await update.message.reply_text("Матчи не найдены или лимит Free API исчерпан.")
            return

        for match in data["data"]:
            league_id = match["league"]["data"]["id"]
            if league_id in LEAGUES:
                league_name = LEAGUES[league_id]
                date = match["starting_at"]["date"]
                time = match["starting_at"]["time"]
                home = match["localTeam"]["data"]["name"]
                away = match["visitorTeam"]["data"]["name"]
                message += f"*{league_name}*\n`{date} {time}` — {home} vs {away}\n\n"
                found = True

        if not found:
            message += "Матчи не найдены."

    except Exception as e:
        message = f"Ошибка при получении матчей: {e}"

    await update.message.reply_text(message, parse_mode="Markdown")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("today", today))
    app.run_polling()

if __name__ == "__main__":
    main()

