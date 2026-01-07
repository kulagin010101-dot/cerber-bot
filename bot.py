import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ====== ENV ======
BOT_TOKEN = os.getenv("BOT_TOKEN")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")

# ====== API CONFIG ======
API_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {
    "x-apisports-key": FOOTBALL_API_KEY
}

SEASON = 2025

# ====== LEAGUES ======
LEAGUES = {
    "🇬🇧 Англия — Премьер-лига": 39,
    "🇪🇸 Испания — Ла Лига": 140,
    "🇮🇹 Италия — Серия A": 135,
    "🇩🇪 Германия — Бундеслига": 78,
    "🇷🇺 Россия — РПЛ": 235
}

# ====== COMMANDS ======

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 *ЦЕРБЕР активирован*\n\n"
        "Я анализирую футбольные матчи топ-лиг Европы.\n\n"
        "📌 Доступные команды:\n"
        "/today — ближайшие матчи\n\n"
        "Скоро:\n"
        "• прогнозы тоталов\n"
        "• угловые и карточки\n"
        "• сигналы с value\n",
        parse_mode="Markdown"
    )

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = "⚽ *Ближайшие матчи:*\n\n"
    found = False

    for league_name, league_id in LEAGUES.items():
        params = {
            "league": league_id,
            "season": SEASON,
            "next": 5
        }

        response = requests.get(API_URL, headers=HEADERS, params=params)
        data = response.json()

        if "response" in data and data["response"]:
            message += f"*{league_name}*\n"
            for match in data["response"]:
                date = match["fixture"]["date"][:10]
                time = match["fixture"]["date"][11:16]
                home = match["teams"]["home"]["name"]
                away = match["teams"]["away"]["name"]
                message += f"`{date} {time}` — {home} vs {away}\n"
                found = True
            message += "\n"

    if not found:
        message += "Матчи не найдены (лимит API или межсезонье)."

    await update.message.reply_text(message, parse_mode="Markdown")

# ====== MAIN ======

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("today", today))
    app.run_polling()

if __name__ == "__main__":
    main()

