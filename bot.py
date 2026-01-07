import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# ====== ENV ======
BOT_TOKEN = os.getenv("BOT_TOKEN")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")

API_URL = "https://v3.football.api-sports.io/fixtures"
HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}

# ⚠️ ВАЖНО: сезон 2024 = сезон 2024/25
SEASON = 2024

# Лиги, которые нам нужны
LEAGUE_IDS = {
    39: "🇬🇧 Англия — Премьер-лига",
    140: "🇪🇸 Испания — Ла Лига",
    135: "🇮🇹 Италия — Серия A",
    78: "🇩🇪 Германия — Бундеслига",
    235: "🇷🇺 Россия — РПЛ",
}

# ====== COMMANDS ======

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован\n\n"
        "Команды:\n"
        "/today — ближайшие матчи\n"
    )

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = "⚽ Ближайшие матчи:\n\n"
    found = False

    # ⚠️ КЛЮЧЕВОЙ МОМЕНТ — БЕРЁМ БЕЗ ЛИГИ
    params = {
        "season": SEASON,
        "next": 50
    }

    response = requests.get(API_URL, headers=HEADERS, params=params)
    data = response.json()

    if "response" not in data:
        await update.message.reply_text("Ошибка API-Football.")
        return

    for match in data["response"]:
        league_id = match["league"]["id"]

        if league_id in LEAGUE_IDS:
            league_name = LEAGUE_IDS[league_id]
            date = match["fixture"]["date"][:10]
            time = match["fixture"]["date"][11:16]
            home = match["teams"]["home"]["name"]
            away = match["teams"]["away"]["name"]

            message += f"{league_name}\n{date} {time} — {home} vs {away}\n\n"
            found = True

    if not found:
        message += "Матчи не найдены (лимит API или пауза в лигах)."

    await update.message.reply_text(message)

# ====== MAIN ======

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("today", today))
    app.run_polling()

if __name__ == "__main__":
    main()


