import os
import requests
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")
if not BOT_TOKEN:
    raise ValueError("❌ BOT_TOKEN не задан! Проверь переменные окружения Railway.")

# Топ-лиги и их internal IDs для FlashScore JSON
LEAGUES = {
    "Англия — Премьер-лига": "1",
    "Испания — Ла Лига": "2",
    "Италия — Серия A": "3",
    "Германия — Бундеслига": "4",
    "Россия — РПЛ": "5"
}

# Базовый endpoint JSON (FlashScore)
FLASH_URL = "https://d.flashscore.com/x/feed/0_football_en_uk.js"

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован!\n\n"
        "Команды:\n"
        "/today — ближайшие матчи топ-лиг"
    )

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = "⚽ *Ближайшие матчи:*\n\n"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0"
        }
        resp = requests.get(FLASH_URL, headers=headers)
        resp.raise_for_status()
        text = resp.text

        # FlashScore JSON приходит как JS-переменная, убираем лишнее
        start_idx = text.find("window['fsFeed'] = ") + len("window['fsFeed'] = ")
        end_idx = text.rfind(";")
        json_text = text[start_idx:end_idx]

        import json
        data = json.loads(json_text)

        found = False
        for league_name, league_id in LEAGUES.items():
            matches = []
            for ev in data.get("events", []):
                if ev.get("leagueId") == league_id:
                    home = ev.get("homeTeam", {}).get("name")
                    away = ev.get("awayTeam", {}).get("name")
                    time = ev.get("startTime")
                    if home and away and time:
                        matches.append({"home": home, "away": away, "time": time})

            if matches:
                message += f"*{league_name}*\n"
                for m in matches[:10]:  # по 10 матчей
                    message += f"`{m['time']}` — {m['home']} vs {m['away']}\n"
                message += "\n"
                found = True
            else:
                message += f"*{league_name}*: матчи не найдены\n\n"

        if not found:
            message += "Матчи не найдены."

    except Exception as e:
        message = f"❌ Ошибка при получении матчей: {e}"

    await update.message.reply_text(message, parse_mode="Markdown")

def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("today", today))
    app.run_polling()

if __name__ == "__main__":
    main()

