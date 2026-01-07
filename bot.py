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
    "🇩🇪 Германия — Бундеслига": 78,import requests
from bs4 import BeautifulSoup
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

BOT_TOKEN = "ВАШ_BOT_TOKEN"  # Лучше через ENV переменные

# Ссылки на топ-лиги FlashScore
LEAGUES = {
    "Англия — Премьер-лига": "https://www.flashscore.com/football/england/premier-league/",
    "Испания — Ла Лига": "https://www.flashscore.com/football/spain/laliga/",
    "Италия — Серия A": "https://www.flashscore.com/football/italy/serie-a/",
    "Германия — Бундеслига": "https://www.flashscore.com/football/germany/bundesliga/",
    "Россия — РПЛ": "https://www.flashscore.com/football/russia/premier-league/"
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован!\n\n"
        "Команды:\n"
        "/today — ближайшие матчи топ-лиг"
    )

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = "⚽ *Ближайшие матчи:*\n\n"
    
    try:
        for league_name, url in LEAGUES.items():
            response = requests.get(url, headers=HEADERS)
            soup = BeautifulSoup(response.text, "html.parser")
            
            matches = []
            for match in soup.select(".event__match")[:10]:  # Берём 10 ближайших
                home = match.select_one(".event__participant--home")
                away = match.select_one(".event__participant--away")
                time = match.select_one(".event__time")
                
                if home and away and time:
                    matches.append({
                        "home": home.text.strip(),
                        "away": away.text.strip(),
                        "time": time.text.strip()
                    })
            
            if matches:
                message += f"*{league_name}*\n"
                for m in matches:
                    message += f"`{m['time']}` — {m['home']} vs {m['away']}\n"
                message += "\n"
        
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

