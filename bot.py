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

SEASON = 2025

async def today(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = "⚽ Ближайшие матчи:\n\n"
    found = False

    for league_name, league_id in LEAGUES.items():
        url = "https://v3.football.api-sports.io/fixtures"
        params = {
            "league": league_id,
            "season": SEASON,
            "next": 5
        }

        response = requests.get(url, headers=HEADERS, params=params)
        data = response.json()

        if "response" in data and data["response"]:
            message += f"🏆 {league_name}\n"
            for match in data["response"]:
                date = match["fixture"]["date"][:10]
                time = match["fixture"]["date"][11:16]
                home = match["teams"]["home"]["name"]
                away = match["teams"]["away"]["name"]
                message += f"{date} {time} — {home} vs {away}\n"
                found = True
            message += "\n"

    if not found:
        message += "Матчи не найдены (лимит API или межсезонье)."

    await update.message.reply_text(message)

