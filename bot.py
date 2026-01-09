import os
import requests
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ======================
# НАСТРОЙКИ
# ======================

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("FOOTBALL_API_KEY")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MIN_PROB = 0.75
SEASON = 2025

LEAGUES = {
    39: "Англия — Премьер-лига",
    140: "Испания — Ла Лига",
    135: "Италия — Серия A",
    78: "Германия — Бундеслига",
    235: "Россия — РПЛ"
}

HEADERS = {"x-apisports-key": API_KEY}

# ======================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ======================

def get_last_matches(team_id, limit=5):
    url = f"https://v3.football.api-sports.io/fixtures"
    params = {
        "team": team_id,
        "last": limit,
        "season": SEASON
    }
    r = requests.get(url, headers=HEADERS, params=params, timeout=15).json()
    return r.get("response", [])

def analyze_goals(matches):
    total_goals = 0
    btts_yes = 0
    over_25 = 0

    for m in matches:
        g_home = m["goals"]["home"]
        g_away = m["goals"]["away"]
        if g_home is None or g_away is None:
            continue

        total_goals += g_home + g_away

        if g_home > 0 and g_away > 0:
            btts_yes += 1
        if g_home + g_away > 2:
            over_25 += 1

    played = len(matches)
    if played == 0:
        return None

    return {
        "avg_goals": total_goals / played,
        "btts_rate": btts_yes / played,
        "over25_rate": over_25 / played
    }

def calculate_probability(home_stats, away_stats):
    base = (home_stats["avg_goals"] + away_stats["avg_goals"]) / 2

    prob = 0.60
    if base >= 2.6:
        prob += 0.08
    if home_stats["over25_rate"] >= 0.6:
        prob += 0.05
    if away_stats["over25_rate"] >= 0.6:
        prob += 0.05
    if home_stats["btts_rate"] >= 0.6 and away_stats["btts_rate"] >= 0.6:
        prob += 0.04

    return min(prob, 0.88)

# ======================
# МАТЧИ СЕГОДНЯ
# ======================

def get_today_matches():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    fixtures = []

    for league_id in LEAGUES:
        url = "https://v3.football.api-sports.io/fixtures"
        params = {
            "date": today,
            "league": league_id,
            "season": SEASON
        }
        r = requests.get(url, headers=HEADERS, params=params, timeout=15).json()
        fixtures.extend(r.get("response", []))

    return fixtures

# ======================
# TELEGRAM
# ======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР — рынок ГОЛОВ\n\n"
        "Я анализирую:\n"
        "• ТБ / ТМ 2.5\n"
        "• Обе забьют\n"
        "• Индивидуальные тоталы\n\n"
        "Сигналы: /signals"
    )

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fixtures = get_today_matches()
    message = "⚽ СИГНАЛЫ ЦЕРБЕРА (ГОЛЫ)\n\n"
    found = False

    for f in fixtures:
        home = f["teams"]["home"]
        away = f["teams"]["away"]

        home_matches = get_last_matches(home["id"])
        away_matches = get_last_matches(away["id"])

        home_stats = analyze_goals(home_matches)
        away_stats = analyze_goals(away_matches)

        if not home_stats or not away_stats:
            continue

        prob = calculate_probability(home_stats, away_stats)

        if prob >= MIN_PROB:
            found = True
            time_msk = datetime.utcfromtimestamp(f["fixture"]["timestamp"]) + timedelta(hours=3)

            message += (
                f"{home['name']} — {away['name']}\n"
                f"🕒 {time_msk.strftime('%H:%M МСК')}\n"
                f"📊 ТБ 2.5\n"
                f"Вероятность: {int(prob*100)}%\n\n"
            )

    if not found:
        message = "Сегодня нет value-сигналов от 75% 🐺"

    await context.bot.send_message(chat_id=CHAT_ID, text=message)

# ======================
# ЗАПУСК
# ======================

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))
    app.run_polling()

if __name__ == "__main__":
    main()


