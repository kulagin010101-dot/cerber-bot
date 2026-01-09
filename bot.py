import os
import requests
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
API_KEY = os.getenv("FOOTBALL_API_KEY")
WEATHER_KEY = os.getenv("WEATHER_API_KEY")

MIN_PROB = 0.75

LEAGUES = [39, 140, 135, 78, 235]  # топ-лиги

# ======================
# ФАКТОРЫ
# ======================

def weather_factor(city):
    if not WEATHER_KEY:
        return 1.0
    try:
        r = requests.get(
            f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_KEY}&units=metric",
            timeout=10
        ).json()
        temp = r["main"]["temp"]
        rain = r.get("rain", {}).get("1h", 0)
        factor = 1.0
        if temp < 5 or rain > 0:
            factor *= 0.93
        return factor
    except:
        return 1.0

def motivation_factor(rank1, rank2):
    if abs(rank1 - rank2) <= 3:
        return 1.1
    return 1.0

def form_factor(goals_scored):
    if goals_scored >= 2:
        return 1.1
    elif goals_scored <= 1:
        return 0.9
    return 1.0

# ======================
# МАТЧИ
# ======================

def get_today_matches():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    headers = {"x-apisports-key": API_KEY}
    matches = []

    for league in LEAGUES:
        url = f"https://v3.football.api-sports.io/fixtures?date={today}&league={league}&season=2025"
        r = requests.get(url, headers=headers, timeout=15).json()
        for e in r.get("response", []):
            fixture = e["fixture"]
            venue = fixture.get("venue") or {}
            matches.append({
                "home": e["teams"]["home"]["name"],
                "away": e["teams"]["away"]["name"],
                "time": fixture["timestamp"],
                "city": venue.get("city", "London"),
                "avg_goals": 2.7,
                "home_goals": 1.6,
                "away_goals": 1.2,
                "home_rank": 10,
                "away_rank": 12,
                "odds": 1.9
            })
    return matches

# ======================
# ПРОГНОЗ
# ======================

def predict(match):
    base = 0.68 if match["avg_goals"] >= 2.7 else 0.66
    prob = (
        base
        * weather_factor(match["city"])
        * motivation_factor(match["home_rank"], match["away_rank"])
        * form_factor(match["home_goals"])
        * form_factor(match["away_goals"])
    )
    return prob

# ======================
# КОМАНДЫ
# ======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР (ГОЛЫ)\n"
        "Рынок: тоталы, ОЗ, ИТ\n"
        "Команда: /signals"
    )

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    matches = get_today_matches()
    text = "⚽ СИГНАЛЫ ЦЕРБЕРА (ГОЛЫ)\n\n"
    found = False

    for m in matches:
        prob = predict(m)
        value = prob * m["odds"] - 1

        if prob >= MIN_PROB and value > 0:
            found = True
            time_msk = datetime.utcfromtimestamp(m["time"]) + timedelta(hours=3)
            text += (
                f"{m['home']} — {m['away']} ({time_msk.strftime('%H:%M МСК')})\n"
                f"ТБ 2.5\n"
                f"Вероятность: {int(prob*100)}%\n"
                f"Коэфф.: {m['odds']}\n"
                f"Value: +{value:.2f}\n\n"
            )

    if not found:
        text = "Сегодня нет value-сигналов от 75% 🐺"

    await context.bot.send_message(chat_id=CHAT_ID, text=text)

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



