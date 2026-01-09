import os
import requests
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("FOOTBALL_API_KEY")
WEATHER_KEY = os.getenv("WEATHER_API_KEY")
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
# ПОГОДА
# ======================

def weather_factor(city):
    if not WEATHER_KEY or not city:
        return 1.0
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={
                "q": city,
                "appid": WEATHER_KEY,
                "units": "metric"
            },
            timeout=10
        ).json()

        temp = r["main"]["temp"]
        rain = r.get("rain", {}).get("1h", 0)
        wind = r["wind"]["speed"]

        factor = 1.0

        if temp < 5 or temp > 28:
            factor *= 0.95
        if rain > 0:
            factor *= 0.95
        if wind > 8:
            factor *= 0.96

        return factor
    except:
        return 1.0

# ======================
# СТАТИСТИКА ГОЛОВ
# ======================

def get_last_matches(team_id, limit=5):
    r = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=HEADERS,
        params={"team": team_id, "last": limit, "season": SEASON},
        timeout=15
    ).json()
    return r.get("response", [])

def analyze_goals(matches):
    total, btts, over = 0, 0, 0

    for m in matches:
        h, a = m["goals"]["home"], m["goals"]["away"]
        if h is None or a is None:
            continue
        total += h + a
        if h > 0 and a > 0:
            btts += 1
        if h + a > 2:
            over += 1

    played = len(matches)
    if played == 0:
        return None

    return {
        "avg": total / played,
        "btts": btts / played,
        "over25": over / played
    }

def calculate_probability(home, away, weather_k):
    base = (home["avg"] + away["avg"]) / 2
    prob = 0.60

    if base >= 2.6:
        prob += 0.08
    if home["over25"] >= 0.6:
        prob += 0.05
    if away["over25"] >= 0.6:
        prob += 0.05
    if home["btts"] >= 0.6 and away["btts"] >= 0.6:
        prob += 0.04

    prob *= weather_k
    return min(prob, 0.88)

# ======================
# МАТЧИ
# ======================

def get_today_matches():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    fixtures = []

    for league in LEAGUES:
        r = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers=HEADERS,
            params={"date": today, "league": league, "season": SEASON},
            timeout=15
        ).json()
        fixtures.extend(r.get("response", []))

    return fixtures

# ======================
# TELEGRAM
# ======================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР — рынок ГОЛОВ\n"
        "Факторы: форма + погода\n\n"
        "Сигналы: /signals"
    )

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fixtures = get_today_matches()
    msg = "⚽ СИГНАЛЫ ЦЕРБЕРА (ГОЛЫ)\n\n"
    found = False

    for f in fixtures:
        home = f["teams"]["home"]
        away = f["teams"]["away"]
        city = f["fixture"]["venue"]["city"]

        hs = analyze_goals(get_last_matches(home["id"]))
        as_ = analyze_goals(get_last_matches(away["id"]))

        if not hs or not as_:
            continue

        weather_k = weather_factor(city)
        prob = calculate_probability(hs, as_, weather_k)

        if prob >= MIN_PROB:
            found = True
            time_msk = datetime.utcfromtimestamp(f["fixture"]["timestamp"]) + timedelta(hours=3)

            msg += (
                f"{home['name']} — {away['name']}\n"
                f"🕒 {time_msk.strftime('%H:%M МСК')}\n"
                f"📊 ТБ 2.5\n"
                f"Вероятность: {int(prob*100)}%\n\n"
            )

    if not found:
        msg = "Сегодня нет value-сигналов от 75% 🐺"

    await context.bot.send_message(chat_id=CHAT_ID, text=msg)

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

