import os
import json
import requests
from datetime import datetime, timedelta
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ==============================
# НАСТРОЙКИ
# ==============================

BOT_TOKEN = os.getenv("BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # OpenWeather ключ

MIN_PROBABILITY = 0.75
MIN_REF_YELLOW = 4.5
REF_FILE = "referees.json"

LEAGUES = {
    "EPL": 39,
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "RPL": 235,
}

# ==============================
# СУДЬИ
# ==============================

def load_referees():
    if not os.path.exists(REF_FILE):
        return {}
    with open(REF_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_referees(data):
    with open(REF_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def update_referee(refs, name, yellow, red, penalty):
    if not name or name == "Unknown":
        return
    r = refs.setdefault(name, {"matches": 0, "yellow": 0, "red": 0, "penalties": 0})
    r["matches"] += 1
    r["yellow"] += yellow
    r["red"] += red
    r["penalties"] += penalty

def referee_avg(refs, name):
    r = refs.get(name)
    if not r or r["matches"] < 5:
        return None
    return {
        "yellow_avg": r["yellow"] / r["matches"],
        "red_avg": r["red"] / r["matches"],
        "penalty_avg": r["penalties"] / r["matches"]
    }

def referee_factor(ref_data, market):
    if not ref_data:
        return 1.0
    if market == "cards":
        if ref_data["yellow_avg"] >= 5.3:
            return 1.18
        elif ref_data["yellow_avg"] <= 4.0:
            return 0.9
    if market == "goals" and ref_data["penalty_avg"] >= 0.35:
        return 1.12
    return 1.0

def referee_label(ref_data):
    if not ref_data:
        return "⚪ нет данных"
    avg = ref_data["yellow_avg"]
    if avg >= 5.3:
        return f"🔥 строгий ({avg:.1f})"
    elif avg >= 4.3:
        return f"🟡 средний ({avg:.1f})"
    else:
        return f"🟢 мягкий ({avg:.1f})"

# ==============================
# ПОГОДА
# ==============================

def get_weather_factor(stadium_city):
    if not WEATHER_API_KEY:
        return 1.0
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather?q={stadium_city}&appid={WEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=10).json()
        temp = r["main"]["temp"]
        rain = r.get("rain", {}).get("1h", 0)
        snow = r.get("snow", {}).get("1h", 0)
        factor = 1.0
        if temp < 5:
            factor *= 0.95
        elif temp > 30:
            factor *= 0.9
        if rain > 0:
            factor *= 0.92
        if snow > 0:
            factor *= 0.9
        return factor
    except:
        return 1.0

# ==============================
# МОТИВАЦИЯ
# ==============================

def get_motivation(home_rank, away_rank):
    diff = abs(home_rank - away_rank)
    if diff <= 2:
        return 1.10
    elif diff <= 5:
        return 1.05
    else:
        return 1.0

# ==============================
# ФОРМА И ТРАВМЫ
# ==============================

def get_form_factor(team_id):
    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    url = f"https://v3.football.api-sports.io/teams/form?team={team_id}&last=5"
    try:
        r = requests.get(url, headers=headers, timeout=10).json()
        form = r.get("response", [])
        wins = sum(1 for x in form if x["win"] == "Win")
        return 1.0 + (wins * 0.02)
    except:
        return 1.0

def get_injuries_factor(team_id):
    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    url = f"https://v3.football.api-sports.io/players/injuries?team={team_id}"
    try:
        r = requests.get(url, headers=headers, timeout=10).json()
        injuries = len(r.get("response", []))
        return 1.0 - min(0.1, injuries*0.02)
    except:
        return 1.0

# ==============================
# СЫГРАННЫЕ МАТЧИ ДЛЯ СУДЕЙ
# ==============================

def get_finished_matches_api():
    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    finished = []
    for league_id in LEAGUES.values():
        url = f"https://v3.football.api-sports.io/fixtures?date={yesterday}&league={league_id}&season=2025"
        r = requests.get(url, headers=headers, timeout=15).json()
        for e in r.get("response", []):
            fixture = e["fixture"]
            cards = e.get("cards", [])
            penalties = e.get("penalty", [])
            yellow = sum(1 for c in cards if c["type"] == "Yellow")
            red = sum(1 for c in cards if c["type"] == "Red")
            penalty_count = len(penalties)
            finished.append({
                "referee": fixture.get("referee") or "Unknown",
                "yellow": yellow,
                "red": red,
                "penalty": penalty_count
            })
    return finished

async def update_referees_api_job(context: ContextTypes.DEFAULT_TYPE):
    refs = load_referees()
    finished_matches = get_finished_matches_api()
    for m in finished_matches:
        update_referee(refs, m["referee"], m["yellow"], m["red"], m["penalty"])
    save_referees(refs)
    print(f"REF UPDATE API: {len(finished_matches)} matches processed")

# ==============================
# СЫГРАННЫЕ МАТЧИ СИГНАЛЫ
# ==============================

def get_today_matches():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    matches = []
    headers = {"x-apisports-key": FOOTBALL_API_KEY}
    for league_id in LEAGUES.values():
        url = f"https://v3.football.api-sports.io/fixtures?date={today}&league={league_id}&season=2025"
        r = requests.get(url, headers=headers, timeout=15).json()
        for e in r.get("response", []):
            fixture = e["fixture"]
            stadium = fixture.get("venue") or {}
            matches.append({
                "league": e["league"]["name"],
                "home": e["teams"]["home"]["name"],
                "away": e["teams"]["away"]["name"],
                "home_id": e["teams"]["home"]["id"],
                "away_id": e["teams"]["away"]["id"],
                "date": fixture["date"],
                "time": fixture["timestamp"],
                "referee": fixture.get("referee") or "Unknown",
                "stadium_city": stadium.get("city", "London"),
                "avg_goals": 2.6,
                "home_rank": e["teams"]["home"].get("rank", 10),
                "away_rank": e["teams"]["away"].get("rank", 10),
                "odds": 1.85
            })
    return matches

def predict_goals(avg):
    return {"market": "ТБ 2.5" if avg >= 2.7 else "ТМ 2.5", "type": "goals", "base_prob": 0.68 if avg >= 2.7 else 0.66}

def predict_cards():
    return {"market": "ТБ 4.5 ЖК", "type": "cards", "base_prob": 0.70}

def calculate_value(prob, odds):
    return prob * odds - 1

# ==============================
# КОМАНДЫ
# ==============================

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован.\n"
        "Сигналы приходят ежедневно в 10:00 МСК.\n"
        "Используй /signals для ручного запроса."
    )

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    refs = load_referees()
    matches = get_today_matches()
    if not matches:
        await update.message.reply_text("Сегодня матчей не найдено.")
        return

    message = "⚡️ СИГНАЛЫ ЦЕРБЕРА:\n\n"
    found = False

    for m in matches:
        t_msk = datetime.utcfromtimestamp(m["time"]) + timedelta(hours=3)
        t_msk_str = t_msk.strftime("%H:%M МСК")

        ref_data = referee_avg(refs, m["referee"])
        weather_factor = get_weather_factor(m["stadium_city"])
        motivation = get_motivation(m["home_rank"], m["away_rank"])
        home_form = get_form_factor(m["home_id"])
        away_form = get_form_factor(m["away_id"])
        home_inj = get_injuries_factor(m["home_id"])
        away_inj = get_injuries_factor(m["away_id"])
        form_inj_factor = (home_form * away_form * home_inj * away_inj)

        for pred in [predict_goals(m["avg_goals"]), predict_cards()]:
            if pred["type"] == "cards" and (not ref_data or ref_data["yellow_avg"] < MIN_REF_YELLOW):
                continue

            prob = pred["base_prob"] * motivation * weather_factor * referee_factor(ref_data, pred["type"]) * form_inj_factor
            value = calculate_value(prob, m["odds"])

            if prob >= MIN_PROBABILITY and value > 0:
                found = True
                message += (
                    f"⚽ {m['home']} — {m['away']} ({t_msk_str})\n"
                    f"{pred['market']}\n"
                    f"🧑‍⚖️ Судья: {m['referee']} — {referee_label(ref_data)}\n"
                    f"Вероятность: {int(prob*100)}%\nКоэфф.: {m['odds']}\nValue: +{value:.2f}\n\n"
                )

    if not found:
        message = "Сегодня нет value-сигналов от 75% 🐺"

    await context.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

# ==============================
# ЗАПУСК
# ==============================

async def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # Хэндлеры
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))

    # JobQueue для автообновления судей
    app.job_queue.run_daily(update_referees_api_job, time=datetime.strptime("00:00", "%H:%M").time())

    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())


