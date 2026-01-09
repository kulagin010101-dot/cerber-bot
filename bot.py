import os
import json
import requests
from datetime import datetime, timedelta, time
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, JobQueue

BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("FOOTBALL_API_KEY")
WEATHER_KEY = os.getenv("WEATHER_API_KEY")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

MIN_PROB = 0.75
SEASON = 2025
HEADERS = {"x-apisports-key": API_KEY}
REF_FILE = "referees.json"
TEAM_IDS = [39, 140, 135, 78, 235]  # Англия, Испания, Италия, Германия, Россия

# ======================
# ЗАГРУЗКА/СОХРАНЕНИЕ СУДЕЙ
# ======================
if os.path.exists(REF_FILE):
    with open(REF_FILE, "r", encoding="utf-8") as f:
        REFEREES = json.load(f)
else:
    REFEREES = {}

def save_referees():
    with open(REF_FILE, "w", encoding="utf-8") as f:
        json.dump(REFEREES, f, ensure_ascii=False, indent=2)

def referee_factor(name):
    if not name or name not in REFEREES:
        return 1.0
    pen = REFEREES[name]["penalties_per_game"]
    if pen >= 0.30:
        return 1.07
    if pen <= 0.18:
        return 0.94
    return 1.0

# ======================
# ОБНОВЛЕНИЕ СУДЕЙ
# ======================
def fetch_team_fixtures(team_id, last=20):
    try:
        r = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers=HEADERS,
            params={"team": team_id, "last": last, "season": SEASON},
            timeout=15
        ).json()
        return r.get("response", [])
    except:
        return []

def update_referees():
    global REFEREES
    for team_id in TEAM_IDS:
        fixtures = fetch_team_fixtures(team_id)
        for f in fixtures:
            referee = f["fixture"]["referee"]
            if not referee:
                continue

            if referee not in REFEREES:
                REFEREES[referee] = {"penalties_per_game": 0, "avg_goals": 0, "count": 0}

            pen = REFEREES[referee]["penalties_per_game"] * REFEREES[referee]["count"]
            goals = REFEREES[referee]["avg_goals"] * REFEREES[referee]["count"]
            count = REFEREES[referee]["count"]

            h, a = f["goals"]["home"], f["goals"]["away"]
            if h is None or a is None:
                continue
            total_goals = h + a

            penalties = 0
            for e in f.get("events", []):
                if e.get("type") == "Penalty":
                    penalties += 1

            count += 1
            pen = (pen + penalties) / count
            goals = (goals + total_goals) / count

            REFEREES[referee] = {
                "penalties_per_game": pen,
                "avg_goals": goals,
                "count": count
            }

    for r in REFEREES:
        REFEREES[r].pop("count", None)

    save_referees()
    print(f"[{datetime.now()}] База судей обновлена. Всего судей: {len(REFEREES)}")

# ======================
# ПОГОДА
# ======================
def weather_factor(city):
    if not WEATHER_KEY or not city:
        return 1.0
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": WEATHER_KEY, "units": "metric"},
            timeout=10
        ).json()
        factor = 1.0
        if r["main"]["temp"] < 5:
            factor *= 0.95
        if r.get("rain"):
            factor *= 0.95
        if r["wind"]["speed"] > 8:
            factor *= 0.96
        return factor
    except:
        return 1.0

# ======================
# ГОЛЫ
# ======================
def get_last_matches(team_id):
    r = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=HEADERS,
        params={"team": team_id, "last": 5, "season": SEASON},
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
    return {"avg": total / played, "btts": btts / played, "over25": over / played}

def calculate_probability(hs, as_, weather_k, ref_k):
    prob = 0.60
    base = (hs["avg"] + as_["avg"]) / 2
    if base >= 2.6:
        prob += 0.08
    if hs["over25"] >= 0.6:
        prob += 0.05
    if as_["over25"] >= 0.6:
        prob += 0.05
    if hs["btts"] >= 0.6 and as_["btts"] >= 0.6:
        prob += 0.04
    prob *= weather_k
    prob *= ref_k
    return min(prob, 0.88)

# ======================
# МАТЧИ
# ======================
def get_today_matches():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    r = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=HEADERS,
        params={"date": today, "season": SEASON},
        timeout=15
    ).json()
    return r.get("response", [])

# ======================
# TELEGRAM
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР — рынок ГОЛОВ\n"
        "Факторы: форма + погода + судья (обновляется ежедневно)\n\n"
        "Команда: /signals"
    )

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fixtures = get_today_matches()
    msg = "⚽ СИГНАЛЫ ЦЕРБЕРА (ГОЛЫ)\n\n"
    found = False
    for f in fixtures:
        home = f["teams"]["home"]
        away = f["teams"]["away"]
        city = f["fixture"]["venue"]["city"]
        referee = f["fixture"]["referee"]

        hs = analyze_goals(get_last_matches(home["id"]))
        as_ = analyze_goals(get_last_matches(away["id"]))
        if not hs or not as_:
            continue

        prob = calculate_probability(
            hs,
            as_,
            weather_factor(city),
            referee_factor(referee)
        )
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
# ФОН
# ======================
async def daily_ref_update(context: ContextTypes.DEFAULT_TYPE):
    print(f"[{datetime.now()}] Ежедневное обновление базы судей...")
    update_referees()
    print(f"[{datetime.now()}] База судей обновлена.")

# ======================
# ЗАПУСК
# ======================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))

    # Планируем фоновое обновление каждый день в 03:00 UTC
    job_queue: JobQueue = app.job_queue
    job_queue.run_daily(daily_ref_update, time=time(hour=3, minute=0))

    app.run_polling()

if __name__ == "__main__":
    main()

