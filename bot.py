import os
import requests
from datetime import datetime, time, timedelta
import asyncio
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ================= НАСТРОЙКИ =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # Обязательно число!
THESPORTSDB_API_KEY = os.getenv("THESPORTSDB_API_KEY", "1")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

MIN_PROBABILITY = 0.75  # порог сильных сигналов
LEAGUES = {
    "English Premier League": 4328,
    "Spanish La Liga": 4335,
    "Italian Serie A": 4332,
    "German Bundesliga": 4331,
    "Russian Premier League": 4398
}

# ================= ФУНКЦИИ =================
def calculate_value(probability, odds):
    return probability * odds - 1

def compute_probability(stat_prob, h2h_prob, motivation_prob, weather_factor, injury_factor):
    prob = stat_prob*0.5 + h2h_prob*0.2 + motivation_prob*0.2
    prob *= weather_factor * injury_factor
    return prob

def predict_goals(avg_goals):
    if avg_goals >= 3.0:
        return {"market": "ТБ 2.5", "probability": 0.78}
    elif avg_goals <= 2.0:
        return {"market": "ТМ 2.5", "probability": 0.76}
    return None

def predict_corners():
    return {"market": "ТБ 8.5 угловых", "probability": 0.77}

def predict_cards():
    return {"market": "ТБ 4.5 ЖК", "probability": 0.79}

# ================= СТАТИСТИКА =================
def get_team_stats(team_name, last_n=10):
    try:
        search_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/searchteams.php"
        resp = requests.get(search_url, params={"t": team_name}, timeout=10).json()
        if not resp.get("teams"):
            return {"scored_avg": 1.5, "conceded_avg": 1.5}
        team_id = resp["teams"][0]["idTeam"]
        events_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/eventslast.php?id={team_id}"
        events = requests.get(events_url, timeout=10).json().get("results", [])
        scored = 0
        conceded = 0
        n = min(len(events), last_n)
        for e in events[:n]:
            home_team = e.get("strHomeTeam")
            away_team = e.get("strAwayTeam")
            home_score = int(e.get("intHomeScore") or 0)
            away_score = int(e.get("intAwayScore") or 0)
            if team_name == home_team:
                scored += home_score
                conceded += away_score
            else:
                scored += away_score
                conceded += home_score
        return {"scored_avg": scored/n, "conceded_avg": conceded/n}
    except Exception as ex:
        print("get_team_stats error:", ex)
        return {"scored_avg": 1.5, "conceded_avg": 1.5}

# ================= H2H =================
def get_h2h_probability(home, away, last_n=5):
    try:
        url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/eventsh2h.php"
        params = {"h": home, "a": away}
        events = requests.get(url, params=params, timeout=10).json().get("results", [])
        if not events:
            return 0.5
        home_wins = sum(1 for e in events[:last_n] if (e["strHomeTeam"]==home and int(e.get("intHomeScore") or 0) > int(e.get("intAwayScore") or 0)) or 
                                                  (e["strAwayTeam"]==home and int(e.get("intAwayScore") or 0) > int(e.get("intHomeScore") or 0)))
        return home_wins/last_n
    except Exception as ex:
        print("get_h2h_probability error:", ex)
        return 0.5

# ================= МОТИВАЦИЯ =================
def get_team_motivation(team_name, league_id):
    try:
        table_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/lookuptable.php"
        params = {"l": league_id, "s": "2025-2026"}
        table = requests.get(table_url, params=params, timeout=10).json().get("table", [])
        for team in table:
            if team_name.lower() == team.get("name").lower():
                pos = int(team.get("intRank") or 999)
                total = len(table)
                if pos <= 3 or pos >= total-2:
                    return 1.12
                elif pos in [4,5,6]:
                    return 1.08
                else:
                    return 1.0
    except Exception as ex:
        print("get_team_motivation error:", ex)
        return 1.0
    return 1.0

# ================= ТРАВМЫ =================
def get_injuries_factor(team_name):
    try:
        search_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/searchteams.php"
        team_data = requests.get(search_url, params={"t": team_name}, timeout=10).json()
        if not team_data.get("teams"):
            return 1.0
        team_id = team_data["teams"][0]["idTeam"]
        roster_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/lookup_all_players.php?id={team_id}"
        roster = requests.get(roster_url, timeout=10).json().get("player", [])
        factor = 1.0
        key_players = roster[:5]
        for player in key_players:
            status = (player.get("strStatus") or "").lower()
            if "injured" in status:
                factor *= 0.85
            elif "suspended" in status:
                factor *= 0.80
        return factor
    except Exception as ex:
        print("get_injuries_factor error:", ex)
        return 1.0

# ================= ПОГОДА =================
def get_weather_factor(city_name):
    if not WEATHER_API_KEY:
        return 1.0
    try:
        url = f"http://api.openweathermap.org/data/2.5/weather"
        params = {"q": city_name, "appid": WEATHER_API_KEY, "units":"metric"}
        resp = requests.get(url, params=params, timeout=10).json()
        weather = resp.get("weather", [{}])[0].get("main","Clear").lower()
        if weather in ["rain","snow","thunderstorm"]:
            return 0.9
        elif weather in ["clear","clouds"]:
            return 1.0
        return 1.0
    except Exception as ex:
        print("get_weather_factor error:", ex)
        return 1.0

# ================= OddsPapi =================
def get_real_odds(home, away):
    if not ODDS_API_KEY:
        return 1.85
    try:
        url = "https://api.oddspapi.io/v4/odds"
        params = {"sport":"soccer","region":"eu","mkt":"totals","apiKey":ODDS_API_KEY}
        resp = requests.get(url, params=params, timeout=10).json()
        for event in resp.get("data", []):
            if home.lower() in event.get("home_team","").lower() and away.lower() in event.get("away_team","").lower():
                odds_list = event.get("odds", [])
                if odds_list:
                    return float(odds_list[0].get("odd",1.85))
    except Exception as ex:
        print("get_real_odds error:", ex)
    return 1.85

# ================= МАТЧИ =================
def get_today_matches():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/eventsday.php"
    params = {"d": today, "s": "Soccer"}
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.text.strip() == "":
            return []
        data = response.json()
        events = data.get("events", [])
        matches = []
        for event in events:
            league_name = event.get("strLeague")
            if league_name in LEAGUES:
                home = event.get("strHomeTeam")
                away = event.get("strAwayTeam")
                league_id = LEAGUES[league_name]

                home_stats = get_team_stats(home)
                away_stats = get_team_stats(away)
                h2h_prob = get_h2h_probability(home, away)
                home_mot = get_team_motivation(home, league_id)
                away_mot = get_team_motivation(away, league_id)
                avg_goals = ((home_stats["scored_avg"] + away_stats["conceded_avg"])/2) * home_mot * away_mot
                odds = get_real_odds(home, away)
                weather_factor = get_weather_factor(event.get("strVenue","London"))
                injury_factor = (get_injuries_factor(home) + get_injuries_factor(away))/2

                matches.append({
                    "home": home,
                    "away": away,
                    "avg_goals": avg_goals,
                    "odds": odds,
                    "h2h_prob": h2h_prob,
                    "motivation_prob": (home_mot + away_mot)/2,
                    "weather_factor": weather_factor,
                    "injury_factor": injury_factor,
                    "dateEvent": event.get("dateEvent"),
                    "strTime": event.get("strTime")
                })
        print(f"DEBUG: {len(matches)} matches fetched")
        return matches
    except Exception as ex:
        print("get_today_matches error:", ex)
        return []

# ================= ОТПРАВКА СИГНАЛОВ =================
async def send_signals(app):
    if not CHAT_ID:
        print("ERROR: CHAT_ID is not set")
        return

    matches = get_today_matches()
    if not matches:
        await app.bot.send_message(chat_id=CHAT_ID, text="Сегодня подходящих матчей нет.")
        return

    message = "🐺 ЦЕРБЕР | СИГНАЛЫ (Value > 0)\n\n"
    found = False
    for match in matches:
        for pred in [predict_goals(match["avg_goals"]), predict_corners(), predict_cards()]:
            if pred:
                probability = compute_probability(
                    pred["probability"],
                    match["h2h_prob"],
                    match["motivation_prob"],
                    match["weather_factor"],
                    match["injury_factor"]
                )
                value = calculate_value(probability, match["odds"])
                try:
                    match_time_utc = datetime.strptime(match["dateEvent"] + " " + match["strTime"], "%Y-%m-%d %H:%M:%S")
                    match_time_msk = match_time_utc + timedelta(hours=3)
                    match_time_formatted = match_time_msk.strftime("%H:%M MSK")
                except:
                    match_time_formatted = "??:?? MSK"

                if value > 0:
                    low_prob_mark = " ⚠️ Низкая вероятность" if probability < MIN_PROBABILITY else ""
                    found = True
                    message += (
                        f"⚽ {match['home']} — {match['away']} ({match_time_formatted}){low_prob_mark}\n"
                        f"{pred['market']}\n"
                        f"Вероятность: {int(probability*100)}%\n"
                        f"Коэфф.: {match['odds']}\n"
                        f"Value: +{value:.2f}\n\n"
                    )

    if not found:
        message += "Сегодня нет value-сигналов."
    await app.bot.send_message(chat_id=CHAT_ID, text=message)

# ================= ДНЕВНАЯ ЗАДАЧА =================
async def daily_task(app):
    while True:
        now = datetime.utcnow()
        target_time = datetime.combine(now.date(), time(10,0))
        if now > target_time:
            target_time += timedelta(days=1)
        wait_seconds = (target_time - now).total_seconds()
        await asyncio.sleep(wait_seconds)
        await send_signals(app)

# ================= КОМАНДЫ =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🐺 ЦЕРБЕР активирован! Сигналы будут приходить ежедневно в 10:00 UTC.")

async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    print("DEBUG: /signals command called")
    await send_signals(context.application)

# ================= ЗАПУСК =================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals_command))
    loop = asyncio.get_event_loop()
    loop.create_task(daily_task(app))
    app.run_polling()

if __name__ == "__main__":
    main()


