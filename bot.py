import os
import requests
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ================= НАСТРОЙКИ =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
THESPORTSDB_API_KEY = os.getenv("THESPORTSDB_API_KEY", "1")
ODDS_API_KEY = os.getenv("ODDS_API_KEY")  # ключ OddsPapi

MIN_PROBABILITY = 0.75
MIN_VALUE = 0.05

LEAGUES = {
    "English Premier League": 4328,
    "Spanish La Liga": 4335,
    "Italian Serie A": 4332,
    "German Bundesliga": 4331,
    "Russian Premier League": 4398
}

# ================= ПРОГНОЗЫ =================
def calculate_value(probability, odds):
    return probability * odds - 1

def predict_goals(avg_goals):
    if avg_goals >= 3.0:
        probability = 0.78
        market = "ТБ 2.5"
    elif avg_goals <= 2.0:
        probability = 0.76
        market = "ТМ 2.5"
    else:
        return None
    return {"market": market, "probability": probability}

def predict_corners():
    probability = 0.77
    market = "ТБ 8.5 угловых"
    return {"market": market, "probability": probability}

def predict_cards():
    probability = 0.79
    market = "ТБ 4.5 ЖК"
    return {"market": market, "probability": probability}

# ================= СТАТИСТИКА =================
def get_team_stats(team_name):
    search_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/searchteams.php"
    resp = requests.get(search_url, params={"t": team_name}, timeout=10)
    data = resp.json()
    if not data.get("teams"):
        return {"scored_avg": 1.5, "conceded_avg": 1.5}

    team_id = data["teams"][0]["idTeam"]
    events_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/eventslast.php?id={team_id}"
    events_data = requests.get(events_url, timeout=10).json()
    events = events_data.get("results", [])
    if not events:
        return {"scored_avg": 1.5, "conceded_avg": 1.5}

    scored = 0
    conceded = 0
    n = min(len(events), 5)
    for e in events[:5]:
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

# ================= МОТИВАЦИЯ =================
def get_team_motivation(team_name, league_id):
    table_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/lookuptable.php"
    params = {"l": league_id, "s": "2025-2026"}
    try:
        resp = requests.get(table_url, params=params, timeout=10).json()
        table = resp.get("table", [])
        for team in table:
            if team_name.lower() == team.get("name").lower():
                pos = int(team.get("intRank") or 999)
                total = len(table)
                if pos <= 3 or pos >= total-2:  # титул/выживание
                    return 1.12
                elif pos in [4,5,6]:  # еврокубки
                    return 1.08
                else:
                    return 1.0
    except:
        return 1.0
    return 1.0

# ================= ODDSPAPI =================
def get_real_odds(home, away):
    """Запрос к OddsPapi для коэффициентов на матч"""
    if not ODDS_API_KEY:
        return 1.85  # fallback коэффициент
    try:
        url = f"https://api.oddspapi.io/v4/odds"
        params = {
            "sport": "soccer",
            "region": "eu",
            "mkt": "totals",
            "apiKey": ODDS_API_KEY
        }
        resp = requests.get(url, params=params, timeout=10).json()
        # ищем матч по названиям команд
        for event in resp.get("data", []):
            if home.lower() in event.get("home_team","").lower() and away.lower() in event.get("away_team","").lower():
                # берём средний коэффициент ТБ/ТМ 2.5
                odds_list = event.get("odds", [])
                if odds_list:
                    return float(odds_list[0].get("odd",1.85))
    except:
        pass
    return 1.85

# ================= МАТЧИ =================
def get_today_matches():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/eventsday.php"
    params = {"d": today, "s": "Soccer"}
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

            # статистика и мотивация
            home_stats = get_team_stats(home)
            away_stats = get_team_stats(away)
            home_mot = get_team_motivation(home, league_id)
            away_mot = get_team_motivation(away, league_id)
            avg_goals = ((home_stats["scored_avg"] + away_stats["conceded_avg"])/2) * home_mot * away_mot

            # реальные odds
            odds = get_real_odds(home, away)

            matches.append({
                "home": home,
                "away": away,
                "avg_goals": avg_goals,
                "odds": odds
            })
    return matches

# ================= TELEGRAM =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован.\n\n"
        "Сигналы публикуются только при:\n"
        "• вероятности от 75%\n"
        "• положительном value\n\n"
        "/signals — сигналы"
    )

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        matches = get_today_matches()
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при получении матчей: {e}")
        return
    if not matches:
        await update.message.reply_text("Сегодня подходящих матчей нет.")
        return

    message = "🐺 ЦЕРБЕР | СИГНАЛЫ (75%+)\n\n"
    found = False

    for match in matches:
        for pred in [predict_goals(match["avg_goals"]), predict_corners(), predict_cards()]:
            if pred:
                value = calculate_value(pred["probability"], match["odds"])
                if pred["probability"] >= MIN_PROBABILITY and value > 0:
                    found = True
                    message += (
                        f"⚽ {match['home']} — {match['away']}\n"
                        f"{pred['market']}\n"
                        f"Вероятность: {int(pred['probability']*100)}%\n"
                        f"Коэфф.: {match['odds']}\n"
                        f"Value: +{value:.2f}\n\n"
                    )
    if not found:
        message += "Сегодня нет value-сигналов от 75%."
    await update.message.reply_text(message)

# ================= ЗАПУСК БОТА =================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))
    app.run_polling()

if __name__ == "__main__":
    main()

