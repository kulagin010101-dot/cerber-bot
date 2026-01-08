import os
import requests
from datetime import datetime
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ================= НАСТРОЙКИ =================
BOT_TOKEN = os.getenv("BOT_TOKEN")
THESPORTSDB_API_KEY = os.getenv("THESPORTSDB_API_KEY", "1")

MIN_PROBABILITY = 0.75
MIN_VALUE = 0.05

# Лиги и их ID для TheSportsDB
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
        odds = 1.85
    elif avg_goals <= 2.0:
        probability = 0.76
        market = "ТМ 2.5"
        odds = 1.90
    else:
        return None

    value = calculate_value(probability, odds)
    if probability >= MIN_PROBABILITY and value >= MIN_VALUE:
        return {"market": market, "probability": probability, "odds": odds, "value": value}
    return None


def predict_corners():
    probability = 0.77
    market = "ТБ 8.5 угловых"
    odds = 1.80
    value = calculate_value(probability, odds)
    if probability >= MIN_PROBABILITY and value >= MIN_VALUE:
        return {"market": market, "probability": probability, "odds": odds, "value": value}
    return None


def predict_cards():
    probability = 0.79
    market = "ТБ 4.5 ЖК"
    odds = 1.85
    value = calculate_value(probability, odds)
    if probability >= MIN_PROBABILITY and value >= MIN_VALUE:
        return {"market": market, "probability": probability, "odds": odds, "value": value}
    return None

# ================= СТАТИСТИКА =================
def get_team_stats(team_name):
    """Последние 5 матчей команды"""
    search_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/searchteams.php"
    search_resp = requests.get(search_url, params={"t": team_name}, timeout=10)
    search_data = search_resp.json()
    if not search_data or not search_data.get("teams"):
        return {"scored_avg": 1.5, "conceded_avg": 1.5}

    team_id = search_data["teams"][0]["idTeam"]

    events_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/eventslast.php?id={team_id}"
    events_resp = requests.get(events_url, timeout=10)
    events_data = events_resp.json()
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

    return {"scored_avg": scored / n, "conceded_avg": conceded / n}

# ================= МОТИВАЦИЯ =================
def get_team_motivation(team_name, league_id):
    """Возвращает множитель мотивации команды от 1.0 до 1.15"""
    table_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/lookuptable.php"
    params = {"l": league_id, "s": "2025-2026"}
    try:
        resp = requests.get(table_url, params=params, timeout=10)
        data = resp.json()
        table = data.get("table", [])
        for team in table:
            if team_name.lower() == team.get("name").lower():
                position = int(team.get("intRank") or 999)
                total_teams = len(table)
                if position <= 3:  # борьба за титул
                    return 1.12
                elif position >= total_teams - 2:  # выживание
                    return 1.12
                elif position in [4,5,6]:  # еврокубки
                    return 1.08
                else:
                    return 1.0
    except:
        return 1.0
    return 1.0

# ================= МАТЧИ =================
def get_today_matches():
    today = datetime.utcnow().strftime("%Y-%m-%d")
    url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/eventsday.php"
    params = {"d": today, "s": "Soccer"}

    response = requests.get(url, params=params, timeout=10)
    if response.text.strip() == "":
        return []

    data = response.json()
    events = data.get("events")
    if not events:
        return []

    matches = []
    for event in events:
        league_name = event.get("strLeague")
        if league_name in LEAGUES:
            home = event.get("strHomeTeam")
            away = event.get("strAwayTeam")
            league_id = LEAGUES[league_name]

            # статистика команд
            home_stats = get_team_stats(home)
            away_stats = get_team_stats(away)

            # мотивация команд
            home_motivation = get_team_motivation(home, league_id)
            away_motivation = get_team_motivation(away, league_id)

            # рассчитываем avg_goals с мотивацией
            avg_goals = ((home_stats["scored_avg"] + away_stats["conceded_avg"]) / 2) * home_motivation * away_motivation

            matches.append({
                "home": home,
                "away": away,
                "avg_goals": avg_goals
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
        for sig in [predict_goals(match["avg_goals"]), predict_corners(), predict_cards()]:
            if sig:
                found = True
                message += (
                    f"⚽ {match['home']} — {match['away']}\n"
                    f"{sig['market']}\n"
                    f"Вероятность: {int(sig['probability']*100)}%\n"
                    f"Коэфф.: {sig['odds']}\n"
                    f"Value: +{sig['value']:.2f}\n\n"
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

