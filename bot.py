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

TOP_LEAGUES = [
    "English Premier League",
    "Spanish La Liga",
    "Italian Serie A",
    "German Bundesliga",
    "Russian Premier League"
]

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
        return {
            "market": market,
            "probability": probability,
            "odds": odds,
            "value": value
        }
    return None


def predict_corners():
    probability = 0.77
    market = "ТБ 8.5 угловых"
    odds = 1.80
    value = calculate_value(probability, odds)
    if probability >= MIN_PROBABILITY and value >= MIN_VALUE:
        return {
            "market": market,
            "probability": probability,
            "odds": odds,
            "value": value
        }
    return None


def predict_cards():
    probability = 0.79
    market = "ТБ 4.5 ЖК"
    odds = 1.85
    value = calculate_value(probability, odds)
    if probability >= MIN_PROBABILITY and value >= MIN_VALUE:
        return {
            "market": market,
            "probability": probability,
            "odds": odds,
            "value": value
        }
    return None

# ================= СТАТИСТИКА КОМАНД =================
def get_team_stats(team_name):
    """
    Возвращает словарь:
    {
        'scored_avg': средние голы команды за последние 5 матчей,
        'conceded_avg': средние голы против команды
    }
    """
    # Получаем команду
    search_url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/searchteams.php"
    search_resp = requests.get(search_url, params={"t": team_name}, timeout=10)
    search_data = search_resp.json()
    if not search_data or not search_data.get("teams"):
        return {"scored_avg": 1.5, "conceded_avg": 1.5}

    team_id = search_data["teams"][0]["idTeam"]

    # Берём последние 5 матчей
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
        # Определяем домашнюю/гостевую роль
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
        league = event.get("strLeague")
        if league in TOP_LEAGUES:
            home = event.get("strHomeTeam")
            away = event.get("strAwayTeam")

            # рассчитываем реальные avg_goals на основе статистики команд
            home_stats = get_team_stats(home)
            away_stats = get_team_stats(away)
            avg_goals = (home_stats["scored_avg"] + away_stats["conceded_avg"]) / 2

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
