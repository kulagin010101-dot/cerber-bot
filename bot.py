import os
import requests
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

# ================= ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ =================
def calculate_value(probability, odds):
    return probability * odds - 1


def predict_goals(avg_goals):
    if avg_goals >= 3.0:
        probability = 0.78
        market = "ТБ 2.5"
        odds = 1.85
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


def get_today_matches():
    url = f"https://www.thesportsdb.com/api/v1/json/{THESPORTSDB_API_KEY}/eventsday.php"
    params = {
        "d": "today",
        "s": "Soccer"
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    matches = []

    if not data or not data.get("events"):
        return matches

    for event in data["events"]:
        league = event.get("strLeague")
        if league in TOP_LEAGUES:
            matches.append({
                "home": event.get("strHomeTeam"),
                "away": event.get("strAwayTeam"),
                # ВРЕМЕННО: средний тотал (улучшим на шаге 2)
                "avg_goals": 2.8
            })

    return matches


# ================= КОМАНДЫ TELEGRAM =================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован.\n\n"
        "Я публикую только сильные сигналы:\n"
        "• вероятность от 75%\n"
        "• только value-события\n\n"
        "Команды:\n"
        "/signals — сигналы ЦЕРБЕРА"
    )


async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        matches = get_today_matches()
    except Exception as e:
        await update.message.reply_text(f"❌ Ошибка при получении матчей: {e}")
        return

    if not matches:
        await update.message.reply_text("Сегодня матчей не найдено.")
        return

    message = "🐺 ЦЕРБЕР | СИГНАЛЫ (75%+)\n\n"
    signals_found = False

    for match in matches:
        signals = [
            predict_goals(match["avg_goals"]),
            predict_corners(),
            predict_cards()
        ]

        for sig in signals:
            if sig:
                signals_found = True
                message += (
                    f"⚽ {match['home']} — {match['away']}\n"
                    f"Рынок: {sig['market']}\n"
                    f"Вероятность: {int(sig['probability'] * 100)}%\n"
                    f"Коэффициент: {sig['odds']}\n"
                    f"Value: +{sig['value']:.2f}\n\n"
                )

    if not signals_found:
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

