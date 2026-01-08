import os
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ====== НАСТРОЙКИ ======
BOT_TOKEN = os.getenv("BOT_TOKEN")

MIN_PROBABILITY = 0.75
MIN_VALUE = 0.05


# ====== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ======
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


# ====== КОМАНДЫ TELEGRAM ======
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован.\n\n"
        "Я публикую только СИЛЬНЫЕ сигналы:\n"
        "• вероятность от 75%\n"
        "• только value-события\n\n"
        "Команды:\n"
        "/signals — сигналы ЦЕРБЕРА"
    )


async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = "🐺 ЦЕРБЕР | СИГНАЛЫ (75%+)\n\n"

    # ТЕСТОВЫЙ МАТЧ (пока без API)
    match = {
        "home": "Arsenal",
        "away": "Tottenham",
        "avg_goals": 3.1
    }

    signals_found = False

    for sig in [
        predict_goals(match["avg_goals"]),
        predict_corners(),
        predict_cards()
    ]:
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
        message += "Сегодня сильных сигналов нет."

    await update.message.reply_text(message)


# ====== ЗАПУСК БОТА ======
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    # 🔗 РЕГИСТРАЦИЯ ХЭНДЛЕРОВ (ОЧЕНЬ ВАЖНО)
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))

    app.run_polling()


if __name__ == "__main__":
    main()


