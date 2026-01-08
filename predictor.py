MIN_PROBABILITY = 0.75
MIN_VALUE = 0.05

def calculate_value(probability, odds):
    return probability * odds - 1


def predict_goals(avg_goals):
    """
    avg_goals — ожидаемый тотал матча
    """
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


def predict_cards(important=True):
    probability = 0.79 if important else 0.74
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
