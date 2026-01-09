import os
import json
import requests
from datetime import datetime

API_KEY = os.getenv("FOOTBALL_API_KEY")
SEASON = 2025
HEADERS = {"x-apisports-key": API_KEY}
REF_FILE = "referees.json"

# Загружаем текущую базу судей
if os.path.exists(REF_FILE):
    with open(REF_FILE, "r", encoding="utf-8") as f:
        REFEREES = json.load(f)
else:
    REFEREES = {}

# Список всех команд, которые нужно проверить
# Можно вручную добавить топ-лиги
TEAM_IDS = [
    39, 140, 135, 78, 235  # Англия, Испания, Италия, Германия, Россия
]

def save_referees():
    with open(REF_FILE, "w", encoding="utf-8") as f:
        json.dump(REFEREES, f, ensure_ascii=False, indent=2)

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

def update_referees_db():
    for team_id in TEAM_IDS:
        fixtures = fetch_team_fixtures(team_id)
        for f in fixtures:
            referee = f["fixture"]["referee"]
            if not referee:
                continue

            # Инициализация
            if referee not in REFEREES:
                REFEREES[referee] = {"penalties_per_game": 0, "avg_goals": 0, "count": 0}

            pen = REFEREES[referee]["penalties_per_game"] * REFEREES[referee]["count"]
            goals = REFEREES[referee]["avg_goals"] * REFEREES[referee]["count"]
            count = REFEREES[referee]["count"]

            h, a = f["goals"]["home"], f["goals"]["away"]
            if h is None or a is None:
                continue
            total_goals = h + a

            # Подсчёт пенальти в событиях
            penalties = 0
            for e in f.get("events", []):
                if e.get("type") == "Penalty":
                    penalties += 1

            # Обновление средней статистики
            count += 1
            pen = (pen + penalties) / count
            goals = (goals + total_goals) / count

            REFEREES[referee] = {
                "penalties_per_game": pen,
                "avg_goals": goals,
                "count": count
            }

    # Убираем счётчик, оставляем чистую статистику
    for r in REFEREES:
        REFEREES[r].pop("count", None)

    save_referees()
    print(f"[{datetime.now()}] База судей обновлена. Всего судей: {len(REFEREES)}")

if __name__ == "__main__":
    update_referees_db()
