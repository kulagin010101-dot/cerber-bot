import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ======================
# ENV
# ======================
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("FOOTBALL_API_KEY")
WEATHER_KEY = os.getenv("WEATHER_API_KEY")
DEFAULT_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # канал/чат для отправки (опционально)

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не задан в переменных окружения!")
if not API_KEY:
    raise ValueError("FOOTBALL_API_KEY не задан в переменных окружения!")

HEADERS = {"x-apisports-key": API_KEY}

SEASON = int(os.getenv("SEASON", "2025"))
MIN_PROB = float(os.getenv("MIN_PROB", "0.75"))

REF_FILE = "referees.json"
STATE_FILE = "state.json"

# Лиги: можно расширить позже, но для отладки можно оставить все матчи дня
# LEAGUES = {39:"EPL",140:"LaLiga",135:"SerieA",78:"Bundesliga",235:"RPL"}

TEAM_IDS_FOR_REF_UPDATE = [39, 140, 135, 78, 235]  # как раньше (ограниченно)

# ======================
# STATE (чтобы обновлять судей раз в сутки)
# ======================
def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_state(state: Dict[str, Any]) -> None:
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

STATE = load_state()

# ======================
# REFEREES DB
# ======================
def load_referees() -> Dict[str, Any]:
    if os.path.exists(REF_FILE):
        try:
            with open(REF_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

REFEREES: Dict[str, Any] = load_referees()

def save_referees() -> None:
    with open(REF_FILE, "w", encoding="utf-8") as f:
        json.dump(REFEREES, f, ensure_ascii=False, indent=2)

def referee_factor(name: Optional[str]) -> float:
    if not name:
        return 1.0
    ref = REFEREES.get(name)
    if not ref:
        return 1.0
    pen = float(ref.get("penalties_per_game", 0))
    # мягкая корректировка
    if pen >= 0.30:
        return 1.07
    if pen <= 0.18:
        return 0.94
    return 1.0

def fetch_team_fixtures(team_id: int, last: int = 20) -> list:
    try:
        r = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers=HEADERS,
            params={"team": team_id, "last": last, "season": SEASON},
            timeout=20
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except Exception as e:
        print(f"[ERROR] fetch_team_fixtures team={team_id}: {e}")
        return []

def update_referees_from_recent_matches() -> int:
    """
    Обновляет REFEREES на основе последних матчей команд из TEAM_IDS_FOR_REF_UPDATE.
    Возвращает число обновлённых/добавленных судей.
    """
    updated = 0

    # локальные аккумуляторы
    agg: Dict[str, Dict[str, float]] = {}

    for tid in TEAM_IDS_FOR_REF_UPDATE:
        fixtures = fetch_team_fixtures(tid, last=25)
        for f in fixtures:
            referee = f.get("fixture", {}).get("referee")
            if not referee:
                continue

            goals = f.get("goals", {})
            h, a = goals.get("home"), goals.get("away")
            if h is None or a is None:
                continue
            total_goals = float(h + a)

            # В API-Football пенальти в events не всегда доступны в fixtures list,
            # поэтому делаем максимально безопасно: если events нет — пенальти=0 (консервативно).
            penalties = 0
            for ev in f.get("events", []) or []:
                if ev.get("type") == "Penalty":
                    penalties += 1

            if referee not in agg:
                agg[referee] = {"matches": 0.0, "goals": 0.0, "pens": 0.0}

            agg[referee]["matches"] += 1.0
            agg[referee]["goals"] += total_goals
            agg[referee]["pens"] += float(penalties)

    for ref, s in agg.items():
        m = s["matches"]
        if m <= 0:
            continue
        new_pen = s["pens"] / m
        new_avg_goals = s["goals"] / m

        old = REFEREES.get(ref)
        if not old:
            REFEREES[ref] = {"penalties_per_game": round(new_pen, 3), "avg_goals": round(new_avg_goals, 3)}
            updated += 1
        else:
            # сглаживание, чтобы база не “прыгала”
            old_pen = float(old.get("penalties_per_game", 0))
            old_avg = float(old.get("avg_goals", 0))

            blended_pen = 0.7 * old_pen + 0.3 * new_pen
            blended_avg = 0.7 * old_avg + 0.3 * new_avg_goals

            REFEREES[ref] = {"penalties_per_game": round(blended_pen, 3), "avg_goals": round(blended_avg, 3)}
            updated += 1

    if updated:
        save_referees()

    return updated

def ensure_daily_referee_update() -> None:
    """
    Обновляем судей максимум раз в сутки, чтобы не упираться в лимиты API.
    """
    today = datetime.utcnow().strftime("%Y-%m-%d")
    last = STATE.get("ref_update_date")

    if last == today:
        return

    print(f"[INFO] Daily referee update started for {today} ...")
    n = update_referees_from_recent_matches()
    STATE["ref_update_date"] = today
    STATE["ref_update_count"] = int(n)
    save_state(STATE)
    print(f"[INFO] Daily referee update done. Updated refs: {n}")

# ======================
# WEATHER
# ======================
def weather_factor(city: Optional[str]) -> float:
    if not WEATHER_KEY or not city:
        return 1.0
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": WEATHER_KEY, "units": "metric"},
            timeout=15
        )
        r.raise_for_status()
        data = r.json()

        temp = float(data["main"]["temp"])
        wind = float(data["wind"]["speed"])
        rain = 0.0
        if isinstance(data.get("rain"), dict):
            rain = float(data["rain"].get("1h", 0.0))

        factor = 1.0
        if temp < 5 or temp > 28:
            factor *= 0.95
        if rain > 0:
            factor *= 0.95
        if wind > 8:
            factor *= 0.96

        return factor
    except Exception as e:
        print(f"[ERROR] weather_factor city={city}: {e}")
        return 1.0

# ======================
# GOALS MODEL
# ======================
def get_last_matches(team_id: int, last: int = 5) -> list:
    try:
        r = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers=HEADERS,
            params={"team": team_id, "last": last, "season": SEASON},
            timeout=20
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except Exception as e:
        print(f"[ERROR] get_last_matches team={team_id}: {e}")
        return []

def analyze_goals(matches: list) -> Optional[Dict[str, float]]:
    total_goals = 0.0
    btts_yes = 0.0
    over25 = 0.0
    n = 0.0

    for m in matches:
        goals = m.get("goals", {})
        h, a = goals.get("home"), goals.get("away")
        if h is None or a is None:
            continue
        h = float(h)
        a = float(a)
        n += 1.0
        total_goals += (h + a)
        if h > 0 and a > 0:
            btts_yes += 1.0
        if (h + a) > 2.0:
            over25 += 1.0

    if n == 0:
        return None

    return {
        "avg": total_goals / n,
        "btts": btts_yes / n,
        "over25": over25 / n
    }

def calc_prob_over25(hs: Dict[str, float], as_: Dict[str, float], w_k: float, r_k: float) -> float:
    # базовая оценка
    prob = 0.60
    base = (hs["avg"] + as_["avg"]) / 2.0

    if base >= 2.6:
        prob += 0.08
    if hs["over25"] >= 0.6:
        prob += 0.05
    if as_["over25"] >= 0.6:
        prob += 0.05
    if hs["btts"] >= 0.6 and as_["btts"] >= 0.6:
        prob += 0.04

    prob *= w_k
    prob *= r_k

    return min(max(prob, 0.05), 0.90)

# ======================
# TODAY FIXTURES
# ======================
def get_today_matches() -> list:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    try:
        r = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers=HEADERS,
            params={"date": today, "season": SEASON},
            timeout=25
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except Exception as e:
        print(f"[ERROR] get_today_matches: {e}")
        return []

# ======================
# TELEGRAM HELPERS
# ======================
def target_chat_id(update: Update) -> str:
    # если задан TELEGRAM_CHAT_ID — шлём туда (канал),
    # иначе — отвечаем туда, откуда пришла команда.
    return DEFAULT_CHAT_ID or str(update.effective_chat.id)

# ======================
# HANDLERS
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР (рынок ГОЛОВ)\n\n"
        "Команды:\n"
        "• /signals — показать матчи дня и вероятность ТБ2.5 (отладка)\n"
        "• /update_refs — принудительно обновить базу судей\n\n"
        f"Порог сигналов: {int(MIN_PROB*100)}%"
    )

async def update_refs(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)
    await context.bot.send_message(chat_id=chat_id, text="🧑‍⚖️ Обновляю базу судей…")
    try:
        n = update_referees_from_recent_matches()
        STATE["ref_update_date"] = datetime.utcnow().strftime("%Y-%m-%d")
        STATE["ref_update_count"] = int(n)
        save_state(STATE)
        await context.bot.send_message(chat_id=chat_id, text=f"✅ Судьи обновлены. Изменено записей: {n}")
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Ошибка обновления судей: {e}")

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)

    # 1) раз в сутки обновляем судей
    try:
        ensure_daily_referee_update()
    except Exception as e:
        # не падаем — просто логируем
        print(f"[ERROR] ensure_daily_referee_update: {e}")

    fixtures = get_today_matches()
    if not fixtures:
        await context.bot.send_message(chat_id=chat_id, text="⚠️ Сегодня матчей не найдено в API.")
        return

    # Отладочный вывод: ВСЕ матчи + вероятность ТБ2.5
    msg_parts = ["⚽ ЦЕРБЕР — матчи дня (вероятность ТБ 2.5)\n"]
    count_lines = 0

    for f in fixtures:
        home = f.get("teams", {}).get("home", {})
        away = f.get("teams", {}).get("away", {})
        venue = f.get("fixture", {}).get("venue", {}) or {}
        city = venue.get("city")
        referee = f.get("fixture", {}).get("referee")

        home_id = home.get("id")
        away_id = away.get("id")
        if not home_id or not away_id:
            continue

        hs = analyze_goals(get_last_matches(int(home_id), last=5))
        as_ = analyze_goals(get_last_matches(int(away_id), last=5))
        if not hs or not as_:
            continue

        w_k = weather_factor(city)
        r_k = referee_factor(referee)
        prob = calc_prob_over25(hs, as_, w_k, r_k)

        # время МСК
        ts = f.get("fixture", {}).get("timestamp")
        if ts:
            time_msk = datetime.utcfromtimestamp(int(ts)) + timedelta(hours=3)
            t_str = time_msk.strftime("%H:%M МСК")
        else:
            t_str = "—"

        line = (
            f"{home.get('name','?')} — {away.get('name','?')}\n"
            f"🕒 {t_str}\n"
            f"ТБ2.5: {int(prob*100)}%\n"
        )

        # чтобы не упереться в лимит телеги, бьём на куски
        msg_parts.append(line)
        msg_parts.append("\n")
        count_lines += 1

        if count_lines % 12 == 0:
            await context.bot.send_message(chat_id=chat_id, text="".join(msg_parts))
            msg_parts = ["(продолжение)\n\n"]

    # отправляем хвост
    if msg_parts:
        await context.bot.send_message(chat_id=chat_id, text="".join(msg_parts))

def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))
    app.add_handler(CommandHandler("update_refs", update_refs))
    app.run_polling()

if __name__ == "__main__":
    main()
