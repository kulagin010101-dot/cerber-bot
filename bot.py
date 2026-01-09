import os
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ======================
# ENV
# ======================
BOT_TOKEN = os.getenv("BOT_TOKEN")
API_KEY = os.getenv("FOOTBALL_API_KEY")
WEATHER_KEY = os.getenv("WEATHER_API_KEY")
DEFAULT_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # опционально

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не задан!")
if not API_KEY:
    raise ValueError("FOOTBALL_API_KEY не задан!")

HEADERS = {"x-apisports-key": API_KEY}

SEASON = int(os.getenv("SEASON", "2025"))
MIN_PROB = float(os.getenv("MIN_PROB", "0.75"))

REF_FILE = "referees.json"
STATE_FILE = "state.json"

# ======================
# TARGET COMPETITIONS
# ======================
TARGET_COMPETITIONS: List[Dict[str, Optional[str]]] = [
    # England
    {"country": "England", "name": "Premier League"},
    {"country": "England", "name": "FA Cup"},
    {"country": "England", "name": "League Cup"},
    {"country": "England", "name": "Community Shield"},

    # Germany
    {"country": "Germany", "name": "Bundesliga"},
    {"country": "Germany", "name": "DFB Pokal"},
    {"country": "Germany", "name": "Super Cup"},

    # Spain
    {"country": "Spain", "name": "La Liga"},
    {"country": "Spain", "name": "Copa del Rey"},
    {"country": "Spain", "name": "Super Cup"},

    # Italy
    {"country": "Italy", "name": "Serie A"},
    {"country": "Italy", "name": "Coppa Italia"},
    {"country": "Italy", "name": "Super Cup"},

    # France
    {"country": "France", "name": "Ligue 1"},
    {"country": "France", "name": "Coupe de France"},
    {"country": "France", "name": "Super Cup"},

    # Russia
    {"country": "Russia", "name": "Premier League"},

    # UEFA
    {"country": None, "name": "UEFA Champions League"},
    {"country": None, "name": "UEFA Europa League"},
]

# ======================
# STATE
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
# REFEREES DB (оставлено как раньше; судья влияет только если есть в базе)
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

def referee_factor(name: Optional[str]) -> float:
    if not name:
        return 1.0
    ref = REFEREES.get(name)
    if not ref:
        return 1.0
    pen = float(ref.get("penalties_per_game", 0))
    if pen >= 0.30:
        return 1.07
    if pen <= 0.18:
        return 0.94
    return 1.0

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
    except:
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
            timeout=25
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", [])
    except:
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
# LEAGUES LOOKUP + CACHE
# ======================
def leagues_search(name: str, country: Optional[str]) -> List[Dict[str, Any]]:
    params = {"search": name}
    if country:
        params["country"] = country
    r = requests.get(
        "https://v3.football.api-sports.io/leagues",
        headers=HEADERS,
        params=params,
        timeout=25
    )
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        print("[API errors]", data["errors"])
    return data.get("response", [])

def resolve_target_league_ids(force: bool = False) -> Tuple[Dict[str, int], List[str]]:
    """
    Возвращает (resolved_ids, missing_keys)
    key = f"{country or 'UEFA'}|{name}"
    """
    cache = STATE.get("league_ids", {})
    cache_date = STATE.get("league_ids_date")

    today = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d")  # МСК-дата
    if (not force) and cache and cache_date == today:
        resolved = {k: int(v) for k, v in cache.items()}
        missing = STATE.get("league_missing", [])
        return resolved, missing

    resolved: Dict[str, int] = {}
    missing: List[str] = []

    for item in TARGET_COMPETITIONS:
        country = item["country"]
        name = item["name"]
        key = f"{(country or 'UEFA')}|{name}"

        results = []
        try:
            results = leagues_search(name=name, country=country)
        except Exception as e:
            print(f"[ERROR] leagues_search {key}: {e}")
            results = []

        best_id = None

        # 1) точное совпадение названия
        for rr in results:
            lg = rr.get("league", {})
            nm = (lg.get("name") or "").strip()
            if nm.lower() == name.lower():
                best_id = lg.get("id")
                break

        # 2) если нет точного — берём первый, но отметим это в логах
        if best_id is None and results:
            best_id = results[0].get("league", {}).get("id")
            got_name = (results[0].get("league", {}).get("name") or "?")
            got_country = (results[0].get("country", {}).get("name") or "?")
            print(f"[WARN] not exact match for {key}. Using first: {got_country}|{got_name} id={best_id}")

        if best_id:
            resolved[key] = int(best_id)
        else:
            missing.append(key)
            print(f"[WARN] League not found: {key}")

    STATE["league_ids"] = {k: int(v) for k, v in resolved.items()}
    STATE["league_ids_date"] = today
    STATE["league_missing"] = missing
    save_state(STATE)

    print(f"[INFO] Resolved leagues count={len(resolved)} missing={len(missing)}")
    return resolved, missing

# ======================
# FIXTURES (filter by league ids)
# ======================
def fetch_fixtures_by_date(date_str: str, use_season: bool) -> list:
    params = {"date": date_str}
    if use_season:
        params["season"] = SEASON

    r = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=HEADERS,
        params=params,
        timeout=25
    )
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        print("[API errors]", data["errors"])
    return data.get("response", [])

def get_today_matches_filtered() -> Tuple[List[Dict[str, Any]], Dict[str, int], List[str], str]:
    league_ids, missing = resolve_target_league_ids(force=False)
    allowed_ids = set(league_ids.values())

    date_utc = datetime.utcnow().strftime("%Y-%m-%d")
    date_msk = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d")

    for date_str in [date_msk, date_utc]:
        for use_season in [True, False]:
            try:
                fixtures = fetch_fixtures_by_date(date_str, use_season=use_season)
                if fixtures:
                    filtered = []
                    for f in fixtures:
                        lid = (f.get("league", {}) or {}).get("id")
                        if lid in allowed_ids:
                            filtered.append(f)
                    print(f"[INFO] Fixtures {date_str} use_season={use_season}: total={len(fixtures)} filtered={len(filtered)}")
                    if filtered:
                        return filtered, league_ids, missing, date_str
            except Exception as e:
                print(f"[ERROR] fixtures date={date_str} use_season={use_season}: {e}")

    return [], league_ids, missing, date_msk

# ======================
# TELEGRAM HELPERS
# ======================
def target_chat_id(update: Update) -> str:
    return DEFAULT_CHAT_ID or str(update.effective_chat.id)

def chunked_send(text: str, limit: int = 3500) -> List[str]:
    # Telegram limit ~4096, оставим запас
    parts = []
    buf = ""
    for line in text.splitlines(True):
        if len(buf) + len(line) > limit:
            parts.append(buf)
            buf = ""
        buf += line
    if buf:
        parts.append(buf)
    return parts

# ======================
# HANDLERS
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР (рынок ГОЛОВ) — фильтр турниров включён.\n\n"
        "Команды:\n"
        "• /signals — матчи дня (только выбранные турниры) + вероятность ТБ2.5\n"
        "• /reload_leagues — пересканировать турниры и показать что найдено/не найдено\n"
    )

async def reload_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)

    await context.bot.send_message(chat_id=chat_id, text="🔄 Пересканирую турниры (league IDs)…")
    resolved, missing = resolve_target_league_ids(force=True)

    lines = []
    lines.append("✅ Найденные турниры:\n")
    if resolved:
        # сортируем красиво
        for k in sorted(resolved.keys()):
            lines.append(f"• {k} → ID {resolved[k]}\n")
    else:
        lines.append("• (пусто)\n")

    lines.append("\n❌ Не найдены:\n")
    if missing:
        for k in missing:
            lines.append(f"• {k}\n")
    else:
        lines.append("• (все найдены)\n")

    text = "".join(lines)
    for part in chunked_send(text):
        await context.bot.send_message(chat_id=chat_id, text=part)

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)

    fixtures, league_ids, missing, used_date = get_today_matches_filtered()
    if not fixtures:
        msg = (
            f"⚠️ На дату {used_date} матчей по выбранным турнирам не найдено.\n\n"
            f"Сейчас в кэше турниров: {len(league_ids)}\n"
        )
        if missing:
            msg += "❌ Не найдены турниры:\n" + "\n".join(f"• {m}" for m in missing[:10])
            if len(missing) > 10:
                msg += f"\n…и ещё {len(missing)-10}"
        msg += "\n\nПопробуй: /reload_leagues"
        await context.bot.send_message(chat_id=chat_id, text=msg)
        return

    msg_parts = [f"⚽ ЦЕРБЕР — матчи ({used_date})\nТолько выбранные турниры.\n\n"]
    chunk = 0

    for f in fixtures:
        home = f.get("teams", {}).get("home", {})
        away = f.get("teams", {}).get("away", {})
        venue = f.get("fixture", {}).get("venue", {}) or {}
        city = venue.get("city")
        referee = f.get("fixture", {}).get("referee")

        league = f.get("league", {})
        league_name = league.get("name", "?")

        home_id = home.get("id")
        away_id = away.get("id")
        if not home_id or not away_id:
            continue

        hs = analyze_goals(get_last_matches(int(home_id), last=5))
        as_ = analyze_goals(get_last_matches(int(away_id), last=5))
        if not hs or not as_:
            continue

        prob = calc_prob_over25(hs, as_, weather_factor(city), referee_factor(referee))

        ts = f.get("fixture", {}).get("timestamp")
        t_str = "—"
        if ts:
            time_msk = datetime.utcfromtimestamp(int(ts)) + timedelta(hours=3)
            t_str = time_msk.strftime("%H:%M МСК")

        msg_parts.append(
            f"🏆 {league_name}\n"
            f"{home.get('name','?')} — {away.get('name','?')}\n"
            f"🕒 {t_str}\n"
            f"ТБ2.5: {int(prob*100)}%\n\n"
        )

        chunk += 1
        if chunk % 10 == 0:
            await context.bot.send_message(chat_id=chat_id, text="".join(msg_parts))
            msg_parts = ["(продолжение)\n\n"]

    if msg_parts:
        await context.bot.send_message(chat_id=chat_id, text="".join(msg_parts))

# ======================
# RUN
# ======================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))
    app.add_handler(CommandHandler("reload_leagues", reload_leagues))
    app.run_polling()

if __name__ == "__main__":
    main()

