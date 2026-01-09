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

STATE_FILE = "state.json"

# Название рынка в API-Football для O/U 2.5 — иногда может отличаться.
# Оставляю дефолт, но если надо — можно переопределить переменной окружения.
ODDS_MARKET_OVER25 = os.getenv("ODDS_MARKET_OVER25", "Over/Under 2.5")

# ======================
# TARGET COMPETITIONS + ALIASES (умный поиск)
# ======================
TARGET_COMPETITIONS: List[Dict[str, Any]] = [
    # England
    {"country": "England", "name": "Premier League", "aliases": ["Premier League"]},
    {"country": "England", "name": "FA Cup", "aliases": ["FA Cup"]},
    {"country": "England", "name": "League Cup", "aliases": ["League Cup", "EFL Cup", "Carabao Cup"]},
    {"country": "England", "name": "Community Shield", "aliases": ["Community Shield", "FA Community Shield"]},

    # Germany
    {"country": "Germany", "name": "Bundesliga", "aliases": ["Bundesliga", "Bundesliga 1", "1. Bundesliga", "Bundesliga - 1"]},
    {"country": "Germany", "name": "DFB Pokal", "aliases": ["DFB Pokal", "DFB-Pokal", "German Cup"]},
    {"country": "Germany", "name": "Super Cup", "aliases": ["Super Cup", "DFL Supercup", "German Super Cup"]},

    # Spain
    {"country": "Spain", "name": "La Liga", "aliases": ["La Liga", "LaLiga", "Primera Division", "La Liga Santander"]},
    {"country": "Spain", "name": "Copa del Rey", "aliases": ["Copa del Rey", "Copa Del Rey", "King's Cup"]},
    {"country": "Spain", "name": "Super Cup", "aliases": ["Super Cup", "Supercopa", "Supercopa de Espana"]},

    # Italy
    {"country": "Italy", "name": "Serie A", "aliases": ["Serie A"]},
    {"country": "Italy", "name": "Coppa Italia", "aliases": ["Coppa Italia", "Italy Cup"]},
    {"country": "Italy", "name": "Super Cup", "aliases": ["Super Cup", "Supercoppa", "Supercoppa Italiana"]},

    # France
    {"country": "France", "name": "Ligue 1", "aliases": ["Ligue 1"]},
    {"country": "France", "name": "Coupe de France", "aliases": ["Coupe de France", "French Cup"]},
    {"country": "France", "name": "Super Cup", "aliases": ["Super Cup", "Trophee des Champions"]},

    # Russia
    {"country": "Russia", "name": "Premier League", "aliases": ["Premier League", "Russian Premier League", "Premier Liga"]},

    # UEFA
    {"country": None, "name": "UEFA Champions League", "aliases": ["UEFA Champions League", "Champions League", "UCL"]},
    {"country": None, "name": "UEFA Europa League", "aliases": ["UEFA Europa League", "Europa League", "UEL"]},
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

def calc_prob_over25(hs: Dict[str, float], as_: Dict[str, float], w_k: float) -> float:
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
    return min(max(prob, 0.05), 0.90)

# ======================
# ODDS (BEST COEF)
# ======================
def fetch_fixture_odds(fixture_id: int) -> list:
    try:
        r = requests.get(
            "https://v3.football.api-sports.io/odds",
            headers=HEADERS,
            params={"fixture": fixture_id},
            timeout=25
        )
        r.raise_for_status()
        data = r.json()
        if data.get("errors"):
            print("[API odds errors]", data["errors"])
        return data.get("response", []) or []
    except Exception as e:
        print(f"[ERROR] fetch_fixture_odds fixture={fixture_id}: {e}")
        return []

def extract_best_over25_odds(odds_response: list) -> Optional[float]:
    """
    Лучший (максимальный) коэффициент на Over 2.5 среди всех букмекеров.
    """
    if not odds_response:
        return None

    best = None
    market_name = ODDS_MARKET_OVER25.strip().lower()

    for item in odds_response:
        for bm in (item.get("bookmakers") or []):
            for b in (bm.get("bets") or []):
                bet_name = (b.get("name") or "").strip().lower()
                if bet_name != market_name:
                    continue

                for v in (b.get("values") or []):
                    label = (v.get("value") or "").strip().lower()
                    odd_str = v.get("odd")
                    try:
                        odd = float(odd_str)
                    except:
                        continue

                    # варианты подписей
                    if label in ["over 2.5", "over2.5", "o2.5", "over"]:
                        if best is None or odd > best:
                            best = odd

    return best

# ======================
# LEAGUES LOOKUP (SMART)
# ======================
def leagues_search(search_text: str, country: Optional[str]) -> List[Dict[str, Any]]:
    params = {"search": search_text}
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
    return data.get("response", [])

def score_candidate(target_country: Optional[str], aliases: List[str], cand_name: str, cand_country: str) -> int:
    tn = cand_name.strip().lower()
    tc = cand_country.strip().lower()
    score = 0

    if target_country:
        if tc == target_country.strip().lower():
            score += 50
        else:
            score -= 10

    for a in aliases:
        al = a.strip().lower()
        if tn == al:
            score += 100
        elif al in tn:
            score += 60
        elif tn in al:
            score += 40

    return score

def resolve_target_league_ids(force: bool = False) -> Tuple[Dict[str, int], List[str]]:
    cache = STATE.get("league_ids", {})
    missing_cache = STATE.get("league_missing", [])
    cache_date = STATE.get("league_ids_date")

    today = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d")
    if (not force) and cache and cache_date == today:
        return {k: int(v) for k, v in cache.items()}, list(missing_cache)

    resolved: Dict[str, int] = {}
    missing: List[str] = []

    for item in TARGET_COMPETITIONS:
        country = item["country"]
        name = item["name"]
        aliases = item.get("aliases") or [name]
        key = f"{(country or 'UEFA')}|{name}"

        best_id = None
        best_score = -999999
        best_debug = None

        for alias in aliases:
            for ctry in [country, None]:
                try:
                    results = leagues_search(alias, ctry)
                except Exception as e:
                    print(f"[ERROR] leagues_search {key} alias={alias} country={ctry}: {e}")
                    results = []

                for rr in results:
                    lg = rr.get("league", {}) or {}
                    cn = rr.get("country", {}) or {}
                    cand_name = (lg.get("name") or "").strip()
                    cand_country = (cn.get("name") or "").strip()
                    if not cand_name:
                        continue

                    sc = score_candidate(country, aliases, cand_name, cand_country)
                    if sc > best_score:
                        best_score = sc
                        best_id = lg.get("id")
                        best_debug = f"{cand_country}|{cand_name} id={best_id} score={sc}"

        if best_id:
            resolved[key] = int(best_id)
            print(f"[INFO] League resolved {key} -> {best_debug}")
        else:
            missing.append(key)
            print(f"[WARN] League NOT found: {key}")

    STATE["league_ids"] = {k: int(v) for k, v in resolved.items()}
    STATE["league_missing"] = missing
    STATE["league_ids_date"] = today
    save_state(STATE)

    return resolved, missing

# ======================
# FIXTURES FILTERED
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
                    filtered = [f for f in fixtures if (f.get("league", {}) or {}).get("id") in allowed_ids]
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

def chunked(text: str, limit: int = 3500) -> List[str]:
    parts, buf = [], ""
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
        "🐺 ЦЕРБЕР — odds включены (беру ЛУЧШИЙ коэф среди букмекеров).\n\n"
        "Команды:\n"
        "• /signals — матчи по выбранным турнирам + вероятность + лучший коэф + value\n"
        "• /reload_leagues — пересканировать турниры\n"
        "• /debug_league <страна|UEFA> <поиск>\n"
    )

async def reload_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)
    await context.bot.send_message(chat_id=chat_id, text="🔄 Пересканирую турниры…")

    resolved, missing = resolve_target_league_ids(force=True)

    text = "✅ Найденные турниры:\n"
    if resolved:
        for k in sorted(resolved.keys()):
            text += f"• {k} → ID {resolved[k]}\n"
    else:
        text += "• (пусто)\n"

    text += "\n❌ Не найдены:\n"
    if missing:
        for m in missing:
            text += f"• {m}\n"
    else:
        text += "• (все найдены)\n"

    for part in chunked(text):
        await context.bot.send_message(chat_id=chat_id, text=part)

async def debug_league(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)
    args = context.args
    if len(args) < 2:
        await context.bot.send_message(chat_id=chat_id, text="Используй: /debug_league <страна|UEFA> <поиск>")
        return

    c = args[0]
    q = " ".join(args[1:]).strip()
    country = None if c.strip().upper() == "UEFA" else c.strip()

    try:
        res1 = leagues_search(q, country)
        res2 = [] if country is None else leagues_search(q, None)
        combined = res1 + res2
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"Ошибка поиска лиг: {e}")
        return

    if not combined:
        await context.bot.send_message(chat_id=chat_id, text="Ничего не найдено.")
        return

    lines = [f"🔎 Результаты API для: country={country} search='{q}' (первые 10)\n\n"]
    shown = 0
    seen = set()
    for rr in combined:
        lg = rr.get("league", {}) or {}
        cn = rr.get("country", {}) or {}
        lid = lg.get("id")
        nm = (lg.get("name") or "").strip()
        cc = (cn.get("name") or "").strip()
        if not lid or not nm:
            continue
        key = (lid, nm, cc)
        if key in seen:
            continue
        seen.add(key)

        lines.append(f"• {cc}|{nm} → ID {lid}\n")
        shown += 1
        if shown >= 10:
            break

    for part in chunked("".join(lines)):
        await context.bot.send_message(chat_id=chat_id, text=part)

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)

    fixtures, league_ids, missing, used_date = get_today_matches_filtered()
    if not fixtures:
        msg = f"⚠️ На дату {used_date} матчей по выбранным турнирам не найдено.\n"
        msg += f"В кэше турниров: {len(league_ids)}\n"
        if missing:
            msg += "\n❌ Не найдены турниры:\n" + "\n".join(f"• {m}" for m in missing)
        msg += "\n\nПопробуй: /reload_leagues"
        await context.bot.send_message(chat_id=chat_id, text=msg)
        return

    out = [f"⚽ ЦЕРБЕР — матчи ({used_date})\nOdds: лучший коэф на ТБ2.5\n\n"]
    count = 0

    for f in fixtures:
        fixture_id = (f.get("fixture") or {}).get("id")
        home = (f.get("teams") or {}).get("home", {}) or {}
        away = (f.get("teams") or {}).get("away", {}) or {}
        league = f.get("league", {}) or {}
        league_name = league.get("name", "?")

        venue = (f.get("fixture") or {}).get("venue", {}) or {}
        city = venue.get("city")

        home_id = home.get("id")
        away_id = away.get("id")
        if not home_id or not away_id:
            continue

        hs = analyze_goals(get_last_matches(int(home_id), last=5))
        as_ = analyze_goals(get_last_matches(int(away_id), last=5))
        if not hs or not as_:
            continue

        prob = calc_prob_over25(hs, as_, weather_factor(city))

        ts = (f.get("fixture") or {}).get("timestamp")
        t_str = "—"
        if ts:
            time_msk = datetime.utcfromtimestamp(int(ts)) + timedelta(hours=3)
            t_str = time_msk.strftime("%H:%M МСК")

        # ODDS
        best_odds = None
        if fixture_id:
            odds_resp = fetch_fixture_odds(int(fixture_id))
            best_odds = extract_best_over25_odds(odds_resp)

        fair = round(1.0 / prob, 2) if prob > 0 else None

        if best_odds:
            value = (prob * best_odds) - 1.0
            odds_line = f"Book(best): {best_odds:.2f} | Fair: {fair:.2f} | Value: {value:+.2f}"
        else:
            odds_line = f"Book(best): нет линии | Fair: {fair:.2f}"

        out.append(
            f"🏆 {league_name}\n"
            f"{home.get('name','?')} — {away.get('name','?')}\n"
            f"🕒 {t_str}\n"
            f"ТБ2.5: {int(prob*100)}%\n"
            f"{odds_line}\n\n"
        )

        count += 1
        if count % 8 == 0:
            await context.bot.send_message(chat_id=chat_id, text="".join(out))
            out = ["(продолжение)\n\n"]

    if out:
        await context.bot.send_message(chat_id=chat_id, text="".join(out))

# ======================
# RUN
# ======================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))
    app.add_handler(CommandHandler("reload_leagues", reload_leagues))
    app.add_handler(CommandHandler("debug_league", debug_league))
    app.run_polling()

if __name__ == "__main__":
    main()
