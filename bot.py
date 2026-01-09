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
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
ODDSPAPI_KEY = os.getenv("ODDSPAPI_KEY")  # <-- odds source (oddspapi.io)
WEATHER_KEY = os.getenv("WEATHER_API_KEY")  # optional
DEFAULT_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # optional

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не задан!")
if not FOOTBALL_API_KEY:
    raise ValueError("FOOTBALL_API_KEY не задан!")
if not ODDSPAPI_KEY:
    raise ValueError("ODDSPAPI_KEY не задан! Добавь его в Railway Variables.")

HEADERS_FOOTBALL = {"x-apisports-key": FOOTBALL_API_KEY}

SEASON = int(os.getenv("SEASON", "2025"))
MIN_PROB = float(os.getenv("MIN_PROB", "0.75"))
MIN_VALUE = float(os.getenv("MIN_VALUE", "0.00"))

# sanity: чтобы не улетали неадекватные цифры
OU_ODDS_MIN = float(os.getenv("OU_ODDS_MIN", "1.10"))
OU_ODDS_MAX = float(os.getenv("OU_ODDS_MAX", "6.00"))
BTTS_ODDS_MIN = float(os.getenv("BTTS_ODDS_MIN", "1.10"))
BTTS_ODDS_MAX = float(os.getenv("BTTS_ODDS_MAX", "6.00"))

STATE_FILE = "state.json"

PIN_BOOKMAKER_SLUG = "pinnacle"  # oddsPapi bookmaker slug

# ======================
# TARGET COMPETITIONS (для API-Football)
# ======================
TARGET_COMPETITIONS: List[Dict[str, Any]] = [
    {"country": "England", "name": "Premier League", "aliases": ["Premier League"]},
    {"country": "England", "name": "FA Cup", "aliases": ["FA Cup"]},
    {"country": "England", "name": "League Cup", "aliases": ["League Cup", "EFL Cup", "Carabao Cup"]},
    {"country": "England", "name": "Community Shield", "aliases": ["Community Shield", "FA Community Shield"]},

    {"country": "Germany", "name": "Bundesliga", "aliases": ["Bundesliga", "Bundesliga 1", "1. Bundesliga", "Bundesliga - 1"]},
    {"country": "Germany", "name": "DFB Pokal", "aliases": ["DFB Pokal", "DFB-Pokal", "German Cup"]},
    {"country": "Germany", "name": "Super Cup", "aliases": ["Super Cup", "DFL Supercup", "German Super Cup"]},

    {"country": "Spain", "name": "La Liga", "aliases": ["La Liga", "LaLiga", "Primera Division", "La Liga Santander"]},
    {"country": "Spain", "name": "Copa del Rey", "aliases": ["Copa del Rey", "Copa Del Rey", "King's Cup"]},
    {"country": "Spain", "name": "Super Cup", "aliases": ["Super Cup", "Supercopa", "Supercopa de Espana"]},

    {"country": "Italy", "name": "Serie A", "aliases": ["Serie A"]},
    {"country": "Italy", "name": "Coppa Italia", "aliases": ["Coppa Italia", "Italy Cup"]},
    {"country": "Italy", "name": "Super Cup", "aliases": ["Super Cup", "Supercoppa", "Supercoppa Italiana"]},

    {"country": "France", "name": "Ligue 1", "aliases": ["Ligue 1"]},
    {"country": "France", "name": "Coupe de France", "aliases": ["Coupe de France", "French Cup"]},
    {"country": "France", "name": "Super Cup", "aliases": ["Super Cup", "Trophee des Champions"]},

    {"country": "Russia", "name": "Premier League", "aliases": ["Premier League", "Russian Premier League", "Premier Liga"]},

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
# HELPERS
# ======================
def norm(s: Any) -> str:
    return str(s or "").strip().lower()

def clamp(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return min(max(x, lo), hi)

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

def safe_team_key(name: str) -> str:
    # лёгкая нормализация, чтобы матчить названия между API-Football и OddsPapi
    s = norm(name)
    repl = {
        "&": "and",
        "fc ": "",
        " fc": "",
        "cf ": "",
        " cf": "",
        ".": "",
        ",": "",
        "’": "'",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    s = " ".join(s.split())
    return s

# ======================
# WEATHER (optional)
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
# MODEL (goals)
# ======================
def get_last_matches(team_id: int, last: int = 5) -> list:
    try:
        r = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers=HEADERS_FOOTBALL,
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

    return {"avg": total_goals / n, "btts": btts_yes / n, "over25": over25 / n}

def prob_over25(hs: Dict[str, float], as_: Dict[str, float], w_k: float) -> float:
    p = 0.60
    base = (hs["avg"] + as_["avg"]) / 2.0
    if base >= 2.6:
        p += 0.08
    if hs["over25"] >= 0.6:
        p += 0.05
    if as_["over25"] >= 0.6:
        p += 0.05
    if hs["btts"] >= 0.6 and as_["btts"] >= 0.6:
        p += 0.04
    p *= w_k
    return clamp(p, 0.05, 0.90)

def prob_btts_yes(hs: Dict[str, float], as_: Dict[str, float], w_k: float) -> float:
    p = 0.50
    p += 0.25 * (hs["btts"] - 0.5)
    p += 0.25 * (as_["btts"] - 0.5)
    p += 0.05 * (((hs["avg"] + as_["avg"]) / 2.0) - 2.4)
    p *= w_k
    return clamp(p, 0.05, 0.90)

def fair_odds(p: float) -> float:
    return round(1.0 / max(p, 1e-9), 2)

def value_ev(p: float, book_odds: float) -> float:
    return (p * book_odds) - 1.0

# ======================
# API-Football leagues resolve (как было)
# ======================
def leagues_search(search_text: str, country: Optional[str]) -> List[Dict[str, Any]]:
    params = {"search": search_text}
    if country:
        params["country"] = country
    r = requests.get(
        "https://v3.football.api-sports.io/leagues",
        headers=HEADERS_FOOTBALL,
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

        for alias in aliases:
            for ctry in [country, None]:
                try:
                    results = leagues_search(alias, ctry)
                except:
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

        if best_id:
            resolved[key] = int(best_id)
        else:
            missing.append(key)

    STATE["league_ids"] = {k: int(v) for k, v in resolved.items()}
    STATE["league_missing"] = missing
    STATE["league_ids_date"] = today
    save_state(STATE)

    return resolved, missing

def fetch_fixtures_by_date(date_str: str, use_season: bool) -> list:
    params = {"date": date_str}
    if use_season:
        params["season"] = SEASON
    r = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=HEADERS_FOOTBALL,
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
# OddsPapi (oddspapi.io) — tournaments resolve + odds cache
# ======================
ODDSPAPI_TOURN_CACHE_KEY = "oddspapi_tournaments"
ODDSPAPI_TOURN_CACHE_DATE = "oddspapi_tournaments_date"

def oddspapi_get(path: str, params: Dict[str, Any]) -> Any:
    # Oddspapi expects apiKey as query param in examples
    params = dict(params)
    params["apiKey"] = ODDSPAPI_KEY
    url = f"https://api.oddspapi.io{path}"
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()

def resolve_oddspapi_tournament_ids(force: bool = False) -> Dict[str, int]:
    """
    Для устойчивости мы берём ВСЕ турниры soccer (sportId=10)
    и подбираем tournamentId по (categoryName + tournamentName) с алиасами.
    """
    today = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d")
    if not force and STATE.get(ODDSPAPI_TOURN_CACHE_DATE) == today and STATE.get(ODDSPAPI_TOURN_CACHE_KEY):
        return {k: int(v) for k, v in STATE[ODDSPAPI_TOURN_CACHE_KEY].items()}

    # sportId=10 = Soccer в примерах документации
    data = oddspapi_get("/v4/tournaments", {"sportId": 10})
    out: Dict[str, int] = {}

    # карта нужных турниров по OddsPapi (они могут называться чуть иначе, поэтому алиасы)
    wanted = [
        ("England", ["Premier League", "FA Cup", "League Cup", "EFL Cup", "Carabao Cup", "Community Shield"]),
        ("Germany", ["Bundesliga", "DFB Pokal", "DFB-Pokal", "Super Cup", "DFL Supercup"]),
        ("Spain", ["LaLiga", "La Liga", "Copa del Rey", "Super Cup", "Supercopa"]),
        ("Italy", ["Serie A", "Coppa Italia", "Super Cup", "Supercoppa"]),
        ("France", ["Ligue 1", "Coupe de France", "Super Cup", "Trophee des Champions"]),
        ("Russia", ["Premier League", "Russian Premier League"]),
        (None, ["UEFA Champions League", "Champions League", "UEFA Europa League", "Europa League"]),
    ]

    for t in data:
        cat = (t.get("categoryName") or "").strip()
        name = (t.get("tournamentName") or "").strip()
        tid = t.get("tournamentId")
        if not tid or not name:
            continue

        for wcat, aliases in wanted:
            if wcat and norm(cat) != norm(wcat):
                continue
            for a in aliases:
                if norm(name) == norm(a) or norm(a) in norm(name) or norm(name) in norm(a):
                    key = f"{cat or 'UEFA'}|{name}"
                    out[key] = int(tid)

    STATE[ODDSPAPI_TOURN_CACHE_KEY] = {k: int(v) for k, v in out.items()}
    STATE[ODDSPAPI_TOURN_CACHE_DATE] = today
    save_state(STATE)
    return out

def fetch_oddspapi_odds_by_tournaments(tournament_ids: List[int]) -> List[Dict[str, Any]]:
    ids = ",".join(str(x) for x in tournament_ids)
    # oddsFormat=decimal, bookmaker=pinnacle as in docs
    return oddspapi_get(
        "/v4/odds-by-tournaments",
        {"tournamentIds": ids, "bookmaker": PIN_BOOKMAKER_SLUG, "oddsFormat": "decimal", "verbosity": 2},
    )

def parse_pinnacle_from_oddspapi_item(item: Dict[str, Any]) -> Dict[str, Optional[float]]:
    """
    Возвращает odds: O25, U25, BTTS_Y, BTTS_N
    В OddsPapi цены лежат в bookmakerOdds[pinnacle].markets[*].outcomes[*].players[0].price
    Для тоталов ориентируемся по bookmakerOutcomeId: '2.5/over' и '2.5/under' (пример в docs).
    """
    out = {"O25": None, "U25": None, "BTTS_Y": None, "BTTS_N": None}
    bm = ((item.get("bookmakerOdds") or {}).get(PIN_BOOKMAKER_SLUG) or {})
    markets = bm.get("markets") or {}

    for _mid, m in markets.items():
        outcomes = m.get("outcomes") or {}
        for _oid, o in outcomes.items():
            players = o.get("players") or {}
            p0 = players.get("0") or players.get(0) or {}
            price = p0.get("price")
            bok = norm(p0.get("bookmakerOutcomeId"))

            try:
                price = float(price)
            except:
                continue

            # totals 2.5
            if "2.5/over" in bok:
                if OU_ODDS_MIN <= price <= OU_ODDS_MAX:
                    out["O25"] = price
            elif "2.5/under" in bok:
                if OU_ODDS_MIN <= price <= OU_ODDS_MAX:
                    out["U25"] = price

            # BTTS
            # marketId=104 is "Both Teams To Score" in markets list docs
            # but here we just match by bookmakerOutcomeId yes/no
            if bok in ("yes", "btts_yes", "btts/yes"):
                if BTTS_ODDS_MIN <= price <= BTTS_ODDS_MAX:
                    out["BTTS_Y"] = price
            elif bok in ("no", "btts_no", "btts/no"):
                if BTTS_ODDS_MIN <= price <= BTTS_ODDS_MAX:
                    out["BTTS_N"] = price

    return out

def build_oddspapi_fixture_map_for_day(date_msk: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """
    Возвращает map для матчинга:
      key = (date_msk, team1_norm, team2_norm)  -> item
    На всякий случай добавляем оба порядка команд.
    """
    tourn_map = resolve_oddspapi_tournament_ids(force=False)
    tourn_ids = list(set(tourn_map.values()))
    items = fetch_oddspapi_odds_by_tournaments(tourn_ids)

    m: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

    for it in items:
        start = it.get("startTime")
        t1 = it.get("participant1Name") or ""
        t2 = it.get("participant2Name") or ""
        if not start or not t1 or not t2:
            continue

        # startTime is UTC Z. Convert to MSK date and time.
        try:
            dt_utc = datetime.fromisoformat(start.replace("Z", "+00:00"))
        except:
            continue
        dt_msk = dt_utc + timedelta(hours=3)
        d_msk = dt_msk.strftime("%Y-%m-%d")

        if d_msk != date_msk:
            continue

        k1 = safe_team_key(t1)
        k2 = safe_team_key(t2)
        m[(d_msk, k1, k2)] = it
        m[(d_msk, k2, k1)] = it

    return m

# ======================
# HANDLERS
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР — Pinnacle odds via OddsPapi.\n\n"
        "Команды:\n"
        "• /signals — только value-сигналы (P≥75% & Value>0)\n"
        "• /all — все матчи с P и Pinnacle odds\n"
        "• /check — диагностика (есть ли odds от OddsPapi)\n"
        "• /reload_leagues — пересканировать лиги (API-Football)\n"
        "• /odds_debug — показать RAW odds по первому матчу\n\n"
        f"Пороги: P ≥ {int(MIN_PROB*100)}%, Value > {MIN_VALUE:+.2f}\n"
        "Odds-источник: oddspapi.io\n"
    )

async def reload_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)
    await context.bot.send_message(chat_id=chat_id, text="🔄 Пересканирую лиги API-Football…")

    resolved, missing = resolve_target_league_ids(force=True)
    text = "✅ Найденные лиги:\n"
    for k in sorted(resolved.keys()):
        text += f"• {k} → ID {resolved[k]}\n"
    text += "\n❌ Не найдены:\n"
    if missing:
        for m in missing:
            text += f"• {m}\n"
    else:
        text += "• (все найдены)\n"

    for part in chunked(text):
        await context.bot.send_message(chat_id=chat_id, text=part)

async def check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)

    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ На дату {used_date} матчей нет (API-Football).")
        return

    # строим карту oddsPapi на этот день (один запрос)
    try:
        odds_map = build_oddspapi_fixture_map_for_day(used_date)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ OddsPapi ошибка: {e}")
        return

    sample = fixtures[:15]
    matched = 0
    with_any = 0

    for f in sample:
        home = ((f.get("teams") or {}).get("home") or {}).get("name", "")
        away = ((f.get("teams") or {}).get("away") or {}).get("name", "")
        key = (used_date, safe_team_key(home), safe_team_key(away))
        it = odds_map.get(key)
        if it:
            matched += 1
            odds = parse_pinnacle_from_oddspapi_item(it)
            if any(odds.values()):
                with_any += 1

    msg = (
        f"🔎 CHECK ({used_date})\n"
        f"Матчей (после фильтра лиг): {len(fixtures)}\n"
        f"Проверено: {len(sample)}\n"
        f"Сматчено по командам (OddsPapi): {matched}/{len(sample)}\n"
        f"Сматчено и есть рынки (O/U2.5 или BTTS): {with_any}/{len(sample)}\n"
    )
    await context.bot.send_message(chat_id=chat_id, text=msg)

async def odds_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)

    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ На дату {used_date} матчей нет.")
        return

    try:
        odds_map = build_oddspapi_fixture_map_for_day(used_date)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ OddsPapi ошибка: {e}")
        return

    f = fixtures[0]
    home = ((f.get("teams") or {}).get("home") or {}).get("name", "")
    away = ((f.get("teams") or {}).get("away") or {}).get("name", "")
    key = (used_date, safe_team_key(home), safe_team_key(away))
    it = odds_map.get(key)

    head = f"🧪 ODDS DEBUG ({used_date})\n{home} — {away}\n\n"
    if not it:
        await context.bot.send_message(chat_id=chat_id, text=head + "❌ Не нашёл соответствие в OddsPapi по названиям команд.")
        return

    odds = parse_pinnacle_from_oddspapi_item(it)
    bm = ((it.get("bookmakerOdds") or {}).get(PIN_BOOKMAKER_SLUG) or {})
    markets = bm.get("markets") or {}

    lines = [head]
    lines.append(f"✅ OddsPapi fixtureId: {it.get('fixtureId')}\n")
    lines.append(f"startTime (UTC): {it.get('startTime')}\n")
    lines.append(f"Pinnacle markets count: {len(markets)}\n\n")
    lines.append("🎯 Extracted:\n")
    lines.append(f"Over 2.5: {odds['O25']}\n")
    lines.append(f"Under 2.5: {odds['U25']}\n")
    lines.append(f"BTTS Yes: {odds['BTTS_Y']}\n")
    lines.append(f"BTTS No: {odds['BTTS_N']}\n\n")

    # выводим несколько outcomeId/bookmakerOutcomeId/price (RAW)
    lines.append("🔍 RAW sample (до 30 строк):\n")
    shown = 0
    for mid, m in markets.items():
        outcomes = m.get("outcomes") or {}
        for oid, o in outcomes.items():
            players = o.get("players") or {}
            p0 = players.get("0") or players.get(0) or {}
            bok = p0.get("bookmakerOutcomeId")
            price = p0.get("price")
            if bok is None or price is None:
                continue
            lines.append(f"m={mid} o={oid} outcome={bok} price={price}\n")
            shown += 1
            if shown >= 30:
                break
        if shown >= 30:
            break

    for part in chunked("".join(lines)):
        await context.bot.send_message(chat_id=chat_id, text=part)

async def all_matches(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ На дату {used_date} матчей нет.")
        return

    try:
        odds_map = build_oddspapi_fixture_map_for_day(used_date)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ OddsPapi ошибка: {e}")
        return

    out = [f"📋 ALL ({used_date}) — odds: Pinnacle via OddsPapi\n\n"]
    sent = 0

    for f in fixtures[:30]:
        fixture = f.get("fixture") or {}
        home = (f.get("teams") or {}).get("home", {}) or {}
        away = (f.get("teams") or {}).get("away", {}) or {}
        league = f.get("league", {}) or {}
        venue = fixture.get("venue", {}) or {}
        city = venue.get("city")

        home_id = home.get("id")
        away_id = away.get("id")
        if not home_id or not away_id:
            continue

        hs = analyze_goals(get_last_matches(int(home_id), last=5))
        as_ = analyze_goals(get_last_matches(int(away_id), last=5))
        if not hs or not as_:
            continue

        w_k = weather_factor(city)
        p_o25 = prob_over25(hs, as_, w_k)
        p_u25 = clamp(1.0 - p_o25, 0.05, 0.90)
        p_btts_y = prob_btts_yes(hs, as_, w_k)
        p_btts_n = clamp(1.0 - p_btts_y, 0.05, 0.90)

        ts = fixture.get("timestamp")
        t_str = "—"
        if ts:
            time_msk = datetime.utcfromtimestamp(int(ts)) + timedelta(hours=3)
            t_str = time_msk.strftime("%H:%M МСК")

        key = (used_date, safe_team_key(home.get("name", "")), safe_team_key(away.get("name", "")))
        it = odds_map.get(key)
        odds = {"O25": None, "U25": None, "BTTS_Y": None, "BTTS_N": None}
        if it:
            odds = parse_pinnacle_from_oddspapi_item(it)

        def line(title: str, p: float, book: Optional[float]) -> str:
            if book is None:
                return f"{title}: P={int(p*100)}% | Book=—"
            v = value_ev(p, book)
            return f"{title}: P={int(p*100)}% | Book={book:.2f} | Value={v:+.2f}"

        out.append(
            f"🏆 {league.get('name','?')}\n"
            f"{home.get('name','?')} — {away.get('name','?')} | 🕒 {t_str}\n"
            f"{line('ТБ2.5', p_o25, odds['O25'])}\n"
            f"{line('ТМ2.5', p_u25, odds['U25'])}\n"
            f"{line('ОЗ Да', p_btts_y, odds['BTTS_Y'])}\n"
            f"{line('ОЗ Нет', p_btts_n, odds['BTTS_N'])}\n\n"
        )

        sent += 1
        if sent % 6 == 0:
            await context.bot.send_message(chat_id=chat_id, text="".join(out))
            out = ["(продолжение)\n\n"]

    if out:
        await context.bot.send_message(chat_id=chat_id, text="".join(out))

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ На дату {used_date} матчей нет.")
        return

    try:
        odds_map = build_oddspapi_fixture_map_for_day(used_date)
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ OddsPapi ошибка: {e}")
        return

    out = [f"⚽ ЦЕРБЕР — value-сигналы ({used_date})\nOdds: Pinnacle via OddsPapi\n\n"]
    found_any = 0
    count = 0

    for f in fixtures:
        fixture = f.get("fixture") or {}
        home = (f.get("teams") or {}).get("home", {}) or {}
        away = (f.get("teams") or {}).get("away", {}) or {}
        league = f.get("league", {}) or {}
        venue = fixture.get("venue", {}) or {}
        city = venue.get("city")

        home_id = home.get("id")
        away_id = away.get("id")
        if not home_id or not away_id:
            continue

        key = (used_date, safe_team_key(home.get("name", "")), safe_team_key(away.get("name", "")))
        it = odds_map.get(key)
        if not it:
            continue
        odds = parse_pinnacle_from_oddspapi_item(it)
        if not any(odds.values()):
            continue

        hs = analyze_goals(get_last_matches(int(home_id), last=5))
        as_ = analyze_goals(get_last_matches(int(away_id), last=5))
        if not hs or not as_:
            continue

        w_k = weather_factor(city)
        p_o25 = prob_over25(hs, as_, w_k)
        p_u25 = clamp(1.0 - p_o25, 0.05, 0.90)
        p_btts_y = prob_btts_yes(hs, as_, w_k)
        p_btts_n = clamp(1.0 - p_btts_y, 0.05, 0.90)

        ts = fixture.get("timestamp")
        t_str = "—"
        if ts:
            time_msk = datetime.utcfromtimestamp(int(ts)) + timedelta(hours=3)
            t_str = time_msk.strftime("%H:%M МСК")

        def market_line(title: str, p: float, book: Optional[float]) -> Optional[str]:
            if book is None:
                return None
            if p < MIN_PROB:
                return None
            v = value_ev(p, book)
            if v <= MIN_VALUE:
                return None
            return f"{title}: P={int(p*100)}% | Book={book:.2f} | Fair={fair_odds(p):.2f} | Value={v:+.2f}"

        lines = []
        for title, p, book in [
            ("ТБ 2.5", p_o25, odds["O25"]),
            ("ТМ 2.5", p_u25, odds["U25"]),
            ("ОЗ Да", p_btts_y, odds["BTTS_Y"]),
            ("ОЗ Нет", p_btts_n, odds["BTTS_N"]),
        ]:
            ml = market_line(title, p, book)
            if ml:
                lines.append(ml)

        if not lines:
            continue

        found_any += 1
        out.append(
            f"🏆 {league.get('name','?')}\n"
            f"{home.get('name','?')} — {away.get('name','?')}\n"
            f"🕒 {t_str}\n" +
            "\n".join(lines) +
            "\n\n"
        )

        count += 1
        if count % 6 == 0:
            await context.bot.send_message(chat_id=chat_id, text="".join(out))
            out = ["(продолжение)\n\n"]

    if found_any == 0:
        await context.bot.send_message(
            chat_id=chat_id,
            text=(
                f"📭 На дату {used_date} нет value-сигналов при порогах:\n"
                f"P ≥ {int(MIN_PROB*100)}% и Value > {MIN_VALUE:+.2f}\n\n"
                "Проверка:\n"
                "• /check\n"
                "• /all\n"
                "• /odds_debug\n"
            )
        )
        return

    if out:
        await context.bot.send_message(chat_id=chat_id, text="".join(out))

# ======================
# RUN
# ======================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))
    app.add_handler(CommandHandler("all", all_matches))
    app.add_handler(CommandHandler("check", check))
    app.add_handler(CommandHandler("reload_leagues", reload_leagues))
    app.add_handler(CommandHandler("odds_debug", odds_debug))
    app.run_polling()

if __name__ == "__main__":
    main()
