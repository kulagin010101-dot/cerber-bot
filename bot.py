import os
import json
import time
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# =====================================================
# ENV
# =====================================================
BOT_TOKEN = os.getenv("BOT_TOKEN")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
ODDSPAPI_KEY = os.getenv("ODDSPAPI_KEY")
ODDSPAPI_BOOKMAKER = os.getenv("ODDSPAPI_BOOKMAKER", "1xbet")  # твой тариф: 1xBet
WEATHER_KEY = os.getenv("WEATHER_API_KEY")  # optional

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не задан")
if not FOOTBALL_API_KEY:
    raise RuntimeError("FOOTBALL_API_KEY не задан")
if not ODDSPAPI_KEY:
    raise RuntimeError("ODDSPAPI_KEY не задан")

HEADERS_FOOTBALL = {"x-apisports-key": FOOTBALL_API_KEY}

SEASON = int(os.getenv("SEASON", "2025"))
MIN_PROB = float(os.getenv("MIN_PROB", "0.75"))   # 75%
MIN_VALUE = float(os.getenv("MIN_VALUE", "0.00")) # value > 0

OU_MIN, OU_MAX = float(os.getenv("OU_ODDS_MIN", "1.10")), float(os.getenv("OU_ODDS_MAX", "6.00"))
BTTS_MIN, BTTS_MAX = float(os.getenv("BTTS_ODDS_MIN", "1.10")), float(os.getenv("BTTS_ODDS_MAX", "6.00"))

STATE_FILE = "state.json"

# =====================================================
# TARGET TOURNAMENTS (Oddspapi)
# categoryName + tournamentName
# =====================================================
TARGET_TOURNAMENTS_ODDSPAPI = [
    ("England", "Premier League"),
    ("England", "FA Cup"),
    ("England", "EFL Cup"),
    ("England", "League Cup"),
    ("England", "Community Shield"),

    ("Germany", "Bundesliga"),
    ("Germany", "DFB Pokal"),
    ("Germany", "DFB-Pokal"),
    ("Germany", "Super Cup"),

    ("Spain", "LaLiga"),
    ("Spain", "La Liga"),
    ("Spain", "Copa del Rey"),
    ("Spain", "Super Cup"),

    ("Italy", "Serie A"),
    ("Italy", "Coppa Italia"),
    ("Italy", "Super Cup"),

    ("France", "Ligue 1"),
    ("France", "Coupe de France"),
    ("France", "Super Cup"),

    ("Russia", "Premier League"),

    ("International", "UEFA Champions League"),
    ("International", "UEFA Europa League"),
]

# =====================================================
# TARGET LEAGUES (API-Football)
# =====================================================
TARGET_COMPETITIONS_API_FOOTBALL: List[Dict[str, Any]] = [
    {"country": "England", "name": "Premier League", "aliases": ["Premier League"]},
    {"country": "England", "name": "FA Cup", "aliases": ["FA Cup"]},
    {"country": "England", "name": "League Cup", "aliases": ["League Cup", "EFL Cup", "Carabao Cup"]},
    {"country": "England", "name": "Community Shield", "aliases": ["Community Shield", "FA Community Shield"]},

    {"country": "Germany", "name": "Bundesliga", "aliases": ["Bundesliga", "1. Bundesliga", "Bundesliga - 1"]},
    {"country": "Germany", "name": "DFB Pokal", "aliases": ["DFB Pokal", "DFB-Pokal", "German Cup"]},
    {"country": "Germany", "name": "Super Cup", "aliases": ["Super Cup", "DFL Supercup", "German Super Cup"]},

    {"country": "Spain", "name": "La Liga", "aliases": ["La Liga", "LaLiga", "Primera Division"]},
    {"country": "Spain", "name": "Copa del Rey", "aliases": ["Copa del Rey", "King's Cup"]},
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

# =====================================================
# STATE
# =====================================================
def load_state() -> Dict[str, Any]:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_state(state: Dict[str, Any]) -> None:
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[WARN] cannot save state: {e}")

STATE = load_state()

# =====================================================
# Helpers
# =====================================================
def norm(s: Any) -> str:
    return str(s or "").strip().lower()

def safe_team_key(name: str) -> str:
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

def clamp(x: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return min(max(x, lo), hi)

def fair_odds(p: float) -> float:
    return round(1.0 / max(p, 1e-9), 2)

def value_ev(p: float, book_odds: float) -> float:
    return (p * book_odds) - 1.0

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

# =====================================================
# Weather (optional)
# =====================================================
def weather_factor(city: Optional[str]) -> float:
    if not WEATHER_KEY or not city:
        return 1.0
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": WEATHER_KEY, "units": "metric"},
            timeout=12
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

# =====================================================
# API-Football: last matches -> goal model
# =====================================================
def get_last_matches(team_id: int, last: int = 5) -> list:
    try:
        r = requests.get(
            "https://v3.football.api-sports.io/fixtures",
            headers=HEADERS_FOOTBALL,
            params={"team": team_id, "last": last, "season": SEASON},
            timeout=25
        )
        r.raise_for_status()
        return r.json().get("response", [])
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
        h = float(h); a = float(a)
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

# =====================================================
# API-Football: resolve league ids + today fixtures filtered
# =====================================================
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
    return r.json().get("response", [])

def score_candidate(target_country: Optional[str], aliases: List[str], cand_name: str, cand_country: str) -> int:
    tn = cand_name.strip().lower()
    tc = cand_country.strip().lower()
    score = 0
    if target_country:
        score += 50 if tc == target_country.strip().lower() else -10
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

    for item in TARGET_COMPETITIONS_API_FOOTBALL:
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
    return r.json().get("response", [])

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

# =====================================================
# OddsPapi: GET with retry on 429
# =====================================================
def oddspapi_get(path: str, params: Dict[str, Any]) -> Any:
    """
    OddsPapi GET с обработкой rate-limit (429).
    При 429 ждём retryMs и повторяем.
    """
    params = dict(params)
    params["apiKey"] = ODDSPAPI_KEY
    url = f"https://api.oddspapi.io{path}"

    for _attempt in range(6):
        r = requests.get(url, params=params, timeout=30)

        if r.status_code == 429:
            try:
                j = r.json()
                retry_ms = int(j.get("error", {}).get("retryMs", 250))
            except:
                retry_ms = 250
            time.sleep(max(0.05, retry_ms / 1000.0))
            continue

        if r.status_code >= 400:
            raise RuntimeError(f"OddsPapi HTTP {r.status_code}: {r.text[:300]}")

        return r.json()

    raise RuntimeError("OddsPapi: слишком много 429 подряд, попробуй позже.")

# ---- participants cache ----
PART_CACHE_KEY = "oddspapi_participants_10"
PART_CACHE_DATE = "oddspapi_participants_10_date"

def get_participants_map(force: bool = False) -> Dict[str, str]:
    today = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d")
    if not force and STATE.get(PART_CACHE_DATE) == today and isinstance(STATE.get(PART_CACHE_KEY), dict):
        return {str(k): str(v) for k, v in STATE[PART_CACHE_KEY].items()}

    data = oddspapi_get("/v4/participants", {"sportId": 10, "language": "en"})
    pm = {str(k): str(v) for k, v in (data or {}).items()}
    STATE[PART_CACHE_KEY] = pm
    STATE[PART_CACHE_DATE] = today
    save_state(STATE)
    return pm

# ---- tournaments cache ----
TOURN_CACHE_KEY = "oddspapi_tournaments_10"
TOURN_CACHE_DATE = "oddspapi_tournaments_10_date"

def resolve_target_tournament_ids(force: bool = False) -> Dict[str, int]:
    today = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d")
    if not force and STATE.get(TOURN_CACHE_DATE) == today and isinstance(STATE.get(TOURN_CACHE_KEY), dict):
        return {k: int(v) for k, v in STATE[TOURN_CACHE_KEY].items()}

    data = oddspapi_get("/v4/tournaments", {"sportId": 10})
    wanted = set((c, n) for (c, n) in TARGET_TOURNAMENTS_ODDSPAPI)

    out: Dict[str, int] = {}
    for t in data or []:
        cat = str(t.get("categoryName") or "").strip()
        name = str(t.get("tournamentName") or "").strip()
        tid = t.get("tournamentId")
        if not tid or not cat or not name:
            continue
        if (cat, name) in wanted:
            out[f"{cat}|{name}"] = int(tid)

    STATE[TOURN_CACHE_KEY] = {k: int(v) for k, v in out.items()}
    STATE[TOURN_CACHE_DATE] = today
    save_state(STATE)
    return out

def fetch_oddspapi_odds_by_tournaments(tournament_ids: List[int]) -> List[Dict[str, Any]]:
    """
    IMPORTANT: Oddspapi ограничение: максимум 5 tournamentIds за запрос
    """
    all_items: List[Dict[str, Any]] = []
    MAX_IDS = 5

    for i in range(0, len(tournament_ids), MAX_IDS):
        chunk = tournament_ids[i:i + MAX_IDS]
        ids = ",".join(str(x) for x in chunk)

        part = oddspapi_get(
            "/v4/odds-by-tournaments",
            {
                "tournamentIds": ids,
                "bookmaker": ODDSPAPI_BOOKMAKER,
                "oddsFormat": "decimal",
                "verbosity": 2,
            },
        )
        if isinstance(part, list):
            all_items.extend(part)

        # маленькая пауза, чтобы реже ловить 429
        time.sleep(0.06)

    return all_items

def parse_book_odds_from_oddspapi_item(item: Dict[str, Any]) -> Dict[str, Optional[float]]:
    out = {"O25": None, "U25": None, "BTTS_Y": None, "BTTS_N": None}

    bm = ((item.get("bookmakerOdds") or {}).get(ODDSPAPI_BOOKMAKER) or {})
    markets = bm.get("markets") or {}

    for _mid, m in markets.items():
        outcomes = m.get("outcomes") or {}
        for _oid, o in outcomes.items():
            players = o.get("players") or {}
            p0 = players.get("0") or players.get(0) or {}
            price = p0.get("price")
            outcome_id = norm(p0.get("bookmakerOutcomeId"))

            try:
                price = float(price)
            except:
                continue

            # OU 2.5
            if ("2.5/over" in outcome_id) or ("over/2.5" in outcome_id):
                if OU_MIN <= price <= OU_MAX:
                    out["O25"] = price
            elif ("2.5/under" in outcome_id) or ("under/2.5" in outcome_id):
                if OU_MIN <= price <= OU_MAX:
                    out["U25"] = price

            # BTTS
            if outcome_id in ("btts/yes", "btts_yes", "both_teams_to_score/yes", "yes"):
                if BTTS_MIN <= price <= BTTS_MAX:
                    out["BTTS_Y"] = price
            elif outcome_id in ("btts/no", "btts_no", "both_teams_to_score/no", "no"):
                if BTTS_MIN <= price <= BTTS_MAX:
                    out["BTTS_N"] = price

    return out

# ---- Odds cache (60 seconds) ----
ODDS_CACHE_KEY = "odds_cache"
ODDS_CACHE_TS = "odds_cache_ts"
ODDS_CACHE_DATE = "odds_cache_date"

def build_oddspapi_fixture_map_for_day(date_msk: str) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    now_ts = time.time()
    cached_date = STATE.get(ODDS_CACHE_DATE)
    cached_ts = float(STATE.get(ODDS_CACHE_TS, 0) or 0)
    cached = STATE.get(ODDS_CACHE_KEY)

    if cached_date == date_msk and isinstance(cached, dict) and (now_ts - cached_ts) < 60:
        out = {}
        for k, v in cached.items():
            d, h, a = k.split("|", 2)
            out[(d, h, a)] = v
        return out

    participants = get_participants_map(force=False)
    tourn_map = resolve_target_tournament_ids(force=False)
    tourn_ids = list(set(tourn_map.values()))

    items = fetch_oddspapi_odds_by_tournaments(tourn_ids)

    m: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for it in items:
        start = it.get("startTime")
        p1 = it.get("participant1Id")
        p2 = it.get("participant2Id")
        if not start or not p1 or not p2:
            continue

        try:
            dt_utc = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        except:
            continue

        dt_msk = dt_utc + timedelta(hours=3)
        d_msk = dt_msk.strftime("%Y-%m-%d")
        if d_msk != date_msk:
            continue

        t1 = participants.get(str(p1))
        t2 = participants.get(str(p2))
        if not t1 or not t2:
            continue

        k1 = safe_team_key(t1)
        k2 = safe_team_key(t2)
        m[(d_msk, k1, k2)] = it
        m[(d_msk, k2, k1)] = it

    compact = {}
    for (d, h, a), v in m.items():
        compact[f"{d}|{h}|{a}"] = v

    STATE[ODDS_CACHE_KEY] = compact
    STATE[ODDS_CACHE_DATE] = date_msk
    STATE[ODDS_CACHE_TS] = time.time()
    save_state(STATE)

    return m

# =====================================================
# Telegram handlers
# =====================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован.\n\n"
        "Команды:\n"
        "• /signals — value-сигналы (P≥75% и Value>0)\n"
        "• /all — все матчи (с P и odds)\n"
        "• /check — диагностика сматчинга odds\n"
        "• /odds_debug — сырой вывод outcomeId/price\n"
        "• /oddspapi_account — проверка ключа\n"
        "• /reload_leagues — пересканировать лиги API-Football\n\n"
        f"Bookmaker: {ODDSPAPI_BOOKMAKER}\n"
    )

async def oddspapi_account(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        oddspapi_get("/v4/account", {})
        await update.message.reply_text("✅ OddsPapi account OK (ключ работает).")
    except Exception as e:
        await update.message.reply_text(f"❌ OddsPapi account FAIL: {e}")

async def reload_leagues(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔄 Пересканирую лиги API-Football…")
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
        await update.message.reply_text(part)

async def check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await update.message.reply_text(f"⚠️ На дату {used_date} матчей нет (API-Football).")
        return

    try:
        odds_map = build_oddspapi_fixture_map_for_day(used_date)
    except Exception as e:
        await update.message.reply_text(f"❌ OddsPapi ошибка: {e}")
        return

    sample = fixtures[:15]
    matched, with_markets = 0, 0

    for f in sample:
        home = ((f.get("teams") or {}).get("home") or {}).get("name", "")
        away = ((f.get("teams") or {}).get("away") or {}).get("name", "")
        it = odds_map.get((used_date, safe_team_key(home), safe_team_key(away)))
        if it:
            matched += 1
            odds = parse_book_odds_from_oddspapi_item(it)
            if any(odds.values()):
                with_markets += 1

    msg = (
        f"🔎 CHECK ({used_date})\n"
        f"Матчей (после фильтра лиг): {len(fixtures)}\n"
        f"Проверено: {len(sample)}\n"
        f"Сматчено (OddsPapi): {matched}/{len(sample)}\n"
        f"Сматчено и есть OU/BTTS: {with_markets}/{len(sample)}\n"
        f"Bookmaker: {ODDSPAPI_BOOKMAKER}\n"
    )
    await update.message.reply_text(msg)

async def odds_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await update.message.reply_text(f"⚠️ На дату {used_date} матчей нет.")
        return

    try:
        odds_map = build_oddspapi_fixture_map_for_day(used_date)
    except Exception as e:
        await update.message.reply_text(f"❌ OddsPapi ошибка: {e}")
        return

    f = fixtures[0]
    home = ((f.get("teams") or {}).get("home") or {}).get("name", "")
    away = ((f.get("teams") or {}).get("away") or {}).get("name", "")

    it = odds_map.get((used_date, safe_team_key(home), safe_team_key(away)))
    if not it:
        await update.message.reply_text("❌ Не сматчился первый матч дня (по названиям команд).")
        return

    bm = ((it.get("bookmakerOdds") or {}).get(ODDSPAPI_BOOKMAKER) or {})
    markets = bm.get("markets") or {}

    lines = [
        f"🧪 ODDS DEBUG ({used_date})\n{home} — {away}\nBookmaker: {ODDSPAPI_BOOKMAKER}\n",
        f"fixtureId: {it.get('fixtureId')}\nstartTime: {it.get('startTime')}\nmarkets: {len(markets)}\n\n",
        "RAW sample (до 50 строк):\n",
    ]

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
            if shown >= 50:
                break
        if shown >= 50:
            break

    for part in chunked("".join(lines)):
        await update.message.reply_text(part)

async def all_matches(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await update.message.reply_text(f"⚠️ На дату {used_date} матчей нет.")
        return

    try:
        odds_map = build_oddspapi_fixture_map_for_day(used_date)
    except Exception as e:
        await update.message.reply_text(f"❌ OddsPapi ошибка: {e}")
        return

    out = [f"📋 ALL ({used_date}) — Book: {ODDSPAPI_BOOKMAKER}\n\n"]
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

        it = odds_map.get((used_date, safe_team_key(home.get("name", "")), safe_team_key(away.get("name", ""))))
        odds = {"O25": None, "U25": None, "BTTS_Y": None, "BTTS_N": None}
        if it:
            odds = parse_book_odds_from_oddspapi_item(it)

        def line(title: str, p: float, book: Optional[float]) -> str:
            if book is None:
                return f"{title}: P={int(p*100)}% | Book=—"
            v = value_ev(p, book)
            return f"{title}: P={int(p*100)}% | Book={book:.2f} | Fair={fair_odds(p):.2f} | Value={v:+.2f}"

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
            await update.message.reply_text("".join(out))
            out = ["(продолжение)\n\n"]

    if out:
        await update.message.reply_text("".join(out))

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await update.message.reply_text(f"⚠️ На дату {used_date} матчей нет.")
        return

    try:
        odds_map = build_oddspapi_fixture_map_for_day(used_date)
    except Exception as e:
        await update.message.reply_text(f"❌ OddsPapi ошибка: {e}")
        return

    out = [f"⚽ ЦЕРБЕР — value-сигналы ({used_date})\nOdds: {ODDSPAPI_BOOKMAKER}\n\n"]
    found_any = 0
    batch = 0

    for f in fixtures:
        fixture = f.get("fixture") or {}
        home = (f.get("teams") or {}).get("home", {}) or {}
        away = (f.get("teams") or {}).get("away", {}) or {}
        league = f.get("league", {}) or {}
        venue = fixture.get("venue") or {}
        city = venue.get("city")

        home_id = home.get("id")
        away_id = away.get("id")
        if not home_id or not away_id:
            continue

        it = odds_map.get((used_date, safe_team_key(home.get("name", "")), safe_team_key(away.get("name", ""))))
        if not it:
            continue

        odds = parse_book_odds_from_oddspapi_item(it)
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

        batch += 1
        if batch % 6 == 0:
            await update.message.reply_text("".join(out))
            out = ["(продолжение)\n\n"]

    if found_any == 0:
        await update.message.reply_text(
            f"📭 На дату {used_date} нет value-сигналов при порогах:\n"
            f"P ≥ {int(MIN_PROB*100)}% и Value > {MIN_VALUE:+.2f}\n\n"
            "Проверка:\n"
            "• /check\n"
            "• /odds_debug (посмотреть реальные outcomeId)\n"
        )
        return

    if out:
        await update.message.reply_text("".join(out))

# =====================================================
# RUN
# =====================================================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))
    app.add_handler(CommandHandler("all", all_matches))
    app.add_handler(CommandHandler("check", check))
    app.add_handler(CommandHandler("odds_debug", odds_debug))
    app.add_handler(CommandHandler("oddspapi_account", oddspapi_account))
    app.add_handler(CommandHandler("reload_leagues", reload_leagues))
    app.run_polling()

if __name__ == "__main__":
    main()
