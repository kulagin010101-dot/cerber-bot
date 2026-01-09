import os
import json
import time
import unicodedata
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

# sanity filters for odds
OU_MIN, OU_MAX = float(os.getenv("OU_ODDS_MIN", "1.05")), float(os.getenv("OU_ODDS_MAX", "20.0"))
BTTS_MIN, BTTS_MAX = float(os.getenv("BTTS_ODDS_MIN", "1.05")), float(os.getenv("BTTS_ODDS_MAX", "20.0"))

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
def safe_team_key(name: str) -> str:
    s = str(name or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    repl = {
        "&": "and",
        "’": "'",
        ".": "",
        ",": "",
        "-": " ",
        " fc": "",
        "fc ": "",
        " cf": "",
        "cf ": "",
        " sc": "",
        "sc ": "",
        " ac": "",
        "ac ": "",
    }
    for a, b in repl.items():
        s = s.replace(a, b)
    return " ".join(s.split())

def tokens(s: str) -> set:
    return set([t for t in safe_team_key(s).split() if t and t not in {"the", "club"}])

def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))

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

def fmt_odds(x: Optional[float]) -> str:
    return "—" if x is None else f"{x:.2f}"

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
# OddsPapi helpers
# =====================================================
def oddspapi_get(path: str, params: Dict[str, Any]) -> Any:
    params = dict(params)
    params["apiKey"] = ODDSPAPI_KEY
    url = f"https://api.oddspapi.io{path}"

    for _ in range(6):
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
            raise RuntimeError(f"OddsPapi HTTP {r.status_code}: {r.text[:500]}")
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
        time.sleep(0.08)
    return all_items

# ---- markets meta cache ----
MARKETS_CACHE_KEY = "oddspapi_markets"
MARKETS_CACHE_DATE = "oddspapi_markets_date"

def get_markets_meta(force: bool = False) -> Dict[int, Dict[str, Any]]:
    today = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d")
    if (not force) and STATE.get(MARKETS_CACHE_DATE) == today and isinstance(STATE.get(MARKETS_CACHE_KEY), dict):
        return {int(k): v for k, v in STATE[MARKETS_CACHE_KEY].items()}

    data = oddspapi_get("/v4/markets", {"language": "en"})
    meta: Dict[int, Dict[str, Any]] = {}
    for m in data or []:
        mid = m.get("marketId")
        if mid is None:
            continue
        mid = int(mid)

        outs: Dict[int, str] = {}
        for o in (m.get("outcomes") or []):
            oid = o.get("outcomeId")
            oname = o.get("outcomeName")
            if oid is None or oname is None:
                continue
            try:
                outs[int(oid)] = str(oname)
            except:
                pass

        meta[mid] = {
            "name": str(m.get("marketName") or ""),
            "handicap": m.get("handicap"),
            "outcomes": outs,
        }

    STATE[MARKETS_CACHE_KEY] = {str(k): v for k, v in meta.items()}
    STATE[MARKETS_CACHE_DATE] = today
    save_state(STATE)
    return meta

# =====================================================
# Odds parsing: OU2.5 + BTTS + Team Totals
# =====================================================
BTTS_MARKET_ID = 104
OU_FT_NAME = "over under full time"
TEAM1_OU_NAME = "over under team 1"
TEAM2_OU_NAME = "over under team 2"

def parse_odds(item: Dict[str, Any], markets_meta: Dict[int, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    """
    Возвращает odds для:
    - O/U 2.5 (Full Time)
    - BTTS Yes/No
    - Team Totals (Team 1 / Team 2) для 0.5 / 1.5 / 2.5
    """
    out: Dict[str, Optional[float]] = {
        "O25": None, "U25": None,
        "BTTS_Y": None, "BTTS_N": None,
        "IT1_O05": None, "IT1_U05": None,
        "IT1_O15": None, "IT1_U15": None,
        "IT1_O25": None, "IT1_U25": None,
        "IT2_O05": None, "IT2_U05": None,
        "IT2_O15": None, "IT2_U15": None,
        "IT2_O25": None, "IT2_U25": None,
    }

    bm = ((item.get("bookmakerOdds") or {}).get(ODDSPAPI_BOOKMAKER) or {})
    markets = bm.get("markets") or {}

    def price_from_players(players_obj: Dict[str, Any]) -> Optional[float]:
        p0 = players_obj.get("0") or players_obj.get(0) or {}
        try:
            return float(p0.get("price"))
        except:
            return None

    def set_ou(out_over_key: str, out_under_key: str, oname: str, price: float, lo: float, hi: float):
        if not (lo <= price <= hi):
            return
        n = oname.lower()
        if "over" in n:
            out[out_over_key] = price
        elif "under" in n:
            out[out_under_key] = price

    for mid_str, m in markets.items():
        try:
            mid = int(mid_str)
        except:
            continue

        meta = markets_meta.get(mid, {})
        mname = str(meta.get("name") or "").lower()
        hcap = meta.get("handicap", None)
        outcomes_meta: Dict[int, str] = meta.get("outcomes") or {}
        outcomes = m.get("outcomes") or {}

        # --- BTTS ---
        if mid == BTTS_MARKET_ID:
            for oid_str, o in outcomes.items():
                try:
                    oid = int(oid_str)
                except:
                    continue
                price = price_from_players(o.get("players") or {})
                if price is None:
                    continue
                oname = (outcomes_meta.get(oid, "") or "").lower()
                if "yes" in oname and BTTS_MIN <= price <= BTTS_MAX:
                    out["BTTS_Y"] = price
                if "no" in oname and BTTS_MIN <= price <= BTTS_MAX:
                    out["BTTS_N"] = price

        # --- helper: handicap float ---
        h = None
        try:
            if hcap is not None:
                h = float(hcap)
        except:
            h = None

        # --- Full Time O/U 2.5 ---
        if (OU_FT_NAME in mname) and (h == 2.5):
            for oid_str, o in outcomes.items():
                try:
                    oid = int(oid_str)
                except:
                    continue
                price = price_from_players(o.get("players") or {})
                if price is None:
                    continue
                oname = outcomes_meta.get(oid, "") or ""
                set_ou("O25", "U25", oname, price, OU_MIN, OU_MAX)

        # --- Team totals: Team 1 ---
        if (TEAM1_OU_NAME in mname) and (h in (0.5, 1.5, 2.5)):
            for oid_str, o in outcomes.items():
                try:
                    oid = int(oid_str)
                except:
                    continue
                price = price_from_players(o.get("players") or {})
                if price is None:
                    continue
                oname = outcomes_meta.get(oid, "") or ""
                if h == 0.5:
                    set_ou("IT1_O05", "IT1_U05", oname, price, OU_MIN, OU_MAX)
                elif h == 1.5:
                    set_ou("IT1_O15", "IT1_U15", oname, price, OU_MIN, OU_MAX)
                elif h == 2.5:
                    set_ou("IT1_O25", "IT1_U25", oname, price, OU_MIN, OU_MAX)

        # --- Team totals: Team 2 ---
        if (TEAM2_OU_NAME in mname) and (h in (0.5, 1.5, 2.5)):
            for oid_str, o in outcomes.items():
                try:
                    oid = int(oid_str)
                except:
                    continue
                price = price_from_players(o.get("players") or {})
                if price is None:
                    continue
                oname = outcomes_meta.get(oid, "") or ""
                if h == 0.5:
                    set_ou("IT2_O05", "IT2_U05", oname, price, OU_MIN, OU_MAX)
                elif h == 1.5:
                    set_ou("IT2_O15", "IT2_U15", oname, price, OU_MIN, OU_MAX)
                elif h == 2.5:
                    set_ou("IT2_O25", "IT2_U25", oname, price, OU_MIN, OU_MAX)

    return out

# =====================================================
# Odds cache (60 sec) + index for fuzzy matching
# =====================================================
ODDS_CACHE_KEY = "odds_cache"
ODDS_CACHE_TS = "odds_cache_ts"
ODDS_CACHE_DATE = "odds_cache_date"
ODDS_INDEX_KEY = "odds_index"
ODDS_INDEX_DATE = "odds_index_date"

def build_oddspapi_for_day(date_msk: str) -> Tuple[Dict[Tuple[str, str, str], Dict[str, Any]], List[Dict[str, Any]]]:
    now_ts = time.time()
    cached_date = STATE.get(ODDS_CACHE_DATE)
    cached_ts = float(STATE.get(ODDS_CACHE_TS, 0) or 0)
    cached = STATE.get(ODDS_CACHE_KEY)
    cached_index = STATE.get(ODDS_INDEX_KEY)
    cached_index_date = STATE.get(ODDS_INDEX_DATE)

    if cached_date == date_msk and isinstance(cached, dict) and (now_ts - cached_ts) < 60 and cached_index_date == date_msk and isinstance(cached_index, list):
        out_map = {}
        for k, v in cached.items():
            d, h, a = k.split("|", 2)
            out_map[(d, h, a)] = v
        return out_map, cached_index

    participants = get_participants_map(force=False)
    tourn_map = resolve_target_tournament_ids(force=False)
    tourn_ids = list(set(tourn_map.values()))
    items = fetch_oddspapi_odds_by_tournaments(tourn_ids)

    out_map: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    out_index: List[Dict[str, Any]] = []

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

        hk = safe_team_key(t1)
        ak = safe_team_key(t2)
        ts_msk = int(dt_msk.timestamp())

        out_map[(d_msk, hk, ak)] = it
        out_map[(d_msk, ak, hk)] = it
        out_index.append({"d": d_msk, "h": hk, "a": ak, "ts": ts_msk, "item": it})

    compact = {}
    for (d, h, a), v in out_map.items():
        compact[f"{d}|{h}|{a}"] = v

    STATE[ODDS_CACHE_KEY] = compact
    STATE[ODDS_CACHE_DATE] = date_msk
    STATE[ODDS_CACHE_TS] = time.time()
    STATE[ODDS_INDEX_KEY] = out_index
    STATE[ODDS_INDEX_DATE] = date_msk
    save_state(STATE)
    return out_map, out_index

def match_odds_for_fixture(date_msk: str, fixture_ts_msk: int, home_name: str, away_name: str,
                           odds_map: Dict[Tuple[str, str, str], Dict[str, Any]],
                           odds_index: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    hk = safe_team_key(home_name)
    ak = safe_team_key(away_name)
    direct = odds_map.get((date_msk, hk, ak))
    if direct:
        return direct

    ht = tokens(home_name)
    at = tokens(away_name)
    best = None
    best_score = 0.0

    for row in odds_index:
        if row.get("d") != date_msk:
            continue
        ts = int(row.get("ts") or 0)
        if abs(ts - fixture_ts_msk) > 2 * 3600:
            continue
        s1 = jaccard(ht, set((row["h"] or "").split()))
        s2 = jaccard(at, set((row["a"] or "").split()))
        score = (s1 + s2) / 2.0
        score += max(0.0, 0.15 - abs(ts - fixture_ts_msk) / (2 * 3600) * 0.15)
        if score > best_score:
            best_score = score
            best = row["item"]

    if best and best_score >= 0.45:
        return best
    return None

# =====================================================
# Telegram commands
# =====================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован.\n\n"
        "Команды:\n"
        "• /signals — value-сигналы (P≥75% и Value>0)\n"
        "• /lines — линии/коэффы на все матчи (без фильтра value)\n"
        "• /check — диагностика odds\n"
        "• /odds_debug — debug рынка\n"
        "• /oddspapi_account — проверка ключа Oddspapi\n"
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
        odds_map, odds_index = build_oddspapi_for_day(used_date)
        markets_meta = get_markets_meta(force=False)
    except Exception as e:
        await update.message.reply_text(f"❌ OddsPapi ошибка: {e}")
        return

    sample = fixtures[:15]
    matched, with_markets = 0, 0

    for f in sample:
        fixture = f.get("fixture") or {}
        ts = int(fixture.get("timestamp") or 0)
        ts_msk = int((datetime.utcfromtimestamp(ts) + timedelta(hours=3)).timestamp()) if ts else 0
        home = ((f.get("teams") or {}).get("home") or {}).get("name", "")
        away = ((f.get("teams") or {}).get("away") or {}).get("name", "")

        it = match_odds_for_fixture(used_date, ts_msk, home, away, odds_map, odds_index)
        if it:
            matched += 1
            odds = parse_odds(it, markets_meta)
            if any([
                odds.get("O25"), odds.get("U25"),
                odds.get("BTTS_Y"), odds.get("BTTS_N")
            ]):
                with_markets += 1

    await update.message.reply_text(
        f"🔎 CHECK ({used_date})\n"
        f"Матчей (после фильтра лиг): {len(fixtures)}\n"
        f"Проверено: {len(sample)}\n"
        f"Сматчено (OddsPapi): {matched}/{len(sample)}\n"
        f"Сматчено и есть линии (OU/BTTS): {with_markets}/{len(sample)}\n"
        f"Bookmaker: {ODDSPAPI_BOOKMAKER}\n"
    )

async def odds_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await update.message.reply_text(f"⚠️ На дату {used_date} матчей нет.")
        return

    try:
        odds_map, odds_index = build_oddspapi_for_day(used_date)
        markets_meta = get_markets_meta(force=False)
    except Exception as e:
        await update.message.reply_text(f"❌ OddsPapi ошибка: {e}")
        return

    matched_fixture = None
    matched_item = None

    for f in fixtures:
        fixture = f.get("fixture") or {}
        ts = int(fixture.get("timestamp") or 0)
        ts_msk = int((datetime.utcfromtimestamp(ts) + timedelta(hours=3)).timestamp()) if ts else 0
        home = ((f.get("teams") or {}).get("home") or {}).get("name", "")
        away = ((f.get("teams") or {}).get("away") or {}).get("name", "")
        it = match_odds_for_fixture(used_date, ts_msk, home, away, odds_map, odds_index)
        if it:
            matched_fixture = f
            matched_item = it
            break

    if not matched_item:
        await update.message.reply_text("❌ Не нашёл ни одного сматченного матча для odds_debug.")
        return

    home = ((matched_fixture.get("teams") or {}).get("home") or {}).get("name", "")
    away = ((matched_fixture.get("teams") or {}).get("away") or {}).get("name", "")

    bm = ((matched_item.get("bookmakerOdds") or {}).get(ODDSPAPI_BOOKMAKER) or {})
    markets = bm.get("markets") or {}

    lines = [
        f"🧪 ODDS DEBUG ({used_date})\n{home} — {away}\nBookmaker: {ODDSPAPI_BOOKMAKER}\n",
        f"fixtureId: {matched_item.get('fixtureId')}\nstartTime: {matched_item.get('startTime')}\nmarkets: {len(markets)}\n\n",
        "format: marketId (marketName h=handicap) | outcomeId (outcomeName) | price\n\n",
    ]

    def p0_price(players_obj: Dict[str, Any]) -> Optional[float]:
        p0 = players_obj.get("0") or players_obj.get(0) or {}
        try:
            return float(p0.get("price"))
        except:
            return None

    shown = 0
    for mid_str, m in markets.items():
        try:
            mid = int(mid_str)
        except:
            continue
        meta = markets_meta.get(mid, {})
        mname = meta.get("name", "?")
        hcap = meta.get("handicap", "")
        outcomes = m.get("outcomes") or {}
        for oid_str, o in outcomes.items():
            try:
                oid = int(oid_str)
            except:
                continue
            oname = (meta.get("outcomes") or {}).get(oid, "?")
            price = p0_price(o.get("players") or {})
            if price is None:
                continue
            lines.append(f"m={mid} ({mname} h={hcap}) | o={oid} ({oname}) | price={price}\n")
            shown += 1
            if shown >= 140:
                break
        if shown >= 140:
            break

    for part in chunked("".join(lines)):
        await update.message.reply_text(part)

async def lines(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await update.message.reply_text(f"⚠️ На дату {used_date} матчей нет.")
        return

    try:
        odds_map, odds_index = build_oddspapi_for_day(used_date)
        markets_meta = get_markets_meta(force=False)
    except Exception as e:
        await update.message.reply_text(f"❌ OddsPapi ошибка: {e}")
        return

    msg = [f"📌 ЛИНИИ ({used_date})\nBookmaker: {ODDSPAPI_BOOKMAKER}\n\n"]
    sent = 0
    batch = 0

    for f in fixtures:
        fixture = f.get("fixture") or {}
        home = (f.get("teams") or {}).get("home", {}) or {}
        away = (f.get("teams") or {}).get("away", {}) or {}
        league = f.get("league", {}) or {}

        ts = int(fixture.get("timestamp") or 0)
        if not ts:
            continue
        ts_msk = int((datetime.utcfromtimestamp(ts) + timedelta(hours=3)).timestamp())
        t_str = (datetime.utcfromtimestamp(ts) + timedelta(hours=3)).strftime("%H:%M МСК")

        it = match_odds_for_fixture(used_date, ts_msk, home.get("name", ""), away.get("name", ""), odds_map, odds_index)
        if not it:
            continue

        o = parse_odds(it, markets_meta)

        # если вообще нет рынков OU/BTTS/IT — пропускаем
        if not any(v is not None for v in o.values()):
            continue

        sent += 1
        msg.append(
            f"🏆 {league.get('name','?')}\n"
            f"{home.get('name','?')} — {away.get('name','?')}\n"
            f"🕒 {t_str}\n"
            f"ТБ 2.5: {fmt_odds(o['O25'])} | ТМ 2.5: {fmt_odds(o['U25'])}\n"
            f"ОЗ Да: {fmt_odds(o['BTTS_Y'])} | ОЗ Нет: {fmt_odds(o['BTTS_N'])}\n"
            f"ИТ1 >0.5: {fmt_odds(o['IT1_O05'])} | ИТ1 >1.5: {fmt_odds(o['IT1_O15'])} | ИТ1 >2.5: {fmt_odds(o['IT1_O25'])}\n"
            f"ИТ1 <0.5: {fmt_odds(o['IT1_U05'])} | ИТ1 <1.5: {fmt_odds(o['IT1_U15'])} | ИТ1 <2.5: {fmt_odds(o['IT1_U25'])}\n"
            f"ИТ2 >0.5: {fmt_odds(o['IT2_O05'])} | ИТ2 >1.5: {fmt_odds(o['IT2_O15'])} | ИТ2 >2.5: {fmt_odds(o['IT2_O25'])}\n"
            f"ИТ2 <0.5: {fmt_odds(o['IT2_U05'])} | ИТ2 <1.5: {fmt_odds(o['IT2_U15'])} | ИТ2 <2.5: {fmt_odds(o['IT2_U25'])}\n"
            "\n"
        )

        batch += 1
        if batch % 4 == 0:
            await update.message.reply_text("".join(msg))
            msg = ["(продолжение)\n\n"]

    if sent == 0:
        await update.message.reply_text(
            f"📭 На дату {used_date} не нашёл линий у {ODDSPAPI_BOOKMAKER} после сматчинга.\n"
            "Проверь /check или /odds_debug."
        )
        return

    if msg:
        await update.message.reply_text("".join(msg))

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await update.message.reply_text(f"⚠️ На дату {used_date} матчей нет.")
        return

    try:
        odds_map, odds_index = build_oddspapi_for_day(used_date)
        markets_meta = get_markets_meta(force=False)
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

        home_id = home.get("id")
        away_id = away.get("id")
        if not home_id or not away_id:
            continue

        ts = int(fixture.get("timestamp") or 0)
        if not ts:
            continue
        ts_msk = int((datetime.utcfromtimestamp(ts) + timedelta(hours=3)).timestamp())
        t_str = (datetime.utcfromtimestamp(ts) + timedelta(hours=3)).strftime("%H:%M МСК")

        it = match_odds_for_fixture(used_date, ts_msk, home.get("name", ""), away.get("name", ""), odds_map, odds_index)
        if not it:
            continue

        odds = parse_odds(it, markets_meta)
        if not any([odds.get("O25"), odds.get("U25"), odds.get("BTTS_Y"), odds.get("BTTS_N")]):
            continue

        venue = fixture.get("venue") or {}
        city = venue.get("city")
        w_k = weather_factor(city)

        hs = analyze_goals(get_last_matches(int(home_id), last=5))
        as_ = analyze_goals(get_last_matches(int(away_id), last=5))
        if not hs or not as_:
            continue

        p_o25 = prob_over25(hs, as_, w_k)
        p_u25 = clamp(1.0 - p_o25, 0.05, 0.90)
        p_btts_y = prob_btts_yes(hs, as_, w_k)
        p_btts_n = clamp(1.0 - p_btts_y, 0.05, 0.90)

        def market_line(title: str, p: float, book: Optional[float]) -> Optional[str]:
            if book is None:
                return None
            if p < MIN_PROB:
                return None
            v = value_ev(p, book)
            if v <= MIN_VALUE:
                return None
            return f"{title}: P={int(p*100)}% | Book={book:.2f} | Fair={fair_odds(p):.2f} | Value={v:+.2f}"

        lines_out = []
        for title, p, book in [
            ("ТБ 2.5", p_o25, odds["O25"]),
            ("ТМ 2.5", p_u25, odds["U25"]),
            ("ОЗ Да", p_btts_y, odds["BTTS_Y"]),
            ("ОЗ Нет", p_btts_n, odds["BTTS_N"]),
        ]:
            ml = market_line(title, p, book)
            if ml:
                lines_out.append(ml)

        if not lines_out:
            continue

        found_any += 1
        out.append(
            f"🏆 {league.get('name','?')}\n"
            f"{home.get('name','?')} — {away.get('name','?')}\n"
            f"🕒 {t_str}\n" +
            "\n".join(lines_out) +
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
            "Если хочешь просто видеть коэффициенты — используй /lines."
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
    app.add_handler(CommandHandler("lines", lines))
    app.add_handler(CommandHandler("check", check))
    app.add_handler(CommandHandler("odds_debug", odds_debug))
    app.add_handler(CommandHandler("oddspapi_account", oddspapi_account))
    app.add_handler(CommandHandler("reload_leagues", reload_leagues))
    app.run_polling()

if __name__ == "__main__":
    main()
