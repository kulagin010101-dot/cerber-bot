import os
import json
import time
import math
import sqlite3
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
ODDSPAPI_BOOKMAKER = os.getenv("ODDSPAPI_BOOKMAKER", "1xbet")
WEATHER_KEY = os.getenv("WEATHER_API_KEY")  # optional (OpenWeather)

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не задан")
if not FOOTBALL_API_KEY:
    raise RuntimeError("FOOTBALL_API_KEY не задан")
if not ODDSPAPI_KEY:
    raise RuntimeError("ODDSPAPI_KEY не задан")

HEADERS_FOOTBALL = {"x-apisports-key": FOOTBALL_API_KEY}

SEASON = int(os.getenv("SEASON", "2025"))
MIN_PROB = float(os.getenv("MIN_PROB", "0.75"))     # 75%
MIN_VALUE = float(os.getenv("MIN_VALUE", "0.00"))   # value > 0
STAKE = float(os.getenv("STAKE", "1.0"))            # 1 unit fixed

LAST_MATCHES_TEAM = int(os.getenv("LAST_MATCHES_TEAM", "8"))

# Anti-anomalies
ANOMALY_VALUE_WARN = float(os.getenv("ANOMALY_VALUE_WARN", "0.50"))

SANITY_OU_MIN = float(os.getenv("SANITY_OU_MIN", "1.05"))
SANITY_OU_MAX = float(os.getenv("SANITY_OU_MAX", "6.00"))

STATE_FILE = "state.json"
DB_FILE = "cerber.db"

# =====================================================
# TARGET TOURNAMENTS (Oddspapi)
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
# DB (SQLite)
# =====================================================
def db_connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB_FILE)
    con.execute("PRAGMA journal_mode=WAL;")
    con.execute("PRAGMA synchronous=NORMAL;")
    return con

def db_init() -> None:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL,
      match_date TEXT NOT NULL,
      fixture_id INTEGER NOT NULL,
      league TEXT,
      home TEXT,
      away TEXT,
      market_code TEXT NOT NULL,
      market_label TEXT NOT NULL,
      line REAL,
      side TEXT,
      p_model REAL NOT NULL,
      book_odds REAL NOT NULL,
      fair_odds REAL NOT NULL,
      value_ev REAL NOT NULL,
      stake REAL NOT NULL,
      status TEXT NOT NULL DEFAULT 'PENDING',
      result TEXT,
      profit REAL,
      settled_at TEXT
    );
    """)
    cur.execute("""
    CREATE UNIQUE INDEX IF NOT EXISTS ux_signals_fixture_market
    ON signals(fixture_id, market_code);
    """)
    con.commit()
    con.close()

def db_upsert_signal(payload: Dict[str, Any]) -> None:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO signals (
      created_at, match_date, fixture_id, league, home, away,
      market_code, market_label, line, side,
      p_model, book_odds, fair_odds, value_ev, stake,
      status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
    """, (
        payload["created_at"], payload["match_date"], payload["fixture_id"],
        payload.get("league"), payload.get("home"), payload.get("away"),
        payload["market_code"], payload["market_label"], payload.get("line"),
        payload.get("side"),
        float(payload["p_model"]), float(payload["book_odds"]), float(payload["fair_odds"]),
        float(payload["value_ev"]), float(payload["stake"]),
    ))
    con.commit()
    con.close()

def db_fetch_pending(limit: int = 250) -> List[Dict[str, Any]]:
    con = db_connect()
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
      SELECT * FROM signals
      WHERE status='PENDING'
      ORDER BY match_date ASC
      LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def db_settle_signal(sig_id: int, result: str, profit: float) -> None:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
      UPDATE signals
      SET status='SETTLED', result=?, profit=?, settled_at=?
      WHERE id=?
    """, (result, float(profit), datetime.utcnow().isoformat(), sig_id))
    con.commit()
    con.close()

def db_stats(days: Optional[int]) -> Dict[str, Any]:
    con = db_connect()
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    if days is None:
        cur.execute("""SELECT result, profit, stake FROM signals WHERE status='SETTLED'""")
    else:
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        cur.execute("""SELECT result, profit, stake FROM signals WHERE status='SETTLED' AND settled_at >= ?""", (since,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()

    wins = sum(1 for r in rows if r.get("result") == "WIN")
    loses = sum(1 for r in rows if r.get("result") == "LOSE")
    pushes = sum(1 for r in rows if r.get("result") == "PUSH")
    stake_sum = sum(float(r.get("stake") or 0.0) for r in rows)
    profit_sum = sum(float(r.get("profit") or 0.0) for r in rows)
    hit = (wins / (wins + loses)) if (wins + loses) > 0 else 0.0
    roi = (profit_sum / stake_sum * 100.0) if stake_sum > 0 else 0.0
    return {
        "n": len(rows), "wins": wins, "loses": loses, "pushes": pushes,
        "stake_sum": stake_sum, "profit_sum": profit_sum, "hit": hit, "roi": roi
    }

def db_last_history(limit: int = 20) -> List[Dict[str, Any]]:
    con = db_connect()
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    cur.execute("""
      SELECT created_at, match_date, league, home, away, market_label, book_odds, p_model, value_ev, result, profit
      FROM signals
      ORDER BY id DESC
      LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

# =====================================================
# Helpers
# =====================================================
def safe_team_key(name: str) -> str:
    s = str(name or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    repl = {
        "&": "and", "’": "'",
        ".": "", ",": "", "-": " ",
        " fc": "", "fc ": "", " cf": "", "cf ": "", " sc": "", "sc ": "", " ac": "", "ac ": "",
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

def is_cup_like(league_name: str) -> bool:
    s = (league_name or "").lower()
    return any(k in s for k in ["cup", "copa", "coppa", "pokal", "shield", "super", "champions league", "europa league"])

# =====================================================
# Anti-anomaly #2: odds sanity check
# =====================================================
def odds_sanity(market_code: str, book: float) -> Tuple[bool, str]:
    if book is None or not isinstance(book, (int, float)):
        return False, "⚠️ подозрительная линия (odds)"
    book = float(book)

    if market_code in {"O15", "U15", "O25", "U25", "O35", "U35"}:
        ok = (SANITY_OU_MIN <= book <= SANITY_OU_MAX)
        return ok, ("⚠️ подозрительная линия (odds)" if not ok else "")

    return True, ""

# =====================================================
# Weather (optional) + debug
# =====================================================
def weather_factor(city: Optional[str]) -> Tuple[float, str]:
    if not WEATHER_KEY or not city:
        return 1.0, "weather=0%"
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
        notes = []
        if temp < 5 or temp > 28:
            factor *= 0.95
            notes.append("temp")
        if rain > 0:
            factor *= 0.95
            notes.append("rain")
        if wind > 8:
            factor *= 0.96
            notes.append("wind")
        pct = int(round((factor - 1.0) * 100))
        return factor, f"weather={pct}% ({'/'.join(notes) if notes else 'ok'})"
    except:
        return 1.0, "weather=0%"

def weather_factor_debug(city: Optional[str]) -> Dict[str, Any]:
    out = {"city": city or None, "enabled": bool(WEATHER_KEY), "ok": False}
    if not WEATHER_KEY or not city:
        out.update({"factor": 1.0, "reason": "weather=0%", "note": "no key or no city"})
        return out
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": WEATHER_KEY, "units": "metric"},
            timeout=12
        )
        out["http"] = r.status_code
        r.raise_for_status()
        data = r.json()
        temp = float(data["main"]["temp"])
        wind = float(data["wind"]["speed"])
        rain = 0.0
        if isinstance(data.get("rain"), dict):
            rain = float(data["rain"].get("1h", 0.0))

        factor = 1.0
        notes = []
        if temp < 5 or temp > 28:
            factor *= 0.95
            notes.append("temp")
        if rain > 0:
            factor *= 0.95
            notes.append("rain")
        if wind > 8:
            factor *= 0.96
            notes.append("wind")
        pct = int(round((factor - 1.0) * 100))
        out.update({
            "ok": True,
            "temp": temp,
            "wind": wind,
            "rain_1h": rain,
            "factor": factor,
            "reason": f"weather={pct}% ({'/'.join(notes) if notes else 'ok'})"
        })
        return out
    except Exception as e:
        out.update({"factor": 1.0, "reason": "weather=0%", "error": str(e)})
        return out

# =====================================================
# API-Football wrappers
# =====================================================
def football_get(path: str, params: Dict[str, Any], timeout: int = 25) -> Any:
    url = f"https://v3.football.api-sports.io{path}"
    r = requests.get(url, headers=HEADERS_FOOTBALL, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_last_matches(team_id: int, last: int = 5) -> list:
    try:
        j = football_get("/fixtures", {"team": team_id, "last": last, "season": SEASON})
        return j.get("response", [])
    except:
        return []

def team_goal_stats_venue(team_id: int, venue: str, last: int = 8) -> Optional[Dict[str, float]]:
    matches = get_last_matches(team_id, last=last + 8)
    scored = 0.0
    conceded = 0.0
    n = 0.0
    opponents: List[int] = []

    for m in matches:
        goals = m.get("goals", {}) or {}
        h, a = goals.get("home"), goals.get("away")
        if h is None or a is None:
            continue

        teams = m.get("teams", {}) or {}
        home = (teams.get("home") or {}).get("id")
        away = (teams.get("away") or {}).get("id")
        if not home or not away:
            continue

        if venue == "home" and int(home) != int(team_id):
            continue
        if venue == "away" and int(away) != int(team_id):
            continue

        h = float(h); a = float(a)

        if int(home) == int(team_id):
            scored += h
            conceded += a
            opponents.append(int(away))
        else:
            scored += a
            conceded += h
            opponents.append(int(home))

        n += 1.0
        if n >= last:
            break

    if n == 0:
        return None

    return {"scored": scored / n, "conceded": conceded / n, "n": n, "opp_ids": opponents}

# =====================================================
# Factors: injuries / fatigue / motivation / SOS + debug
# =====================================================
def get_injuries_raw(fixture_id: int) -> List[Dict[str, Any]]:
    try:
        j = football_get("/injuries", {"fixture": fixture_id}, timeout=25)
        return j.get("response", []) or []
    except:
        return []

def get_injuries_by_fixture(fixture_id: int) -> Dict[int, Dict[str, int]]:
    out: Dict[int, Dict[str, int]] = {}
    resp = get_injuries_raw(fixture_id)

    for it in resp:
        team = it.get("team", {}) or {}
        tid = team.get("id")
        if not tid:
            continue
        player = it.get("player", {}) or {}
        ptype = str(player.get("type") or "").lower()

        is_gk = "goalkeeper" in ptype or ptype in {"gk"}
        is_def = "defender" in ptype or ptype in {"df", "def"}
        is_mid = "midfielder" in ptype or ptype in {"mf", "mid"}
        is_fwd = "attacker" in ptype or "forward" in ptype or ptype in {"fw", "fwd", "st", "striker"}

        if int(tid) not in out:
            out[int(tid)] = {"att": 0, "def": 0}

        if is_gk or is_def:
            out[int(tid)]["def"] += 1
        elif is_mid or is_fwd:
            out[int(tid)]["att"] += 1
        else:
            out[int(tid)]["att"] += 1
    return out

def apply_injury_adjustments(lam_home: float, lam_away: float, injuries: Dict[int, Dict[str, int]], home_id: int, away_id: int) -> Tuple[float, float, str, str]:
    h = injuries.get(int(home_id), {"att": 0, "def": 0})
    a = injuries.get(int(away_id), {"att": 0, "def": 0})

    h_att = min(int(h.get("att", 0)), 3)
    h_def = min(int(h.get("def", 0)), 3)
    a_att = min(int(a.get("att", 0)), 3)
    a_def = min(int(a.get("def", 0)), 3)

    home_attack_mult = 1.0 - 0.02 * h_att
    away_attack_mult = 1.0 - 0.02 * a_att

    home_vs_away_def_mult = 1.0 + 0.02 * a_def
    away_vs_home_def_mult = 1.0 + 0.02 * h_def

    lam_home2 = max(0.05, lam_home * home_attack_mult * home_vs_away_def_mult)
    lam_away2 = max(0.05, lam_away * away_attack_mult * away_vs_home_def_mult)

    why_h = f"injH(att-{2*h_att}%,oppDef+{2*a_def}%)"
    why_a = f"injA(att-{2*a_att}%,oppDef+{2*h_def}%)"
    return lam_home2, lam_away2, why_h, why_a

def get_fatigue_factor(team_id: int, match_ts_utc: int) -> Tuple[float, str, int]:
    try:
        dt_match = datetime.utcfromtimestamp(int(match_ts_utc))
        dt_from = (dt_match - timedelta(days=7)).date().isoformat()
        dt_to = dt_match.date().isoformat()
        j = football_get("/fixtures", {"team": team_id, "from": dt_from, "to": dt_to}, timeout=25)
        resp = j.get("response", []) or []
        played = 0
        for it in resp:
            fx = it.get("fixture", {}) or {}
            st = (fx.get("status", {}) or {}).get("short", "")
            if st in {"FT", "AET", "PEN"}:
                played += 1
        if played >= 3:
            return 0.95, "fatigue=-5%", played
        if played == 2:
            return 0.97, "fatigue=-3%", played
        return 1.00, "fatigue=0%", played
    except:
        return 1.00, "fatigue=0%", 0

def get_fatigue_debug(team_id: int, match_ts_utc: int) -> Dict[str, Any]:
    out = {"team_id": team_id, "ok": False}
    try:
        dt_match = datetime.utcfromtimestamp(int(match_ts_utc))
        dt_from = (dt_match - timedelta(days=7)).date().isoformat()
        dt_to = dt_match.date().isoformat()
        j = football_get("/fixtures", {"team": team_id, "from": dt_from, "to": dt_to}, timeout=25)
        resp = j.get("response", []) or []
        played = 0
        statuses: Dict[str, int] = {}
        for it in resp:
            fx = it.get("fixture", {}) or {}
            st = (fx.get("status", {}) or {}).get("short", "") or "?"
            statuses[st] = statuses.get(st, 0) + 1
            if st in {"FT", "AET", "PEN"}:
                played += 1
        k, why, _played = get_fatigue_factor(team_id, match_ts_utc)
        out.update({
            "ok": True,
            "from": dt_from,
            "to": dt_to,
            "resp_count": len(resp),
            "played_finished": played,
            "factor": k,
            "why": why,
            "statuses": statuses
        })
        return out
    except Exception as e:
        out["error"] = str(e)
        return out

def get_standings_map(league_id: int) -> Dict[int, Dict[str, Any]]:
    key = f"standings_{league_id}_{SEASON}"
    today = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d")
    cache = STATE.get(key)
    cache_date = STATE.get(key + "_date")
    if isinstance(cache, dict) and cache_date == today:
        try:
            return {int(k): v for k, v in cache.items()}
        except:
            pass

    out: Dict[int, Dict[str, Any]] = {}
    try:
        j = football_get("/standings", {"league": league_id, "season": SEASON}, timeout=25)
        resp = j.get("response", []) or []
        if not resp:
            return out
        league = resp[0].get("league", {}) or {}
        standings = league.get("standings", []) or []
        for group in standings:
            for row in group:
                team = row.get("team", {}) or {}
                tid = team.get("id")
                if not tid:
                    continue
                out[int(tid)] = {"rank": int(row.get("rank") or 0), "points": int(row.get("points") or 0)}
    except:
        return out

    STATE[key] = {str(k): v for k, v in out.items()}
    STATE[key + "_date"] = today
    save_state(STATE)
    return out

def standings_debug(league_id: Optional[int]) -> Dict[str, Any]:
    out = {"league_id": league_id, "ok": False}
    if not league_id:
        out["note"] = "no league_id"
        return out
    try:
        smap = get_standings_map(int(league_id))
        out["ok"] = True
        out["teams"] = len(smap)
        if smap:
            pts = [int(v.get("points") or 0) for v in smap.values()]
            out["points_min"] = min(pts) if pts else None
            out["points_max"] = max(pts) if pts else None
        return out
    except Exception as e:
        out["error"] = str(e)
        return out

def strength_of_schedule_factor(league_id: Optional[int], opp_ids: List[int]) -> Tuple[float, str]:
    if not league_id or not opp_ids:
        return 1.0, "sos=0%"
    smap = get_standings_map(int(league_id))
    if not smap:
        return 1.0, "sos=0%"

    pts = []
    for oid in opp_ids:
        row = smap.get(int(oid))
        if row and row.get("points") is not None:
            pts.append(int(row["points"]))
    if len(pts) < 3:
        return 1.0, "sos=0%"

    league_pts = [int(v.get("points") or 0) for v in smap.values() if v.get("points") is not None]
    if not league_pts:
        return 1.0, "sos=0%"

    avg_opp = sum(pts) / len(pts)
    avg_lg = sum(league_pts) / len(league_pts)
    if avg_lg <= 0:
        return 1.0, "sos=0%"

    diff = (avg_opp - avg_lg) / avg_lg
    diff = max(-0.15, min(0.15, diff))
    k = 1.0 + 0.04 * diff
    pct = int(round((k - 1.0) * 100))
    return k, f"sos={pct:+d}%"

def sos_debug(league_id: Optional[int], opp_ids: List[int]) -> Dict[str, Any]:
    out = {"league_id": league_id, "ok": False, "opp_ids": len(opp_ids or [])}
    if not league_id:
        out["note"] = "no league_id"
        return out
    try:
        smap = get_standings_map(int(league_id))
        out["standings_teams"] = len(smap)
        if not smap:
            out["note"] = "standings empty"
            return out

        pts = []
        for oid in (opp_ids or []):
            row = smap.get(int(oid))
            if row and row.get("points") is not None:
                pts.append(int(row["points"]))
        out["opp_pts_count"] = len(pts)
        if len(pts) < 3:
            out["note"] = "opp points < 3"
            return out

        league_pts = [int(v.get("points") or 0) for v in smap.values() if v.get("points") is not None]
        if not league_pts:
            out["note"] = "league points empty"
            return out

        avg_opp = sum(pts) / len(pts)
        avg_lg = sum(league_pts) / len(league_pts)
        k, why = strength_of_schedule_factor(league_id, opp_ids or [])
        out.update({"ok": True, "avg_opp": avg_opp, "avg_league": avg_lg, "factor": k, "why": why})
        return out
    except Exception as e:
        out["error"] = str(e)
        return out

def get_motivation_factor(league_name: str, league_id: Optional[int], home_id: int, away_id: int) -> Tuple[float, float, str, str]:
    if is_cup_like(league_name):
        return 1.03, 1.03, "mot(cup)=+3%", "mot(cup)=+3%"

    if not league_id:
        return 1.00, 1.00, "mot=0%", "mot=0%"

    smap = get_standings_map(int(league_id))
    h = smap.get(int(home_id))
    a = smap.get(int(away_id))
    if not h or not a:
        return 1.00, 1.00, "mot=0%", "mot=0%"

    rows = list(smap.values())
    rank_points = {int(r["rank"]): int(r["points"]) for r in rows if int(r.get("rank", 0)) > 0}
    pts_1 = rank_points.get(1)
    pts_4 = rank_points.get(4)
    pts_17 = rank_points.get(17)

    def compute(team_row: Dict[str, Any]) -> Tuple[float, str]:
        rank = int(team_row.get("rank") or 0)
        pts = int(team_row.get("points") or 0)
        f = 1.00
        reasons = []

        if pts_1 is not None and rank <= 6 and (pts_1 - pts) <= 3:
            f *= 1.03
            reasons.append("title")
        if pts_4 is not None and rank <= 8 and abs(pts_4 - pts) <= 3:
            f *= 1.03
            reasons.append("top4")
        if pts_17 is not None and rank >= 13 and abs(pts - pts_17) <= 3:
            f *= 1.03
            reasons.append("survival")

        if not reasons:
            return 1.00, "mot=0%"
        return f, "mot(" + "+".join(reasons) + ")+3%"

    fh, wh = compute(h)
    fa, wa = compute(a)
    return fh, fa, wh, wa

# =====================================================
# Poisson + probabilities
# =====================================================
def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)

def poisson_cdf(k: int, lam: float) -> float:
    s = 0.0
    for i in range(0, k + 1):
        s += poisson_pmf(i, lam)
    return min(max(s, 0.0), 1.0)

def prob_over_line_poisson(lam: float, line: float) -> float:
    k = int(math.floor(line))
    p = 1.0 - poisson_cdf(k, lam)
    return clamp(p, 0.05, 0.95)

# =====================================================
# API-Football: league ids + fixtures today filtered
# =====================================================
def leagues_search(search_text: str, country: Optional[str]) -> List[Dict[str, Any]]:
    params = {"search": search_text}
    if country:
        params["country"] = country
    try:
        j = football_get("/leagues", params, timeout=25)
        return j.get("response", [])
    except:
        return []

def score_candidate(target_country: Optional[str], aliases: List[str], cand_name: str, cand_country: str) -> int:
    tn = cand_name.strip().lower()
    tc = cand_country.strip().lower()
    score = 0
    if target_country:
        score += 50 if tc == target_country.strip().lower() else -10
    for a in aliases:
        al = a.strip().lower()
        if tn == al: score += 100
        elif al in tn: score += 60
        elif tn in al: score += 40
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
                results = leagues_search(alias, ctry)
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
    j = football_get("/fixtures", params, timeout=25)
    return j.get("response", [])

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
# OddsPapi
# =====================================================
def oddspapi_get(path: str, params: Dict[str, Any]) -> Any:
    params = dict(params)
    params["apiKey"] = ODDSPAPI_KEY
    url = f"https://api.oddspapi.io{path}"

    for _ in range(7):
        r = requests.get(url, params=params, timeout=30)
        if r.status_code == 429:
            try:
                j = r.json()
                retry_ms = int(j.get("error", {}).get("retryMs", 250))
            except:
                retry_ms = 250
            time.sleep(max(0.06, retry_ms / 1000.0))
            continue
        if r.status_code >= 400:
            raise RuntimeError(f"OddsPapi HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    raise RuntimeError("OddsPapi: слишком много 429 подряд, попробуй позже.")

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
        time.sleep(0.09)
    return all_items

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

        meta[mid] = {"name": str(m.get("marketName") or ""), "handicap": m.get("handicap"), "outcomes": outs}

    STATE[MARKETS_CACHE_KEY] = {str(k): v for k, v in meta.items()}
    STATE[MARKETS_CACHE_DATE] = today
    save_state(STATE)
    return meta

# =====================================================
# Odds parsing: ONLY Full Time OU 1.5 / 2.5 / 3.5
# =====================================================
OU_FT_NAME = "over under full time"

def parse_odds(item: Dict[str, Any], markets_meta: Dict[int, Dict[str, Any]]) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {
        "O15": None, "U15": None,
        "O25": None, "U25": None,
        "O35": None, "U35": None,
    }

    bm = ((item.get("bookmakerOdds") or {}).get(ODDSPAPI_BOOKMAKER) or {})
    markets = bm.get("markets") or {}

    def price_from_players(players_obj: Dict[str, Any]) -> Optional[float]:
        p0 = players_obj.get("0") or players_obj.get(0) or {}
        try:
            return float(p0.get("price"))
        except:
            return None

    def set_ou(line: float, oname: str, price: float):
        n = (oname or "").lower()
        if line == 1.5:
            if "over" in n: out["O15"] = price
            if "under" in n: out["U15"] = price
        if line == 2.5:
            if "over" in n: out["O25"] = price
            if "under" in n: out["U25"] = price
        if line == 3.5:
            if "over" in n: out["O35"] = price
            if "under" in n: out["U35"] = price

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

        if OU_FT_NAME not in mname:
            continue

        try:
            line = float(hcap)
        except:
            continue
        if line not in (1.5, 2.5, 3.5):
            continue

        for oid_str, o in outcomes.items():
            try:
                oid = int(oid_str)
            except:
                continue
            price = price_from_players(o.get("players") or {})
            if price is None:
                continue
            oname = outcomes_meta.get(oid, "") or ""
            set_ou(line, oname, price)

    return out

# =====================================================
# Odds cache + matching
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
# Settlement helpers
# =====================================================
def fetch_fixtures_by_ids(ids: List[int]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not ids:
        return out
    CH = 20
    for i in range(0, len(ids), CH):
        chunk = ids[i:i+CH]
        id_param = "-".join(str(x) for x in chunk)
        try:
            j = football_get("/fixtures", {"id": id_param}, timeout=25)
            resp = j.get("response", []) or []
            for it in resp:
                fx = it.get("fixture", {}) or {}
                fid = fx.get("id")
                if fid is not None:
                    out[int(fid)] = it
        except Exception as e:
            print(f"[ERROR] fetch_fixtures_by_ids chunk failed: {e}")
        time.sleep(0.08)
    return out

def is_finished_status(short: str) -> bool:
    return (short or "").upper() in {"FT", "AET", "PEN"}

def settle_result_for_market(market_code: str, hg: int, ag: int) -> str:
    total = hg + ag

    if market_code == "O15":
        return "WIN" if total >= 2 else "LOSE"
    if market_code == "U15":
        return "WIN" if total <= 1 else "LOSE"

    if market_code == "O25":
        return "WIN" if total >= 3 else "LOSE"
    if market_code == "U25":
        return "WIN" if total <= 2 else "LOSE"

    if market_code == "O35":
        return "WIN" if total >= 4 else "LOSE"
    if market_code == "U35":
        return "WIN" if total <= 3 else "LOSE"

    return "PUSH"

def profit_for_result(result: str, odds: float, stake: float) -> float:
    if result == "WIN":
        return (odds - 1.0) * stake
    if result == "LOSE":
        return -1.0 * stake
    return 0.0

# =====================================================
# Core factor computation per fixture (lambda model)
# =====================================================
def compute_lambdas_and_factors(f: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
    fixture = f.get("fixture") or {}
    teams = f.get("teams") or {}
    home_t = teams.get("home") or {}
    away_t = teams.get("away") or {}
    league_obj = f.get("league") or {}

    fixture_id = fixture.get("id")
    home_id = home_t.get("id")
    away_id = away_t.get("id")
    ts = int(fixture.get("timestamp") or 0)
    league_name = league_obj.get("name", "?")
    league_id = league_obj.get("id")

    if not fixture_id or not home_id or not away_id or not ts:
        return None, None, "Factors: (no ids)"

    home_stats = team_goal_stats_venue(int(home_id), venue="home", last=LAST_MATCHES_TEAM)
    away_stats = team_goal_stats_venue(int(away_id), venue="away", last=LAST_MATCHES_TEAM)
    if not home_stats or not away_stats:
        return None, None, "Factors: (no venue stats)"

    lam_home = max(0.05, (home_stats["scored"] + away_stats["conceded"]) / 2.0)
    lam_away = max(0.05, (away_stats["scored"] + home_stats["conceded"]) / 2.0)

    sos_h_k, sos_h_why = strength_of_schedule_factor(league_id, home_stats.get("opp_ids") or [])
    sos_a_k, sos_a_why = strength_of_schedule_factor(league_id, away_stats.get("opp_ids") or [])
    lam_home *= sos_h_k
    lam_away *= sos_a_k

    venue = fixture.get("venue") or {}
    city = venue.get("city")
    w_k, w_why = weather_factor(city)
    lam_home *= w_k
    lam_away *= w_k

    injuries_map = get_injuries_by_fixture(int(fixture_id))
    lam_home, lam_away, inj_h, inj_a = apply_injury_adjustments(lam_home, lam_away, injuries_map, int(home_id), int(away_id))

    fat_h_k, fat_h_why, fat_h_n = get_fatigue_factor(int(home_id), ts)
    fat_a_k, fat_a_why, fat_a_n = get_fatigue_factor(int(away_id), ts)
    lam_home *= fat_h_k
    lam_away *= fat_a_k

    mot_h_k, mot_a_k, mot_h_why, mot_a_why = get_motivation_factor(league_name, league_id, int(home_id), int(away_id))
    lam_home *= mot_h_k
    lam_away *= mot_a_k

    lam_home = max(0.05, min(lam_home, 3.5))
    lam_away = max(0.05, min(lam_away, 3.5))

    factors_line = (
        f"Factors: {inj_h}, {inj_a}; "
        f"{fat_h_why}({fat_h_n}), {fat_a_why}({fat_a_n}); "
        f"{mot_h_why}, {mot_a_why}; {sos_h_why}, {sos_a_why}; {w_why}"
    )
    return lam_home, lam_away, factors_line

# =====================================================
# Telegram commands
# =====================================================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован.\n\n"
        "Рынок: общий тотал голов (1.5 / 2.5 / 3.5)\n\n"
        "Команды:\n"
        "• /signals — value-сигналы (P≥75% и Value>0)\n"
        "• /lines — линии без фильтра\n"
        "• /settle — рассчитать результаты PENDING\n"
        "• /roi — ROI/проходимость\n"
        "• /history — последние 20 сигналов\n"
        "• /oddspapi_account — проверка OddsPapi\n"
        "• /reload_leagues — пересканировать лиги\n"
        "• /factors_debug [n] — debug факторов по n-му матчу дня\n\n"
        f"Bookmaker: {ODDSPAPI_BOOKMAKER}\n"
        f"Пороги: P≥{int(MIN_PROB*100)}% и Value>{MIN_VALUE:+.2f}\n"
        f"Анти-анома Value: >{ANOMALY_VALUE_WARN:+.2f} → ⚠️\n"
        f"Stake: {STAKE:.2f} unit\n"
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

# ---------------------------
# /factors_debug [n]
# ---------------------------
async def factors_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await update.message.reply_text(f"⚠️ На дату {used_date} матчей нет.")
        return

    idx = 1
    if context.args:
        try:
            idx = max(1, int(context.args[0]))
        except:
            idx = 1
    if idx > len(fixtures):
        await update.message.reply_text(f"⚠️ В списке только {len(fixtures)} матчей. Ты запросил {idx}.")
        return

    f = fixtures[idx - 1]
    fixture = f.get("fixture") or {}
    teams = f.get("teams") or {}
    home_t = teams.get("home") or {}
    away_t = teams.get("away") or {}
    league_obj = f.get("league") or {}

    fixture_id = int(fixture.get("id") or 0)
    ts = int(fixture.get("timestamp") or 0)
    home_id = int(home_t.get("id") or 0)
    away_id = int(away_t.get("id") or 0)

    if not fixture_id or not ts or not home_id or not away_id:
        await update.message.reply_text("❌ Не хватает данных по матчу (fixture_id/team_id/timestamp).")
        return

    league_name = league_obj.get("name", "?")
    league_id = league_obj.get("id")
    home_name = home_t.get("name", "?")
    away_name = away_t.get("name", "?")
    dt_msk = datetime.utcfromtimestamp(ts) + timedelta(hours=3)

    venue = fixture.get("venue") or {}
    city = venue.get("city")

    # venue stats
    hs = team_goal_stats_venue(home_id, "home", last=LAST_MATCHES_TEAM)
    as_ = team_goal_stats_venue(away_id, "away", last=LAST_MATCHES_TEAM)

    # injuries raw
    inj_raw = get_injuries_raw(fixture_id)
    inj_map = get_injuries_by_fixture(fixture_id)
    h_inj = inj_map.get(home_id, {"att": 0, "def": 0})
    a_inj = inj_map.get(away_id, {"att": 0, "def": 0})

    # fatigue debug
    fat_h = get_fatigue_debug(home_id, ts)
    fat_a = get_fatigue_debug(away_id, ts)

    # standings + sos debug
    st_dbg = standings_debug(league_id)
    sos_h_dbg = sos_debug(league_id, (hs or {}).get("opp_ids") or [])
    sos_a_dbg = sos_debug(league_id, (as_ or {}).get("opp_ids") or [])

    # weather debug
    w_dbg = weather_factor_debug(city)

    # final factors line
    lam_home, lam_away, factors_line = compute_lambdas_and_factors(f)

    lines = []
    lines.append(f"🧪 FACTORS DEBUG ({used_date}) матч #{idx}/{len(fixtures)}")
    lines.append(f"🏆 {league_name} (league_id={league_id})")
    lines.append(f"{home_name} — {away_name}")
    lines.append(f"🕒 {dt_msk.strftime('%Y-%m-%d %H:%M')} МСК | fixture_id={fixture_id}")
    lines.append("")

    # venue stats
    if hs and as_:
        lines.append("1) Venue stats (последние матчи по месту):")
        lines.append(f"Home venue stats: n={int(hs['n'])} scored={hs['scored']:.2f} conceded={hs['conceded']:.2f} opp_ids={len(hs.get('opp_ids') or [])}")
        lines.append(f"Away venue stats: n={int(as_['n'])} scored={as_['scored']:.2f} conceded={as_['conceded']:.2f} opp_ids={len(as_.get('opp_ids') or [])}")
    else:
        lines.append("1) Venue stats: ❌ не смог собрать (нет матчей/данных)")

    lines.append("")
    # injuries
    lines.append("2) Injuries:")
    lines.append(f"API /injuries response count: {len(inj_raw)}")
    lines.append(f"Parsed injuries home(att/def): {h_inj.get('att',0)}/{h_inj.get('def',0)} | away(att/def): {a_inj.get('att',0)}/{a_inj.get('def',0)}")
    if inj_raw:
        lines.append("Sample (до 5 строк):")
        for it in inj_raw[:5]:
            tname = ((it.get("team") or {}).get("name") or "?")
            pname = ((it.get("player") or {}).get("name") or "?")
            ptype = ((it.get("player") or {}).get("type") or "?")
            reason = ((it.get("player") or {}).get("reason") or "")
            lines.append(f"- {tname}: {pname} | type={ptype} | {reason}")
    else:
        lines.append("NOTE: если тут 0 — значит API реально не отдаёт injuries по этому матчу (поэтому inj=0%).")

    lines.append("")
    # fatigue
    lines.append("3) Fatigue (окно 7 дней):")
    for tag, d in [("Home", fat_h), ("Away", fat_a)]:
        if not d.get("ok"):
            lines.append(f"{tag}: ❌ ошибка/нет данных: {d.get('error','?')}")
            continue
        lines.append(f"{tag}: from={d.get('from')} to={d.get('to')} resp={d.get('resp_count')} finished={d.get('played_finished')} factor={d.get('factor'):.3f} {d.get('why')}")
        st = d.get("statuses") or {}
        top = sorted(st.items(), key=lambda x: -x[1])[:6]
        lines.append(f"{tag}: statuses sample: " + ", ".join([f"{k}:{v}" for k,v in top]))

    lines.append("")
    # standings/sos
    lines.append("4) Standings / SOS:")
    if st_dbg.get("ok"):
        lines.append(f"Standings teams: {st_dbg.get('teams')} | points min/max: {st_dbg.get('points_min')}/{st_dbg.get('points_max')}")
    else:
        lines.append(f"Standings: ❌ {st_dbg.get('note') or st_dbg.get('error') or 'no data'}")

    for tag, d in [("SOS home", sos_h_dbg), ("SOS away", sos_a_dbg)]:
        if d.get("ok"):
            lines.append(f"{tag}: opp_ids={d.get('opp_ids')} opp_pts={d.get('opp_pts_count')} avg_opp={d.get('avg_opp'):.2f} avg_lg={d.get('avg_league'):.2f} factor={d.get('factor'):.4f} {d.get('why')}")
        else:
            lines.append(f"{tag}: ❌ {d.get('note') or d.get('error') or 'no data'} (opp_ids={d.get('opp_ids')}, standings_teams={d.get('standings_teams')})")

    lines.append("")
    # weather
    lines.append("5) Weather:")
    lines.append(f"Venue city: {city}")
    if w_dbg.get("ok"):
        lines.append(f"temp={w_dbg.get('temp')}C wind={w_dbg.get('wind')}m/s rain1h={w_dbg.get('rain_1h')} | factor={w_dbg.get('factor'):.4f} | {w_dbg.get('reason')}")
    else:
        lines.append(f"weather debug: {w_dbg.get('reason')} | note={w_dbg.get('note') or w_dbg.get('error')}")

    lines.append("")
    # final
    lines.append("6) Итоговая строка Factors и лямбды:")
    lines.append(factors_line)
    if lam_home is not None and lam_away is not None:
        lines.append(f"Model λ: Home={lam_home:.2f} Away={lam_away:.2f} Total={lam_home+lam_away:.2f}")
    else:
        lines.append("Model λ: ❌ не посчиталось")

    for part in chunked("\n".join(lines), 3500):
        await update.message.reply_text(part)

# ---------------------------
# /lines — totals only
# ---------------------------
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

    msg = [f"📌 ЛИНИИ (тоталы) ({used_date})\nBookmaker: {ODDSPAPI_BOOKMAKER}\n\n"]
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
        if not any(v is not None for v in o.values()):
            continue

        sent += 1
        msg.append(
            f"🏆 {league.get('name','?')}\n"
            f"{home.get('name','?')} — {away.get('name','?')}\n"
            f"🕒 {t_str}\n"
            f"ТБ 1.5: {fmt_odds(o['O15'])} | ТМ 1.5: {fmt_odds(o['U15'])}\n"
            f"ТБ 2.5: {fmt_odds(o['O25'])} | ТМ 2.5: {fmt_odds(o['U25'])}\n"
            f"ТБ 3.5: {fmt_odds(o['O35'])} | ТМ 3.5: {fmt_odds(o['U35'])}\n\n"
        )

        batch += 1
        if batch % 5 == 0:
            await update.message.reply_text("".join(msg))
            msg = ["(продолжение)\n\n"]

    if sent == 0:
        await update.message.reply_text(f"📭 На дату {used_date} не нашёл линий тоталов у {ODDSPAPI_BOOKMAKER}.")
        return

    if msg:
        await update.message.reply_text("".join(msg))

# ---------------------------
# /signals — totals only
# ---------------------------
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

    out = [f"⚽ ЦЕРБЕР — value-сигналы (тоталы) ({used_date})\nOdds: {ODDSPAPI_BOOKMAKER}\n\n"]
    found_any = 0
    batch = 0

    def try_emit_signal(fixture_id: int, match_date: str, league: str, home: str, away: str,
                        market_code: str, market_label: str, line: float, side: str,
                        p: float, book: Optional[float]) -> Optional[str]:
        if book is None:
            return None

        ok_odds, odds_warn = odds_sanity(market_code, float(book))

        if p < MIN_PROB:
            return None

        v = value_ev(p, float(book))
        if v <= MIN_VALUE:
            return None

        fair = fair_odds(p)

        value_warn = "⚠️ проверь линию вручную" if v > ANOMALY_VALUE_WARN else ""
        warns = " ".join([w for w in [value_warn, odds_warn] if w]).strip()
        warns = f" {warns}" if warns else ""

        text = f"{market_label}: P={int(p*100)}% | Book={float(book):.2f} | Fair={fair:.2f} | Value={v:+.2f}{warns}"

        # Сохраняем только если odds нормальные
        if ok_odds:
            db_upsert_signal({
                "created_at": datetime.utcnow().isoformat(),
                "match_date": match_date,
                "fixture_id": fixture_id,
                "league": league,
                "home": home,
                "away": away,
                "market_code": market_code,
                "market_label": market_label,
                "line": float(line),
                "side": side,
                "p_model": p,
                "book_odds": float(book),
                "fair_odds": fair,
                "value_ev": v,
                "stake": STAKE,
            })

        return text

    for f in fixtures:
        fixture = f.get("fixture") or {}
        teams = f.get("teams") or {}
        home_t = teams.get("home") or {}
        away_t = teams.get("away") or {}
        league_obj = f.get("league") or {}

        fixture_id = fixture.get("id")
        if not fixture_id:
            continue

        ts = int(fixture.get("timestamp") or 0)
        if not ts:
            continue

        ts_msk = int((datetime.utcfromtimestamp(ts) + timedelta(hours=3)).timestamp())
        t_str = (datetime.utcfromtimestamp(ts) + timedelta(hours=3)).strftime("%H:%M МСК")

        home_name = home_t.get("name", "?")
        away_name = away_t.get("name", "?")
        league_name = league_obj.get("name", "?")

        it = match_odds_for_fixture(used_date, ts_msk, home_name, away_name, odds_map, odds_index)
        if not it:
            continue

        odds = parse_odds(it, markets_meta)
        if not any(v is not None for v in odds.values()):
            continue

        lam_home, lam_away, factors_line = compute_lambdas_and_factors(f)
        if lam_home is None or lam_away is None:
            continue

        lam_total = lam_home + lam_away

        p_o15 = prob_over_line_poisson(lam_total, 1.5)
        p_u15 = clamp(1.0 - p_o15, 0.05, 0.95)

        p_o25 = prob_over_line_poisson(lam_total, 2.5)
        p_u25 = clamp(1.0 - p_o25, 0.05, 0.95)

        p_o35 = prob_over_line_poisson(lam_total, 3.5)
        p_u35 = clamp(1.0 - p_o35, 0.05, 0.95)

        markets = [
            ("O15", "ТБ 1.5", 1.5, "O", p_o15, odds.get("O15")),
            ("U15", "ТМ 1.5", 1.5, "U", p_u15, odds.get("U15")),
            ("O25", "ТБ 2.5", 2.5, "O", p_o25, odds.get("O25")),
            ("U25", "ТМ 2.5", 2.5, "U", p_u25, odds.get("U25")),
            ("O35", "ТБ 3.5", 3.5, "O", p_o35, odds.get("O35")),
            ("U35", "ТМ 3.5", 3.5, "U", p_u35, odds.get("U35")),
        ]

        lines_out: List[str] = []
        for code, label, line, side, p, book in markets:
            s = try_emit_signal(int(fixture_id), datetime.utcfromtimestamp(ts).isoformat(), league_name, home_name, away_name,
                                code, label, line, side, p, book)
            if s:
                lines_out.append(s)

        if not lines_out:
            continue

        found_any += 1
        out.append(
            f"🏆 {league_name}\n"
            f"{home_name} — {away_name}\n"
            f"🕒 {t_str}\n"
            f"{factors_line}\n"
            f"Model λ: Home={lam_home:.2f} | Away={lam_away:.2f} | Total={lam_total:.2f}\n"
            + "\n".join(lines_out)
            + "\n\n"
        )

        batch += 1
        if batch % 5 == 0:
            await update.message.reply_text("".join(out))
            out = ["(продолжение)\n\n"]

    if found_any == 0:
        await update.message.reply_text(
            f"📭 На дату {used_date} нет value-сигналов по тоталам при порогах:\n"
            f"P ≥ {int(MIN_PROB*100)}% и Value > {MIN_VALUE:+.2f}\n\n"
            "Подсказка: /lines покажет коэффициенты без фильтра."
        )
        return

    if out:
        await update.message.reply_text("".join(out))

# ---------------------------
# /settle
# ---------------------------
async def settle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pending = db_fetch_pending(limit=250)
    if not pending:
        await update.message.reply_text("✅ Нет PENDING сигналов. Всё рассчитано.")
        return

    fixture_ids = sorted(list(set(int(s["fixture_id"]) for s in pending)))
    fx_map = fetch_fixtures_by_ids(fixture_ids)

    settled_count = 0
    not_finished = 0
    missing = 0

    for s in pending:
        fid = int(s["fixture_id"])
        fx = fx_map.get(fid)
        if not fx:
            missing += 1
            continue

        fixture = fx.get("fixture", {}) or {}
        status = (fixture.get("status", {}) or {}).get("short", "")
        if not is_finished_status(status):
            not_finished += 1
            continue

        goals = fx.get("goals", {}) or {}
        hg = goals.get("home")
        ag = goals.get("away")
        if hg is None or ag is None:
            not_finished += 1
            continue

        hg = int(hg)
        ag = int(ag)

        market_code = s["market_code"]
        res = settle_result_for_market(market_code, hg, ag)
        prof = profit_for_result(res, float(s["book_odds"]), float(s["stake"]))
        db_settle_signal(int(s["id"]), res, prof)
        settled_count += 1

    await update.message.reply_text(
        "✅ SETTLE готово.\n"
        f"Рассчитано: {settled_count}\n"
        f"Ещё не завершились: {not_finished}\n"
        f"Не найдены в API: {missing}"
    )

# ---------------------------
# /roi
# ---------------------------
async def roi(update: Update, context: ContextTypes.DEFAULT_TYPE):
    s7 = db_stats(7)
    s30 = db_stats(30)
    sall = db_stats(None)

    def line(title: str, s: Dict[str, Any]) -> str:
        return (
            f"{title}\n"
            f"• Ставок: {s['n']} (W {s['wins']} / L {s['loses']} / P {s['pushes']})\n"
            f"• Проходимость: {int(s['hit']*100)}%\n"
            f"• Profit: {s['profit_sum']:.2f}u на {s['stake_sum']:.2f}u\n"
            f"• ROI: {s['roi']:.2f}%\n"
        )

    await update.message.reply_text(
        "📈 ROI / Проходимость (фикс 1 unit)\n\n"
        + line("🗓️ 7 дней:", s7) + "\n"
        + line("🗓️ 30 дней:", s30) + "\n"
        + line("🧾 Всё время:", sall)
        + "\nПодсказка: сначала /settle, чтобы обновить результаты."
    )

# ---------------------------
# /history
# ---------------------------
async def history(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = db_last_history(20)
    if not rows:
        await update.message.reply_text("Пока нет истории сигналов.")
        return

    msg = ["🧾 Последние сигналы:\n\n"]
    for r in rows:
        res = r.get("result") or "PENDING"
        prof = r.get("profit")
        prof_s = "—" if prof is None else f"{prof:+.2f}u"
        msg.append(
            f"{(r.get('league') or '?')} | {r.get('home') or '?'} — {r.get('away') or '?'}\n"
            f"{r.get('market_label')} | Book={float(r.get('book_odds') or 0):.2f} | P={int(float(r.get('p_model') or 0)*100)}% | Value={float(r.get('value_ev') or 0):+.2f}\n"
            f"Result: {res} | Profit: {prof_s}\n\n"
        )

    for part in chunked("".join(msg)):
        await update.message.reply_text(part)

# =====================================================
# RUN
# =====================================================
def main():
    db_init()
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))
    app.add_handler(CommandHandler("lines", lines))
    app.add_handler(CommandHandler("settle", settle))
    app.add_handler(CommandHandler("roi", roi))
    app.add_handler(CommandHandler("history", history))
    app.add_handler(CommandHandler("oddspapi_account", oddspapi_account))
    app.add_handler(CommandHandler("reload_leagues", reload_leagues))
    app.add_handler(CommandHandler("factors_debug", factors_debug))

    app.run_polling()

if __name__ == "__main__":
    main()
