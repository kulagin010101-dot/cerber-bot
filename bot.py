import os
import json
import time
import math
import sqlite3
import unicodedata
import requests
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List, Tuple

from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# =====================================================
# ENV
# =====================================================
BOT_TOKEN = os.getenv("BOT_TOKEN")
FOOTBALL_API_KEY = os.getenv("FOOTBALL_API_KEY")
ODDSPAPI_KEY = os.getenv("ODDSPAPI_KEY")

# bookmaker in your Oddspapi plan: only 1xbet доступен
ODDSPAPI_BOOKMAKER = os.getenv("ODDSPAPI_BOOKMAKER", "1xbet")

# optional weather (OpenWeather)
WEATHER_KEY = os.getenv("WEATHER_API_KEY")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не задан")
if not FOOTBALL_API_KEY:
    raise RuntimeError("FOOTBALL_API_KEY не задан")
if not ODDSPAPI_KEY:
    raise RuntimeError("ODDSPAPI_KEY не задан")

HEADERS_FOOTBALL = {"x-apisports-key": FOOTBALL_API_KEY}

SEASON = int(os.getenv("SEASON", "2025"))

# thresholds for signals (P_final and Value)
MIN_PROB = float(os.getenv("MIN_PROB", "0.75"))     # 75%
MIN_VALUE = float(os.getenv("MIN_VALUE", "0.00"))   # EV>0

# stake in units (fixed)
STAKE = float(os.getenv("STAKE", "1.0"))

# Blend: P_final = W_MARKET * P_market + (1-W_MARKET) * P_model
W_MARKET = float(os.getenv("W_MARKET", "0.65"))
W_MARKET = max(0.0, min(1.0, W_MARKET))

# model configuration
LAST_MATCHES_TEAM = int(os.getenv("LAST_MATCHES_TEAM", "8"))
FATIGUE_WINDOW_DAYS = int(os.getenv("FATIGUE_WINDOW_DAYS", "7"))
FATIGUE_LAST_N = int(os.getenv("FATIGUE_LAST_N", "10"))
INJ_TTL = int(os.getenv("INJ_TTL_SECONDS", str(6 * 3600)))  # 6h

# Anti-anomalies
ANOMALY_VALUE_WARN = float(os.getenv("ANOMALY_VALUE_WARN", "0.50"))

# Odds sanity limits for totals
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

def _has_column(con: sqlite3.Connection, table: str, column: str) -> bool:
    cur = con.cursor()
    cur.execute(f"PRAGMA table_info({table})")
    cols = [r[1] for r in cur.fetchall()]
    return column in cols

def db_init() -> None:
    con = db_connect()
    cur = con.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS signals (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      created_at TEXT NOT NULL,
      match_date TEXT NOT NULL,
      fixture_id INTEGER NOT NULL,
      kickoff_ts_utc INTEGER,
      league TEXT,
      home TEXT,
      away TEXT,
      market_code TEXT NOT NULL,
      market_label TEXT NOT NULL,
      line REAL,
      side TEXT,
      p_model REAL NOT NULL,
      p_market REAL,
      p_final REAL NOT NULL,
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

    cur.execute("""
    CREATE TABLE IF NOT EXISTS odds_history (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      observed_at TEXT NOT NULL,
      fixture_id INTEGER,
      kickoff_ts_utc INTEGER,
      league TEXT,
      home TEXT,
      away TEXT,
      bookmaker TEXT NOT NULL,
      market_code TEXT NOT NULL,
      line REAL,
      side TEXT,
      odds REAL NOT NULL
    );
    """)
    cur.execute("""
    CREATE INDEX IF NOT EXISTS ix_odds_history_fixture_market
    ON odds_history(fixture_id, market_code, observed_at);
    """)

    # migrations (safe)
    if not _has_column(con, "signals", "kickoff_ts_utc"):
        cur.execute("ALTER TABLE signals ADD COLUMN kickoff_ts_utc INTEGER;")
    if not _has_column(con, "signals", "p_market"):
        cur.execute("ALTER TABLE signals ADD COLUMN p_market REAL;")
    if not _has_column(con, "signals", "p_final"):
        cur.execute("ALTER TABLE signals ADD COLUMN p_final REAL;")

    con.commit()
    con.close()

def db_insert_odds_snapshot(row: Dict[str, Any]) -> None:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
      INSERT INTO odds_history (
        observed_at, fixture_id, kickoff_ts_utc, league, home, away,
        bookmaker, market_code, line, side, odds
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        row["observed_at"],
        row.get("fixture_id"),
        row.get("kickoff_ts_utc"),
        row.get("league"),
        row.get("home"),
        row.get("away"),
        row["bookmaker"],
        row["market_code"],
        row.get("line"),
        row.get("side"),
        float(row["odds"]),
    ))
    con.commit()
    con.close()

def db_upsert_signal(payload: Dict[str, Any]) -> None:
    con = db_connect()
    cur = con.cursor()
    cur.execute("""
    INSERT OR IGNORE INTO signals (
      created_at, match_date, fixture_id, kickoff_ts_utc, league, home, away,
      market_code, market_label, line, side,
      p_model, p_market, p_final, book_odds, fair_odds, value_ev, stake, status
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PENDING')
    """, (
        payload["created_at"], payload["match_date"], payload["fixture_id"], payload.get("kickoff_ts_utc"),
        payload.get("league"), payload.get("home"), payload.get("away"),
        payload["market_code"], payload["market_label"], payload.get("line"), payload.get("side"),
        float(payload["p_model"]), payload.get("p_market"), float(payload["p_final"]),
        float(payload["book_odds"]), float(payload["fair_odds"]), float(payload["value_ev"]),
        float(payload["stake"]),
    ))
    con.commit()
    con.close()

def db_fetch_pending(limit: int = 300) -> List[Dict[str, Any]]:
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
    """, (result, float(profit), datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(), sig_id))
    con.commit()
    con.close()

def db_stats(days: Optional[int]) -> Dict[str, Any]:
    con = db_connect()
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    if days is None:
        cur.execute("""SELECT result, profit, stake FROM signals WHERE status='SETTLED'""")
    else:
        since = (datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=days)).isoformat()
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
      SELECT created_at, match_date, league, home, away, market_label, book_odds, p_final, p_model, p_market, value_ev, result, profit
      FROM signals
      ORDER BY id DESC
      LIMIT ?
    """, (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    con.close()
    return rows

def db_clv_report(days: int = 7) -> Dict[str, Any]:
    """
    CLV = (taken_odds / closing_odds) - 1
    closing_odds = последняя цена из odds_history до kickoff.
    """
    con = db_connect()
    con.row_factory = sqlite3.Row
    cur = con.cursor()
    since = (datetime.utcnow().replace(tzinfo=timezone.utc) - timedelta(days=days)).isoformat()
    cur.execute("""
      SELECT id, created_at, fixture_id, kickoff_ts_utc, league, home, away, market_code, market_label, book_odds
      FROM signals
      WHERE created_at >= ?
      ORDER BY id DESC
      LIMIT 500
    """, (since,))
    sigs = [dict(r) for r in cur.fetchall()]

    total = 0
    matched = 0
    clv_vals = []
    samples = []

    for s in sigs:
        total += 1
        fid = s.get("fixture_id")
        mcode = s.get("market_code")
        taken = float(s.get("book_odds") or 0.0)
        kickoff = s.get("kickoff_ts_utc")

        if not fid or not mcode or taken <= 1.0:
            continue

        if kickoff:
            cutoff_iso = datetime.fromtimestamp(int(kickoff), tz=timezone.utc).isoformat()
            cur.execute("""
              SELECT odds, observed_at
              FROM odds_history
              WHERE fixture_id=? AND market_code=? AND observed_at <= ?
              ORDER BY observed_at DESC
              LIMIT 1
            """, (int(fid), str(mcode), cutoff_iso))
        else:
            cur.execute("""
              SELECT odds, observed_at
              FROM odds_history
              WHERE fixture_id=? AND market_code=?
              ORDER BY observed_at DESC
              LIMIT 1
            """, (int(fid), str(mcode)))

        row = cur.fetchone()
        if not row:
            continue

        closing = float(row["odds"])
        if closing <= 1.0:
            continue

        clv = (taken / closing) - 1.0
        clv_vals.append(clv)
        matched += 1

        if len(samples) < 5:
            samples.append({
                "match": f"{s.get('home','?')} — {s.get('away','?')}",
                "market": s.get("market_label"),
                "taken": taken,
                "closing": closing,
                "clv": clv,
                "observed_at": row["observed_at"],
            })

    con.close()

    avg = sum(clv_vals) / len(clv_vals) if clv_vals else 0.0
    med = sorted(clv_vals)[len(clv_vals)//2] if clv_vals else 0.0
    pos = sum(1 for x in clv_vals if x > 0) if clv_vals else 0
    neg = sum(1 for x in clv_vals if x < 0) if clv_vals else 0

    return {
        "days": days,
        "signals": total,
        "matched": matched,
        "avg": avg,
        "median": med,
        "pos": pos,
        "neg": neg,
        "samples": samples
    }

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
# Anti-anomaly: odds sanity
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
# Vig removal
# =====================================================
def vig_free_probs(odds_over: Optional[float], odds_under: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if not odds_over or not odds_under:
        return None, None, None
    if odds_over <= 1.0 or odds_under <= 1.0:
        return None, None, None
    po_raw = 1.0 / odds_over
    pu_raw = 1.0 / odds_under
    s = po_raw + pu_raw
    if s <= 0:
        return None, None, None
    po = po_raw / s
    pu = pu_raw / s
    return clamp(po, 0.01, 0.99), clamp(pu, 0.01, 0.99), s

def p_final(p_model: float, p_market: Optional[float]) -> float:
    if p_market is None:
        return clamp(p_model, 0.05, 0.95)
    return clamp(W_MARKET * p_market + (1.0 - W_MARKET) * p_model, 0.05, 0.95)

# =====================================================
# Weather (optional)
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
        pct = (factor - 1.0) * 100.0
        return factor, f"weather={pct:+.1f}% ({'/'.join(notes) if notes else 'ok'})"
    except:
        return 1.0, "weather=0%"

# =====================================================
# API-Football wrappers
# =====================================================
def football_get(path: str, params: Dict[str, Any], timeout: int = 25) -> Any:
    url = f"https://v3.football.api-sports.io{path}"
    r = requests.get(url, headers={"x-apisports-key": FOOTBALL_API_KEY}, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()

def get_last_matches(team_id: int, last: int = 5) -> list:
    try:
        j = football_get("/fixtures", {"team": team_id, "last": last}, timeout=25)
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
# Injuries by TEAM (+cache)
# =====================================================
def _inj_cache_key(team_id: int, league_id: Optional[int]) -> str:
    return f"inj_team_{team_id}_{league_id or 'any'}_{SEASON}"

def _is_injury_active(item: Dict[str, Any]) -> bool:
    status = str(item.get("player", {}).get("status") or item.get("status") or "").lower()
    reason = str(item.get("player", {}).get("reason") or item.get("reason") or "").lower()
    if not status and not reason:
        return True
    bad = ["suspended", "suspension", "red card", "yellow card"]
    if any(b in status for b in bad) or any(b in reason for b in bad):
        return False
    active_keys = ["inj", "out", "doubt", "question", "strain", "hamstring", "ankle", "knee", "muscle", "fract", "ill"]
    return any(k in status for k in active_keys) or any(k in reason for k in active_keys) or ("out" in status)

def get_team_injuries_raw(team_id: int, league_id: Optional[int]) -> List[Dict[str, Any]]:
    key = _inj_cache_key(team_id, league_id)
    cached = STATE.get(key)
    cached_ts = float(STATE.get(key + "_ts", 0) or 0)
    if isinstance(cached, list) and (time.time() - cached_ts) < INJ_TTL:
        return cached

    params = {"team": team_id, "season": SEASON}
    if league_id:
        params["league"] = league_id

    try:
        j = football_get("/injuries", params, timeout=25)
        resp = j.get("response", []) or []
    except Exception as e:
        print(f"[WARN] injuries team fetch failed (primary): {e}")
        resp = []

    if not resp and league_id:
        try:
            j = football_get("/injuries", {"team": team_id, "season": SEASON}, timeout=25)
            resp = j.get("response", []) or []
        except Exception as e:
            print(f"[WARN] injuries team fetch failed (fallback): {e}")
            resp = []

    STATE[key] = resp
    STATE[key + "_ts"] = time.time()
    save_state(STATE)
    return resp

def count_team_injuries_att_def(team_id: int, league_id: Optional[int]) -> Tuple[Dict[str, int], int]:
    raw = get_team_injuries_raw(team_id, league_id)
    att = 0
    deff = 0

    for it in raw:
        if not _is_injury_active(it):
            continue
        player = it.get("player", {}) or {}
        ptype = str(player.get("type") or it.get("type") or "").lower()

        is_gk = "goalkeeper" in ptype or ptype in {"gk"}
        is_def = "defender" in ptype or ptype in {"df", "def"}
        is_mid = "midfielder" in ptype or ptype in {"mf", "mid"}
        is_fwd = "attacker" in ptype or "forward" in ptype or ptype in {"fw", "fwd", "st", "striker"}

        if is_gk or is_def:
            deff += 1
        elif is_mid or is_fwd:
            att += 1
        else:
            att += 1

    return {"att": att, "def": deff}, len(raw)

def get_injuries_map_for_match(home_id: int, away_id: int, league_id: Optional[int]) -> Dict[int, Dict[str, int]]:
    h_counts, _ = count_team_injuries_att_def(home_id, league_id)
    a_counts, _ = count_team_injuries_att_def(away_id, league_id)
    return {int(home_id): h_counts, int(away_id): a_counts}

def apply_injury_adjustments(lam_home: float, lam_away: float,
                            injuries: Dict[int, Dict[str, int]],
                            home_id: int, away_id: int) -> Tuple[float, float, str, str]:
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

# =====================================================
# Fatigue
# =====================================================
def get_fatigue_factor(team_id: int, match_ts_utc: int) -> Tuple[float, str, int]:
    try:
        dt_match = datetime.fromtimestamp(int(match_ts_utc), tz=timezone.utc)
        window_start = dt_match - timedelta(days=FATIGUE_WINDOW_DAYS)
        resp = get_last_matches(team_id, last=FATIGUE_LAST_N)

        played = 0
        for it in resp:
            fx = it.get("fixture", {}) or {}
            ts = fx.get("timestamp")
            if not ts:
                continue
            dt = datetime.fromtimestamp(int(ts), tz=timezone.utc)
            if not (window_start <= dt < dt_match):
                continue
            st = (fx.get("status", {}) or {}).get("short", "")
            if st in {"FT", "AET", "PEN"}:
                played += 1

        if played >= 3:
            k, why = 0.95, "fatigue=-5%"
        elif played == 2:
            k, why = 0.97, "fatigue=-3%"
        else:
            k, why = 1.00, "fatigue=0%"
        return k, why, played
    except:
        return 1.00, "fatigue=0%", 0

# =====================================================
# Standings + SOS + motivation (lightweight)
# =====================================================
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

def sos_fmt(k: float) -> str:
    pct = (k - 1.0) * 100.0
    if abs(pct) < 0.05:
        return "sos=0%"
    return f"sos={pct:+.1f}%"

def strength_of_schedule_factor(league_id: Optional[int], opp_ids: List[int]) -> Tuple[float, str]:
    if not league_id or not opp_ids:
        return 1.0, sos_fmt(1.0)
    smap = get_standings_map(int(league_id))
    if not smap:
        return 1.0, sos_fmt(1.0)

    pts = []
    for oid in opp_ids:
        row = smap.get(int(oid))
        if row and row.get("points") is not None:
            pts.append(int(row["points"]))
    if len(pts) < 3:
        return 1.0, sos_fmt(1.0)

    league_pts = [int(v.get("points") or 0) for v in smap.values() if v.get("points") is not None]
    if not league_pts:
        return 1.0, sos_fmt(1.0)

    avg_opp = sum(pts) / len(pts)
    avg_lg = sum(league_pts) / len(league_pts)
    if avg_lg <= 0:
        return 1.0, sos_fmt(1.0)

    diff = (avg_opp - avg_lg) / avg_lg
    diff = max(-0.15, min(0.15, diff))
    k = 1.0 + 0.04 * diff
    return k, sos_fmt(k)

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
# Poisson probabilities
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
    MAX_IDS = 5  # plan limitation
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
# Core: lambdas + factors
# =====================================================
def compute_lambdas_and_factors(f: Dict[str, Any]) -> Tuple[Optional[float], Optional[float], str]:
    fixture = f.get("fixture") or {}
    teams = f.get("teams") or {}
    home_t = teams.get("home") or {}
    away_t = teams.get("away") or {}
    league_obj = f.get("league") or {}

    home_id = home_t.get("id")
    away_id = away_t.get("id")
    ts = int(fixture.get("timestamp") or 0)
    league_name = league_obj.get("name", "?")
    league_id = league_obj.get("id")

    if not home_id or not away_id or not ts:
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

    inj_map = get_injuries_map_for_match(int(home_id), int(away_id), league_id)
    lam_home, lam_away, inj_h, inj_a = apply_injury_adjustments(lam_home, lam_away, inj_map, int(home_id), int(away_id))

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
        "• /signals — value-сигналы (P_final≥75% и Value>0)\n"
        "• /lines — линии тоталов без фильтра\n"
        "• /settle — рассчитать результаты PENDING\n"
        "• /roi — ROI/проходимость\n"
        "• /history — последние сигналы\n"
        "• /clv [days] — CLV по сигналам\n"
        "• /oddspapi_account — проверка OddsPapi\n"
        "• /reload_leagues — пересканировать лиги\n\n"
        f"Bookmaker: {ODDSPAPI_BOOKMAKER}\n"
        f"P_final = {W_MARKET:.2f}*P_market(vig-free) + {1-W_MARKET:.2f}*P_model\n"
        f"Пороги: P≥{int(MIN_PROB*100)}% и Value>{MIN_VALUE:+.2f}\n"
        f"Анти-аномалия: Value>{ANOMALY_VALUE_WARN:+.2f} → ⚠️ проверь линию вручную\n"
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

async def clv_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    days = 7
    if context.args:
        try:
            days = max(1, min(90, int(context.args[0])))
        except:
            days = 7

    rep = db_clv_report(days=days)
    msg = []
    msg.append(f"📉 CLV отчёт ({rep['days']} дней)\n\n")
    msg.append(f"• Сигналов: {rep['signals']}\n")
    msg.append(f"• Найдено closing odds: {rep['matched']}\n")
    if rep["matched"] == 0:
        msg.append("\nПока нет данных для CLV.\n")
        msg.append("Чтобы появился CLV:\n• 2–4 раза в день делай /lines (или /signals)\n• бот накопит odds_history\n")
        await update.message.reply_text("".join(msg))
        return

    msg.append(f"• Avg CLV: {rep['avg']*100:+.2f}%\n")
    msg.append(f"• Median CLV: {rep['median']*100:+.2f}%\n")
    msg.append(f"• Positive: {rep['pos']} | Negative: {rep['neg']}\n")

    if rep["samples"]:
        msg.append("\nПримеры:\n")
        for s in rep["samples"]:
            msg.append(
                f"• {s['match']} | {s['market']}\n"
                f"  taken={s['taken']:.2f} closing={s['closing']:.2f} → CLV={s['clv']*100:+.2f}%\n"
            )

    await update.message.reply_text("".join(msg))

# ---------------------------
# /lines — totals only (also saves odds_history)
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

    observed_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

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

        fixture_id = int(fixture.get("id") or 0) or None
        kickoff_ts_utc = ts

        ts_msk = int((datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=3)).timestamp())
        t_str = (datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=3)).strftime("%H:%M МСК")

        it = match_odds_for_fixture(used_date, ts_msk, home.get("name", ""), away.get("name", ""), odds_map, odds_index)
        if not it:
            continue

        o = parse_odds(it, markets_meta)
        if not any(v is not None for v in o.values()):
            continue

        # store odds snapshots
        for code, line, side in [
            ("O15", 1.5, "O"), ("U15", 1.5, "U"),
            ("O25", 2.5, "O"), ("U25", 2.5, "U"),
            ("O35", 3.5, "O"), ("U35", 3.5, "U"),
        ]:
            val = o.get(code)
            if val is None:
                continue
            db_insert_odds_snapshot({
                "observed_at": observed_at,
                "fixture_id": fixture_id,
                "kickoff_ts_utc": kickoff_ts_utc,
                "league": league.get("name"),
                "home": home.get("name"),
                "away": away.get("name"),
                "bookmaker": ODDSPAPI_BOOKMAKER,
                "market_code": code,
                "line": line,
                "side": side,
                "odds": float(val),
            })

        # vig-free implied
        po15, pu15, ov15 = vig_free_probs(o.get("O15"), o.get("U15"))
        po25, pu25, ov25 = vig_free_probs(o.get("O25"), o.get("U25"))
        po35, pu35, ov35 = vig_free_probs(o.get("O35"), o.get("U35"))

        sent += 1
        msg.append(
            f"🏆 {league.get('name','?')}\n"
            f"{home.get('name','?')} — {away.get('name','?')}\n"
            f"🕒 {t_str}\n"
            f"ТБ 1.5: {fmt_odds(o['O15'])} | ТМ 1.5: {fmt_odds(o['U15'])}"
            + (f" | Pm(vigfree)={int((po15 or 0)*100)}%/{int((pu15 or 0)*100)}% (OR={ov15:.3f})" if ov15 else "")
            + "\n"
            f"ТБ 2.5: {fmt_odds(o['O25'])} | ТМ 2.5: {fmt_odds(o['U25'])}"
            + (f" | Pm(vigfree)={int((po25 or 0)*100)}%/{int((pu25 or 0)*100)}% (OR={ov25:.3f})" if ov25 else "")
            + "\n"
            f"ТБ 3.5: {fmt_odds(o['O35'])} | ТМ 3.5: {fmt_odds(o['U35'])}"
            + (f" | Pm(vigfree)={int((po35 or 0)*100)}%/{int((pu35 or 0)*100)}% (OR={ov35:.3f})" if ov35 else "")
            + "\n\n"
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
# /signals — totals only (value on P_final + stores odds_history)
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

    observed_at = datetime.utcnow().replace(tzinfo=timezone.utc).isoformat()

    out = [f"⚽ ЦЕРБЕР — value-сигналы (тоталы) ({used_date})\nOdds: {ODDSPAPI_BOOKMAKER}\n"
           f"P_final={W_MARKET:.2f}*P_market(vigfree)+{1-W_MARKET:.2f}*P_model\n\n"]
    found_any = 0
    batch = 0

    def emit_if_value(fixture_id: int, kickoff_ts_utc: int, match_date_iso: str, league: str, home: str, away: str,
                      market_code: str, market_label: str, line: float, side: str,
                      p_model: float, p_market_side: Optional[float], book: Optional[float]) -> Optional[str]:
        if book is None:
            return None

        ok_odds, odds_warn = odds_sanity(market_code, float(book))
        pf = p_final(p_model, p_market_side)

        if pf < MIN_PROB:
            return None

        v = value_ev(pf, float(book))
        if v <= MIN_VALUE:
            return None

        fair = fair_odds(pf)
        value_warn = "⚠️ проверь линию вручную" if v > ANOMALY_VALUE_WARN else ""
        warns = " ".join([w for w in [value_warn, odds_warn] if w]).strip()
        warns = f" {warns}" if warns else ""

        # store signal if odds is sane
        if ok_odds:
            db_upsert_signal({
                "created_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
                "match_date": match_date_iso,
                "fixture_id": int(fixture_id),
                "kickoff_ts_utc": int(kickoff_ts_utc) if kickoff_ts_utc else None,
                "league": league,
                "home": home,
                "away": away,
                "market_code": market_code,
                "market_label": market_label,
                "line": float(line),
                "side": side,
                "p_model": float(p_model),
                "p_market": float(p_market_side) if p_market_side is not None else None,
                "p_final": float(pf),
                "book_odds": float(book),
                "fair_odds": fair,
                "value_ev": float(v),
                "stake": STAKE,
            })

        return (
            f"{market_label}: "
            f"P_final={int(pf*100)}% "
            f"(P_market={int((p_market_side or 0)*100)}%, P_model={int(p_model*100)}%) | "
            f"Book={float(book):.2f} | Fair={fair:.2f} | Value={v:+.2f}{warns}"
        )

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
        kickoff_ts_utc = ts

        ts_msk = int((datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=3)).timestamp())
        t_str = (datetime.fromtimestamp(ts, tz=timezone.utc) + timedelta(hours=3)).strftime("%H:%M МСК")

        home_name = home_t.get("name", "?")
        away_name = away_t.get("name", "?")
        league_name = league_obj.get("name", "?")
        match_date_iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()

        it = match_odds_for_fixture(used_date, ts_msk, home_name, away_name, odds_map, odds_index)
        if not it:
            continue

        odds = parse_odds(it, markets_meta)
        if not any(v is not None for v in odds.values()):
            continue

        # store odds snapshots
        for code, line, side in [
            ("O15", 1.5, "O"), ("U15", 1.5, "U"),
            ("O25", 2.5, "O"), ("U25", 2.5, "U"),
            ("O35", 3.5, "O"), ("U35", 3.5, "U"),
        ]:
            val = odds.get(code)
            if val is None:
                continue
            db_insert_odds_snapshot({
                "observed_at": observed_at,
                "fixture_id": int(fixture_id),
                "kickoff_ts_utc": int(kickoff_ts_utc),
                "league": league_name,
                "home": home_name,
                "away": away_name,
                "bookmaker": ODDSPAPI_BOOKMAKER,
                "market_code": code,
                "line": line,
                "side": side,
                "odds": float(val),
            })

        # vig-free market probs
        pO15, pU15, _ = vig_free_probs(odds.get("O15"), odds.get("U15"))
        pO25, pU25, _ = vig_free_probs(odds.get("O25"), odds.get("U25"))
        pO35, pU35, _ = vig_free_probs(odds.get("O35"), odds.get("U35"))

        lam_home, lam_away, factors_line = compute_lambdas_and_factors(f)
        if lam_home is None or lam_away is None:
            continue

        lam_total = lam_home + lam_away

        # Model probabilities (totals)
        p_model_O15 = prob_over_line_poisson(lam_total, 1.5)
        p_model_U15 = clamp(1.0 - p_model_O15, 0.05, 0.95)

        p_model_O25 = prob_over_line_poisson(lam_total, 2.5)
        p_model_U25 = clamp(1.0 - p_model_O25, 0.05, 0.95)

        p_model_O35 = prob_over_line_poisson(lam_total, 3.5)
        p_model_U35 = clamp(1.0 - p_model_O35, 0.05, 0.95)

        markets = [
            ("O15", "ТБ 1.5", 1.5, "O", p_model_O15, pO15, odds.get("O15")),
            ("U15", "ТМ 1.5", 1.5, "U", p_model_U15, pU15, odds.get("U15")),
            ("O25", "ТБ 2.5", 2.5, "O", p_model_O25, pO25, odds.get("O25")),
            ("U25", "ТМ 2.5", 2.5, "U", p_model_U25, pU25, odds.get("U25")),
            ("O35", "ТБ 3.5", 3.5, "O", p_model_O35, pO35, odds.get("O35")),
            ("U35", "ТМ 3.5", 3.5, "U", p_model_U35, pU35, odds.get("U35")),
        ]

        lines_out: List[str] = []
        for code, label, line, side, pm, pmark, book in markets:
            s = emit_if_value(
                int(fixture_id), int(kickoff_ts_utc), match_date_iso,
                league_name, home_name, away_name,
                code, label, line, side,
                pm, pmark, book
            )
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
            f"P_final ≥ {int(MIN_PROB*100)}% и Value > {MIN_VALUE:+.2f}\n\n"
            "Подсказка: /lines покажет коэффициенты без фильтра.\n"
            "CLV: делай /lines несколько раз в день → появится /clv."
        )
        return

    if out:
        await update.message.reply_text("".join(out))

# ---------------------------
# /settle
# ---------------------------
async def settle(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pending = db_fetch_pending(limit=300)
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
        pm = r.get("p_market")
        pm_s = "—" if pm is None else f"{int(float(pm)*100)}%"
        msg.append(
            f"{(r.get('league') or '?')} | {r.get('home') or '?'} — {r.get('away') or '?'}\n"
            f"{r.get('market_label')} | Book={float(r.get('book_odds') or 0):.2f} | "
            f"P_final={int(float(r.get('p_final') or 0)*100)}% (P_market={pm_s}, P_model={int(float(r.get('p_model') or 0)*100)}%) | "
            f"Value={float(r.get('value_ev') or 0):+.2f}\n"
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
    app.add_handler(CommandHandler("clv", clv_cmd))
    app.add_handler(CommandHandler("oddspapi_account", oddspapi_account))
    app.add_handler(CommandHandler("reload_leagues", reload_leagues))

    app.run_polling()

if __name__ == "__main__":
    main()
