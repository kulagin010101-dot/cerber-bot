import os
import json
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
WEATHER_KEY = os.getenv("WEATHER_API_KEY")

if not BOT_TOKEN or not FOOTBALL_API_KEY or not ODDSPAPI_KEY:
    raise RuntimeError("❌ Не заданы переменные окружения")

HEADERS = {"x-apisports-key": FOOTBALL_API_KEY}

SEASON = 2025
MIN_PROB = 0.75
MIN_VALUE = 0.0

OU_MIN, OU_MAX = 1.10, 6.00
BTTS_MIN, BTTS_MAX = 1.10, 6.00

STATE_FILE = "state.json"

# =====================================================
# TARGET LEAGUES
# =====================================================
TARGET_LEAGUES = [
    ("England", "Premier League"),
    ("England", "FA Cup"),
    ("Germany", "Bundesliga"),
    ("Spain", "La Liga"),
    ("Italy", "Serie A"),
    ("France", "Ligue 1"),
    ("International", "UEFA Champions League"),
    ("International", "UEFA Europa League"),
]

# =====================================================
# STATE
# =====================================================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(s):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

STATE = load_state()

# =====================================================
# HELPERS
# =====================================================
def norm(s): return str(s or "").lower().strip()

def clamp(x, a=0.05, b=0.95): return max(a, min(b, x))

def safe_team(name):
    for ch in ["fc", ".", ",", "&"]:
        name = name.replace(ch, "")
    return norm(name)

# =====================================================
# WEATHER
# =====================================================
def weather_factor(city):
    if not WEATHER_KEY or not city:
        return 1.0
    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": WEATHER_KEY, "units": "metric"},
            timeout=10
        ).json()
        f = 1.0
        if r["main"]["temp"] < 5 or r["main"]["temp"] > 28: f *= 0.95
        if r.get("rain"): f *= 0.95
        if r["wind"]["speed"] > 8: f *= 0.96
        return f
    except:
        return 1.0

# =====================================================
# MODEL
# =====================================================
def get_last_matches(team_id):
    r = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=HEADERS,
        params={"team": team_id, "last": 5, "season": SEASON},
        timeout=20
    ).json()
    return r.get("response", [])

def analyze(matches):
    g, o25, btts, n = 0, 0, 0, 0
    for m in matches:
        h, a = m["goals"]["home"], m["goals"]["away"]
        if h is None or a is None: continue
        g += h + a
        if h + a > 2: o25 += 1
        if h > 0 and a > 0: btts += 1
        n += 1
    if n == 0: return None
    return {
        "avg": g / n,
        "o25": o25 / n,
        "btts": btts / n
    }

def prob_over25(h, a, w):
    p = 0.60
    if (h["avg"] + a["avg"]) / 2 > 2.6: p += 0.08
    if h["o25"] > 0.6: p += 0.05
    if a["o25"] > 0.6: p += 0.05
    if h["btts"] > 0.6 and a["btts"] > 0.6: p += 0.04
    return clamp(p * w)

def prob_btts(h, a, w):
    p = 0.50
    p += 0.25 * (h["btts"] - 0.5)
    p += 0.25 * (a["btts"] - 0.5)
    return clamp(p * w)

def fair(p): return round(1 / p, 2)
def value(p, o): return p * o - 1

# =====================================================
# LEAGUES RESOLVE
# =====================================================
def resolve_leagues():
    if "league_ids" in STATE:
        return STATE["league_ids"]

    league_ids = {}
    r = requests.get(
        "https://v3.football.api-sports.io/leagues",
        headers=HEADERS,
        timeout=30
    ).json()["response"]

    for c, n in TARGET_LEAGUES:
        for l in r:
            if norm(l["league"]["name"]) == norm(n):
                league_ids[n] = l["league"]["id"]

    STATE["league_ids"] = league_ids
    save_state(STATE)
    return league_ids

# =====================================================
# FIXTURES
# =====================================================
def get_today_fixtures():
    leagues = resolve_leagues()
    today = (datetime.utcnow() + timedelta(hours=3)).strftime("%Y-%m-%d")

    r = requests.get(
        "https://v3.football.api-sports.io/fixtures",
        headers=HEADERS,
        params={"date": today},
        timeout=30
    ).json()["response"]

    return [f for f in r if f["league"]["id"] in leagues.values()], today

# =====================================================
# ODDS (Oddspapi / 1xBet)
# =====================================================
def oddspapi(path, params):
    params["apiKey"] = ODDSPAPI_KEY
    r = requests.get("https://api.oddspapi.io" + path, params=params, timeout=30)
    if r.status_code != 200:
        raise RuntimeError(r.text)
    return r.json()

def get_odds_map(date):
    tournaments = oddspapi("/v4/tournaments", {"sportId": 10})
    ids = [t["tournamentId"] for t in tournaments][:5]  # limit 5

    odds = oddspapi("/v4/odds-by-tournaments", {
        "tournamentIds": ",".join(map(str, ids)),
        "bookmaker": ODDSPAPI_BOOKMAKER,
        "oddsFormat": "decimal"
    })

    m = {}
    for o in odds:
        dt = datetime.fromisoformat(o["startTime"].replace("Z", "+00:00")) + timedelta(hours=3)
        if dt.strftime("%Y-%m-%d") != date: continue
        k1 = safe_team(o["participant1Name"])
        k2 = safe_team(o["participant2Name"])
        m[(k1, k2)] = o
        m[(k2, k1)] = o
    return m

def parse_odds(o):
    res = {"O25": None, "U25": None, "BTTS_Y": None, "BTTS_N": None}
    bm = o["bookmakerOdds"].get(ODDSPAPI_BOOKMAKER, {})
    for m in bm.get("markets", {}).values():
        for out in m.get("outcomes", {}).values():
            p = out["players"]["0"]
            oid = norm(p["bookmakerOutcomeId"])
            price = float(p["price"])
            if "2.5/over" in oid and OU_MIN <= price <= OU_MAX: res["O25"] = price
            if "2.5/under" in oid and OU_MIN <= price <= OU_MAX: res["U25"] = price
            if oid in ("btts/yes", "yes") and BTTS_MIN <= price <= BTTS_MAX: res["BTTS_Y"] = price
            if oid in ("btts/no", "no") and BTTS_MIN <= price <= BTTS_MAX: res["BTTS_N"] = price
    return res

# =====================================================
# TELEGRAM COMMANDS
# =====================================================
async def start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР активирован\n\n"
        "Команды:\n"
        "/signals — value-сигналы\n"
        "/all — все матчи\n"
    )

async def signals(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    fixtures, date = get_today_fixtures()
    odds_map = get_odds_map(date)
    out = f"⚽ ЦЕРБЕР — value ({date})\n\n"
    found = False

    for f in fixtures:
        h, a = f["teams"]["home"], f["teams"]["away"]
        o = odds_map.get((safe_team(h["name"]), safe_team(a["name"])))
        if not o: continue
        odds = parse_odds(o)

        hs, as_ = analyze(get_last_matches(h["id"])), analyze(get_last_matches(a["id"]))
        if not hs or not as_: continue

        w = weather_factor(f["fixture"]["venue"]["city"])
        p_o = prob_over25(hs, as_, w)
        p_b = prob_btts(hs, as_, w)

        lines = []
        if odds["O25"] and p_o >= MIN_PROB and value(p_o, odds["O25"]) > MIN_VALUE:
            lines.append(f"ТБ2.5 P={int(p_o*100)}% | {odds['O25']:.2f}")
        if odds["BTTS_Y"] and p_b >= MIN_PROB and value(p_b, odds["BTTS_Y"]) > MIN_VALUE:
            lines.append(f"ОЗ Да P={int(p_b*100)}% | {odds['BTTS_Y']:.2f}")

        if lines:
            found = True
            out += f"{h['name']} — {a['name']}\n" + "\n".join(lines) + "\n\n"

    if not found:
        out += "📭 Сегодня нет value-сигналов"

    await update.message.reply_text(out)

async def all_matches(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    fixtures, date = get_today_fixtures()
    await update.message.reply_text(f"📋 Матчей сегодня: {len(fixtures)}")

# =====================================================
# RUN
# =====================================================
def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("signals", signals))
    app.add_handler(CommandHandler("all", all_matches))
    app.run_polling()

if __name__ == "__main__":
    main()
