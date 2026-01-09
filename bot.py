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

# ======================
# TARGET COMPETITIONS + ALIASES
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
# GOALS MODEL (простая, но реальная)
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

def clamp(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    return min(max(x, lo), hi)

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

# ======================
# ODDS (STRICT): только правильные рынки
# ======================
def norm(s: Any) -> str:
    return str(s or "").strip().lower()

# для OU/BTTS отсечём мусорные “космические” коэффициенты
OU_MAX_ODDS = float(os.getenv("OU_MAX_ODDS", "10.0"))      # для ТБ/ТМ 2.5
BTTS_MAX_ODDS = float(os.getenv("BTTS_MAX_ODDS", "15.0"))  # для ОЗ

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
        return data.get("response", []) or []
    except Exception as e:
        print(f"[ERROR] fetch_fixture_odds fixture={fixture_id}: {e}")
        return []

def is_ou_market(bet_name: str) -> bool:
    """
    Признаки рынка тоталов.
    """
    bn = norm(bet_name)
    keys = [
        "over/under",
        "over under",
        "total goals",
        "totals",
        "goals over/under",
        "goals - over/under",
        "match goals",
    ]
    return any(k in bn for k in keys)

def is_btts_market(bet_name: str) -> bool:
    bn = norm(bet_name)
    keys = [
        "both teams to score",
        "btts",
        "both teams score",
    ]
    return any(k in bn for k in keys)

def label_is_over25(label: str) -> bool:
    lb = norm(label).replace(",", ".")
    return ("over" in lb or lb.startswith("o")) and "2.5" in lb

def label_is_under25(label: str) -> bool:
    lb = norm(label).replace(",", ".")
    return ("under" in lb or lb.startswith("u")) and "2.5" in lb

def label_is_yes(label: str) -> bool:
    lb = norm(label)
    return lb in ["yes", "y", "да"]

def label_is_no(label: str) -> bool:
    lb = norm(label)
    return lb in ["no", "n", "нет"]

def extract_best_odds_strict(odds_response: list) -> Dict[str, Optional[float]]:
    """
    Возвращает лучший коэф среди всех букмекеров, но ТОЛЬКО в правильных рынках:
    - O25 / U25 из OU рынков
    - BTTS_Y / BTTS_N из BTTS рынков
    """
    best = {"O25": None, "U25": None, "BTTS_Y": None, "BTTS_N": None}
    if not odds_response:
        return best

    for item in odds_response:
        for bm in (item.get("bookmakers") or []):
            for bet in (bm.get("bets") or []):
                bet_name = bet.get("name") or ""
                bn = norm(bet_name)
                values = bet.get("values") or []

                # ---- Over/Under 2.5 ----
                if is_ou_market(bn):
                    for v in values:
                        label = v.get("value")
                        odd_raw = v.get("odd")
                        try:
                            odd = float(odd_raw)
                        except:
                            continue

                        # фильтр “мусора”
                        if odd <= 1.01 or odd > OU_MAX_ODDS:
                            continue

                        if label_is_over25(label):
                            if best["O25"] is None or odd > best["O25"]:
                                best["O25"] = odd
                        elif label_is_under25(label):
                            if best["U25"] is None or odd > best["U25"]:
                                best["U25"] = odd

                # ---- BTTS ----
                if is_btts_market(bn):
                    for v in values:
                        label = v.get("value")
                        odd_raw = v.get("odd")
                        try:
                            odd = float(odd_raw)
                        except:
                            continue

                        if odd <= 1.01 or odd > BTTS_MAX_ODDS:
                            continue

                        if label_is_yes(label):
                            if best["BTTS_Y"] is None or odd > best["BTTS_Y"]:
                                best["BTTS_Y"] = odd
                        elif label_is_no(label):
                            if best["BTTS_N"] is None or odd > best["BTTS_N"]:
                                best["BTTS_N"] = odd

    return best

def fair_odds(p: float) -> float:
    return round(1.0 / max(p, 1e-9), 2)

def value_ev(p: float, book_odds: float) -> float:
    return (p * book_odds) - 1.0

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
        "🐺 ЦЕРБЕР — odds исправлены (строгий парсинг рынков).\n\n"
        "Команды:\n"
        "• /signals — матчи + P + best odds + fair + value\n"
        "• /reload_leagues — пересканировать турниры\n\n"
        f"Фильтр “космических” odds: OU<= {OU_MAX_ODDS}, BTTS<= {BTTS_MAX_ODDS}\n"
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

    out = [f"⚽ ЦЕРБЕР — матчи ({used_date})\nРынки: ТБ2.5 / ТМ2.5 / ОЗ(Да/Нет)\nOdds: лучший коэф среди букмекеров (строго по рынкам)\n\n"]
    count = 0

    for f in fixtures:
        fixture = f.get("fixture") or {}
        fixture_id = fixture.get("id")

        home = (f.get("teams") or {}).get("home", {}) or {}
        away = (f.get("teams") or {}).get("away", {}) or {}
        league = f.get("league", {}) or {}
        league_name = league.get("name", "?")

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

        odds_best = {"O25": None, "U25": None, "BTTS_Y": None, "BTTS_N": None}
        if fixture_id:
            odds_resp = fetch_fixture_odds(int(fixture_id))
            odds_best = extract_best_odds_strict(odds_resp)

        def fmt_market(title: str, p: float, book: Optional[float]) -> str:
            fo = fair_odds(p)
            if book:
                v = value_ev(p, book)
                return f"{title}: P={int(p*100)}% | Book={book:.2f} | Fair={fo:.2f} | Value={v:+.2f}"
            return f"{title}: P={int(p*100)}% | Book=нет линии | Fair={fo:.2f}"

        out.append(
            f"🏆 {league_name}\n"
            f"{home.get('name','?')} — {away.get('name','?')}\n"
            f"🕒 {t_str}\n"
            f"{fmt_market('ТБ 2.5', p_o25, odds_best['O25'])}\n"
            f"{fmt_market('ТМ 2.5', p_u25, odds_best['U25'])}\n"
            f"{fmt_market('ОЗ Да', p_btts_y, odds_best['BTTS_Y'])}\n"
            f"{fmt_market('ОЗ Нет', p_btts_n, odds_best['BTTS_N'])}\n\n"
        )

        count += 1
        if count % 6 == 0:
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
    app.run_polling()

if __name__ == "__main__":
    main()
