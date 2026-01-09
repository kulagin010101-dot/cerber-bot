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
DEFAULT_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")  # optional

if not BOT_TOKEN:
    raise ValueError("BOT_TOKEN не задан!")
if not API_KEY:
    raise ValueError("FOOTBALL_API_KEY не задан!")

HEADERS = {"x-apisports-key": API_KEY}

SEASON = int(os.getenv("SEASON", "2025"))
MIN_PROB = float(os.getenv("MIN_PROB", "0.75"))
MIN_VALUE = float(os.getenv("MIN_VALUE", "0.00"))

OU_MAX_ODDS = float(os.getenv("OU_MAX_ODDS", "10.0"))
BTTS_MAX_ODDS = float(os.getenv("BTTS_MAX_ODDS", "15.0"))

STATE_FILE = "state.json"

# В API может быть "Pinnacle", "Pinnacle Sports" и т.п.
PINNACLE_MATCH = "pinnacle"

# ======================
# COMPETITIONS
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
# MODEL
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
# ODDS (Pinnacle)
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
        return data.get("response", []) or []
    except Exception as e:
        print(f"[ERROR] fetch_fixture_odds fixture={fixture_id}: {e}")
        return []

def is_ou_market(bet_name: str) -> bool:
    bn = norm(bet_name)
    keys = ["over/under", "over under", "total goals", "totals", "goals over/under", "match goals"]
    return any(k in bn for k in keys)

def is_btts_market(bet_name: str) -> bool:
    bn = norm(bet_name)
    keys = ["both teams to score", "btts", "both teams score"]
    return any(k in bn for k in keys)

def label_is_over25(label: str) -> bool:
    lb = norm(label).replace(",", ".")
    return ("over" in lb or lb.startswith("o")) and "2.5" in lb

def label_is_under25(label: str) -> bool:
    lb = norm(label).replace(",", ".")
    return ("under" in lb or lb.startswith("u")) and "2.5" in lb

def label_is_yes(label: str) -> bool:
    return norm(label) in ["yes", "y", "да"]

def label_is_no(label: str) -> bool:
    return norm(label) in ["no", "n", "нет"]

def extract_pinnacle_odds(odds_response: list) -> Tuple[Dict[str, Optional[float]], Optional[str], List[Dict[str, Any]]]:
    """
    returns (odds_dict, bookmaker_actual_name, pinnacle_bets_list)
    """
    out = {"O25": None, "U25": None, "BTTS_Y": None, "BTTS_N": None}
    if not odds_response:
        return out, None, []

    for item in odds_response:
        for bm in (item.get("bookmakers") or []):
            bm_name = norm(bm.get("name"))
            if PINNACLE_MATCH not in bm_name:
                continue

            actual_name = bm.get("name") or "Pinnacle"
            bets = bm.get("bets") or []

            for bet in bets:
                bet_name = bet.get("name") or ""
                values = bet.get("values") or []

                if is_ou_market(bet_name):
                    for v in values:
                        label = v.get("value")
                        try:
                            odd = float(v.get("odd"))
                        except:
                            continue
                        if odd <= 1.01 or odd > OU_MAX_ODDS:
                            continue
                        if label_is_over25(label):
                            out["O25"] = odd
                        elif label_is_under25(label):
                            out["U25"] = odd

                if is_btts_market(bet_name):
                    for v in values:
                        label = v.get("value")
                        try:
                            odd = float(v.get("odd"))
                        except:
                            continue
                        if odd <= 1.01 or odd > BTTS_MAX_ODDS:
                            continue
                        if label_is_yes(label):
                            out["BTTS_Y"] = odd
                        elif label_is_no(label):
                            out["BTTS_N"] = odd

            return out, actual_name, bets

    return out, None, []

# ======================
# LEAGUES (resolve + cache)
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
# HANDLERS
# ======================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🐺 ЦЕРБЕР — Pinnacle mode.\n\n"
        "Команды:\n"
        "• /signals — только value-сигналы\n"
        "• /all — все матчи с P и Pinnacle odds (без фильтров)\n"
        "• /check — диагностика\n"
        "• /reload_leagues — пересканировать турниры\n"
        "• /odds_debug — показать рынки Pinnacle по первому матчу дня\n\n"
        f"Пороги: P ≥ {int(MIN_PROB*100)}%, Value > {MIN_VALUE:+.2f}\n"
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

async def check(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ На дату {used_date} матчей нет.")
        return

    total = len(fixtures)
    sample = fixtures[:15]

    odds_present = 0
    pinn_present = 0
    pinn_with_any_market = 0

    for f in sample:
        fixture_id = (f.get("fixture") or {}).get("id")
        if not fixture_id:
            continue

        odds_resp = fetch_fixture_odds(int(fixture_id))
        if odds_resp:
            odds_present += 1

        odds_pin, pin_name, _ = extract_pinnacle_odds(odds_resp)
        if pin_name:
            pinn_present += 1
            if any(odds_pin[k] is not None for k in odds_pin.keys()):
                pinn_with_any_market += 1

    text = (
        f"🔎 CHECK ({used_date})\n"
        f"Матчей (после фильтра турниров): {total}\n\n"
        f"Проверено матчей: {len(sample)}\n"
        f"Odds ответ не пустой: {odds_present}/{len(sample)}\n"
        f"Pinnacle найден: {pinn_present}/{len(sample)}\n"
        f"Pinnacle и есть рынки (OU/BTTS): {pinn_with_any_market}/{len(sample)}\n"
    )
    await context.bot.send_message(chat_id=chat_id, text=text)

async def odds_debug(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)
    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ На дату {used_date} матчей нет.")
        return

    # берём первый матч
    f = fixtures[0]
    fixture = f.get("fixture") or {}
    fixture_id = fixture.get("id")
    home = (f.get("teams") or {}).get("home", {}) or {}
    away = (f.get("teams") or {}).get("away", {}) or {}
    league = f.get("league", {}) or {}
    league_name = league.get("name", "?")

    if not fixture_id:
        await context.bot.send_message(chat_id=chat_id, text="Не удалось взять fixture_id.")
        return

    odds_resp = fetch_fixture_odds(int(fixture_id))
    odds_pin, pin_name, bets = extract_pinnacle_odds(odds_resp)

    header = (
        f"🧪 ODDS DEBUG ({used_date})\n"
        f"🏆 {league_name}\n"
        f"{home.get('name','?')} — {away.get('name','?')}\n"
        f"fixture_id={fixture_id}\n\n"
    )

    if not pin_name:
        await context.bot.send_message(chat_id=chat_id, text=header + "❌ Pinnacle в odds не найден по этому матчу.")
        return

    lines = [header + f"✅ Найден букмекер: {pin_name}\n\n"]

    # список рынков
    lines.append("📌 Рынки (bet.name) у Pinnacle (первые 25):\n")
    for bet in bets[:25]:
        lines.append(f"• {bet.get('name','?')}\n")

    # показываем конкретные нужные значения
    lines.append("\n🎯 Найденные значения (что бот пытается парсить):\n")
    lines.append(f"Over 2.5: {odds_pin['O25']}\n")
    lines.append(f"Under 2.5: {odds_pin['U25']}\n")
    lines.append(f"BTTS Yes: {odds_pin['BTTS_Y']}\n")
    lines.append(f"BTTS No: {odds_pin['BTTS_N']}\n\n")

    # детальный вывод values для OU и BTTS рынков
    lines.append("🔍 Детально (OU/BTTS рынки и их values):\n")
    shown = 0
    for bet in bets:
        name = bet.get("name") or ""
        if not (is_ou_market(name) or is_btts_market(name)):
            continue
        lines.append(f"\n[{name}]\n")
        for v in (bet.get("values") or []):
            val = v.get("value")
            odd = v.get("odd")
            lines.append(f"  - {val}: {odd}\n")
        shown += 1
        if shown >= 6:
            break

    for part in chunked("".join(lines)):
        await context.bot.send_message(chat_id=chat_id, text=part)

async def all_matches(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = target_chat_id(update)

    fixtures, _, _, used_date = get_today_matches_filtered()
    if not fixtures:
        await context.bot.send_message(chat_id=chat_id, text=f"⚠️ На дату {used_date} матчей нет.")
        return

    out = [f"📋 ALL ({used_date}) — все матчи (без фильтра value)\nИсточник odds: Pinnacle\n\n"]
    sent = 0

    for f in fixtures[:20]:
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
        if not home_id or not away_id or not fixture_id:
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

        odds_resp = fetch_fixture_odds(int(fixture_id))
        odds_pin, pin_name, _ = extract_pinnacle_odds(odds_resp)
        pin_name = pin_name or "Pinnacle (not found)"

        def line(title: str, p: float, book: Optional[float]) -> str:
            if book is None:
                return f"{title}: P={int(p*100)}% | Book=—"
            v = value_ev(p, book)
            return f"{title}: P={int(p*100)}% | Book={float(book):.2f} | Value={v:+.2f}"

        out.append(
            f"🏆 {league_name}\n{home.get('name','?')} — {away.get('name','?')} | 🕒 {t_str}\n"
            f"Odds: {pin_name}\n"
            f"{line('ТБ2.5', p_o25, odds_pin['O25'])}\n"
            f"{line('ТМ2.5', p_u25, odds_pin['U25'])}\n"
            f"{line('ОЗ Да', p_btts_y, odds_pin['BTTS_Y'])}\n"
            f"{line('ОЗ Нет', p_btts_n, odds_pin['BTTS_N'])}\n\n"
        )

        sent += 1
        if sent % 5 == 0:
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

    out = [f"⚽ ЦЕРБЕР — value-сигналы ({used_date})\nИсточник odds: Pinnacle\n\n"]
    found_any = 0
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
        if not home_id or not away_id or not fixture_id:
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

        odds_resp = fetch_fixture_odds(int(fixture_id))
        odds_pin, pin_name, _ = extract_pinnacle_odds(odds_resp)
        if not pin_name:
            continue

        def market_line(title: str, p: float, book: Optional[float]) -> Optional[str]:
            if book is None:
                return None
            if p < MIN_PROB:
                return None
            v = value_ev(p, book)
            if v <= MIN_VALUE:
                return None
            return f"{title}: P={int(p*100)}% | Book={float(book):.2f} | Fair={fair_odds(p):.2f} | Value={v:+.2f}"

        lines = []
        for title, p, book in [
            ("ТБ 2.5", p_o25, odds_pin["O25"]),
            ("ТМ 2.5", p_u25, odds_pin["U25"]),
            ("ОЗ Да", p_btts_y, odds_pin["BTTS_Y"]),
            ("ОЗ Нет", p_btts_n, odds_pin["BTTS_N"]),
        ]:
            ml = market_line(title, p, book)
            if ml:
                lines.append(ml)

        if not lines:
            continue

        found_any += 1
        out.append(
            f"🏆 {league_name}\n"
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
                f"📭 На дату {used_date} нет value-сигналов по Pinnacle при порогах:\n"
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


