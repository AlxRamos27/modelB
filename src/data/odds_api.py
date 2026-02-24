from __future__ import annotations
import warnings
from datetime import date
from pathlib import Path
import pandas as pd
from .http import HttpClient
from ..config import RAW_DIR, env
from ..utils import ensure_dir

ODDS_API_BASE = "https://api.the-odds-api.com/v4"

SPORT_KEYS = {
    "nba": "basketball_nba",
    "nhl": "icehockey_nhl",
    "mlb": "baseball_mlb",
}


def fetch_live_odds(sport: str, today: date | None = None) -> Path | None:
    """Fetch live H2H + spreads + totals for a sport from The Odds API.

    Returns path to CSV with columns:
        date, home_team, away_team,
        odds_H, odds_A,           -- decimal moneyline (best available)
        spread_home, spread_away, -- point spread lines (home / away)
        total_over                -- Over/Under line (total points)
    Returns None if ODDS_API_KEY not set or sport not supported.
    """
    api_key = env("ODDS_API_KEY")
    if not api_key:
        return None
    sport_key = SPORT_KEYS.get(sport)
    if not sport_key:
        return None

    if today is None:
        today = date.today()

    out_dir = RAW_DIR / sport
    ensure_dir(out_dir)
    client = HttpClient(min_delay_s=0.5, timeout_s=30)

    rows: dict[tuple, dict] = {}

    for market in ("h2h", "spreads", "totals"):
        try:
            data = client.get_json(
                f"{ODDS_API_BASE}/sports/{sport_key}/odds/",
                params={
                    "apiKey": api_key,
                    "regions": "us",
                    "markets": market,
                    "oddsFormat": "decimal",
                },
            )
        except Exception as exc:
            warnings.warn(f"Odds API {market} fetch failed for {sport}: {exc}")
            continue

        for game in data:
            game_date = str(game.get("commence_time", ""))[:10]
            home = game.get("home_team", "")
            away = game.get("away_team", "")
            key = (game_date, home, away)
            if key not in rows:
                rows[key] = {"date": game_date, "home_team": home, "away_team": away,
                             "odds_H": None, "odds_A": None,
                             "spread_home": None, "spread_away": None,
                             "total_over": None}

            for bm in game.get("bookmakers", []):
                for mkt in bm.get("markets", []):
                    if mkt.get("key") != market:
                        continue
                    for outcome in mkt.get("outcomes", []):
                        price = outcome.get("price")
                        name = outcome.get("name", "")
                        point = outcome.get("point")
                        if market == "h2h":
                            if name == home:
                                if rows[key]["odds_H"] is None or price > rows[key]["odds_H"]:
                                    rows[key]["odds_H"] = price
                            elif name == away:
                                if rows[key]["odds_A"] is None or price > rows[key]["odds_A"]:
                                    rows[key]["odds_A"] = price
                        elif market == "spreads":
                            if name == home and point is not None:
                                rows[key]["spread_home"] = point
                            elif name == away and point is not None:
                                rows[key]["spread_away"] = point
                        elif market == "totals":
                            if name == "Over" and point is not None:
                                if rows[key]["total_over"] is None or point > rows[key]["total_over"]:
                                    rows[key]["total_over"] = point

    if not rows:
        warnings.warn(f"fetch_live_odds: No odds returned for {sport}.")
        return None

    df = pd.DataFrame(list(rows.values()))
    p = out_dir / f"odds_live_{today.isoformat()}.csv"
    df.to_csv(p, index=False)
    return p


# Keep old name for backward compatibility
def fetch_live_odds_nba(today: date | None = None) -> Path:
    result = fetch_live_odds("nba", today)
    if result is None:
        raise RuntimeError("ODDS_API_KEY is required. Add it to .env")
    return result
