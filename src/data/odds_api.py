from __future__ import annotations
import warnings
from datetime import date
from pathlib import Path
import pandas as pd
from .http import HttpClient
from ..config import RAW_DIR, env
from ..utils import ensure_dir

ODDS_API_BASE = "https://api.the-odds-api.com/v4"


def fetch_live_odds_nba(today: date | None = None) -> Path:
    """Fetch live NBA H2H odds from The Odds API (free tier: 500 req/month).

    Calls /sports/basketball_nba/odds with regions=us, markets=h2h,
    oddsFormat=decimal. Takes the best (maximum) decimal odds across all
    bookmakers for each side.

    Requires ODDS_API_KEY in .env.

    Args:
        today: Date label for the output filename. Defaults to date.today().

    Returns:
        Path to data/raw/nba/odds_live_{YYYY-MM-DD}.csv
        Columns: date, home_team, away_team, odds_H, odds_A

    Raises:
        RuntimeError: if ODDS_API_KEY is not set or the request fails.
    """
    api_key = env("ODDS_API_KEY")
    if not api_key:
        raise RuntimeError("ODDS_API_KEY is required. Add it to .env")

    if today is None:
        today = date.today()

    out_dir = RAW_DIR / "nba"
    ensure_dir(out_dir)

    client = HttpClient(min_delay_s=0.5, timeout_s=30)
    data = client.get_json(
        f"{ODDS_API_BASE}/sports/basketball_nba/odds/",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "decimal",
        },
    )

    rows = []
    for game in data:
        game_date = str(game.get("commence_time", ""))[:10]  # "YYYY-MM-DD"
        home_team = game.get("home_team", "")
        away_team = game.get("away_team", "")

        best_home: float | None = None
        best_away: float | None = None

        for bm in game.get("bookmakers", []):
            for market in bm.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    price = outcome.get("price")
                    name = outcome.get("name", "")
                    if name == home_team:
                        if best_home is None or price > best_home:
                            best_home = price
                    elif name == away_team:
                        if best_away is None or price > best_away:
                            best_away = price

        if home_team and away_team:
            rows.append({
                "date": game_date,
                "home_team": home_team,
                "away_team": away_team,
                "odds_H": best_home,
                "odds_A": best_away,
            })

    if not rows:
        warnings.warn("fetch_live_odds_nba: No odds returned from The Odds API.")

    df = pd.DataFrame(rows, columns=["date", "home_team", "away_team", "odds_H", "odds_A"])
    p = out_dir / f"odds_live_{today.isoformat()}.csv"
    df.to_csv(p, index=False)
    return p
