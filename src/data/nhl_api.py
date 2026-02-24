from __future__ import annotations
from pathlib import Path
import pandas as pd
from .http import HttpClient
from ..config import RAW_DIR
from ..utils import ensure_dir

BASE = "https://api-web.nhle.com/v1"

def fetch_nhl_schedule(season: int) -> Path:
    # NHL uses season format like 20252026; we accept either 2025 (and expand) or full
    if season < 10000:
        season_id = int(f"{season}{season+1}")
    else:
        season_id = season

    out_dir = RAW_DIR / "nhl"
    ensure_dir(out_dir)

    client = HttpClient(min_delay_s=0.25)
    # Schedule is available through the gamecenter endpoints; simplest: use "schedule" for a date range.
    # We'll pull October 1 .. June 30 for the season start year.
    start_year = int(str(season_id)[:4])
    start = f"{start_year}-10-01"
    end = f"{start_year+1}-06-30"

    data = client.get_json(f"{BASE}/schedule/{start}", params={"endDate": end})
    weeks = data.get("gameWeek", []) or []
    rows = []
    for w in weeks:
        for g in w.get("games", []) or []:
            home = (g.get("homeTeam") or {})
            away = (g.get("awayTeam") or {})
            rows.append({
                "game_id": g.get("id"),
                "date": g.get("startTimeUTC"),
                "status": g.get("gameState"),
                "season": season_id,
                "home_team": home.get("name", {}).get("default"),
                "away_team": away.get("name", {}).get("default"),
                "home_score": home.get("score"),
                "away_score": away.get("score"),
            })

    df = pd.DataFrame(rows)
    p = out_dir / f"schedule_{season_id}.csv"
    df.to_csv(p, index=False)
    return p
