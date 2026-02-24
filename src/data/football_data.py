from __future__ import annotations
from pathlib import Path
import pandas as pd
from .http import HttpClient
from ..config import RAW_DIR, env
from ..leagues import SOCCER_COMPETITION_CODES
from ..utils import ensure_dir

BASE = "https://api.football-data.org/v4"

def fetch_soccer_matches(league: str, season: int) -> Path:
    token = env("FOOTBALL_DATA_TOKEN")
    if not token:
        raise RuntimeError("FOOTBALL_DATA_TOKEN is required for soccer adapters (football-data.org). Put it in .env")
    comp = SOCCER_COMPETITION_CODES[league]
    out_dir = RAW_DIR / league
    ensure_dir(out_dir)

    client = HttpClient()
    url = f"{BASE}/competitions/{comp}/matches"
    data = client.get_json(url, headers={"X-Auth-Token": token}, params={"season": season})
    matches = data.get("matches", [])

    rows = []
    for m in matches:
        if m.get("status") not in ("FINISHED", "SCHEDULED", "TIMED", "IN_PLAY", "PAUSED"):
            continue
        score = m.get("score", {}).get("fullTime", {}) or {}
        rows.append({
            "match_id": m.get("id"),
            "utc_date": m.get("utcDate"),
            "status": m.get("status"),
            "home_team": m.get("homeTeam", {}).get("name"),
            "away_team": m.get("awayTeam", {}).get("name"),
            "home_goals": score.get("home"),
            "away_goals": score.get("away"),
            "season": season,
            "competition": comp,
        })

    df = pd.DataFrame(rows)
    p = out_dir / f"matches_{season}.csv"
    df.to_csv(p, index=False)
    return p
