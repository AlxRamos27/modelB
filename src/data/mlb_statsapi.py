from __future__ import annotations
from pathlib import Path
import pandas as pd
from .http import HttpClient
from ..config import RAW_DIR
from ..utils import ensure_dir

BASE = "https://statsapi.mlb.com/api/v1"

def fetch_mlb_schedule(season: int) -> Path:
    out_dir = RAW_DIR / "mlb"
    ensure_dir(out_dir)

    client = HttpClient(min_delay_s=0.25)
    # schedule endpoint is public
    data = client.get_json(f"{BASE}/schedule", params={"sportId": 1, "season": season, "gameType": "R"})
    dates = data.get("dates", []) or []

    rows = []
    for d in dates:
        for g in d.get("games", []) or []:
            teams = g.get("teams", {}) or {}
            home = (teams.get("home") or {}).get("team", {}) or {}
            away = (teams.get("away") or {}).get("team", {}) or {}
            rows.append({
                "game_id": g.get("gamePk"),
                "date": g.get("gameDate"),
                "status": (g.get("status") or {}).get("detailedState"),
                "season": season,
                "home_team": home.get("name"),
                "away_team": away.get("name"),
                "home_score": (teams.get("home") or {}).get("score"),
                "away_score": (teams.get("away") or {}).get("score"),
            })

    df = pd.DataFrame(rows)
    p = out_dir / f"schedule_{season}.csv"
    df.to_csv(p, index=False)
    return p
