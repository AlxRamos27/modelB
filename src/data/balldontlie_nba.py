from __future__ import annotations
from pathlib import Path
import pandas as pd
from .http import HttpClient
from ..config import RAW_DIR, env
from ..utils import ensure_dir

BASE = "https://api.balldontlie.io/v1"

def fetch_nba_games(season: int) -> Path:
    api_key = env("BALLDONTLIE_API_KEY")
    if not api_key:
        raise RuntimeError("BALLDONTLIE_API_KEY is required for NBA adapter. Put it in .env")

    out_dir = RAW_DIR / "nba"
    ensure_dir(out_dir)

    client = HttpClient(min_delay_s=0.35)
    headers = {"Authorization": api_key}
    page = 1
    per_page = 100
    rows = []

    while True:
        data = client.get_json(f"{BASE}/games", headers=headers, params={"seasons[]": season, "per_page": per_page, "page": page})
        for g in data.get("data", []):
            status = g.get("status", "")
            is_final = str(status).strip().lower() == "final"
            rows.append({
                "game_id": g.get("id"),
                "date": g.get("date"),
                "status": status,
                "season": g.get("season"),
                "home_team": (g.get("home_team") or {}).get("full_name"),
                "away_team": (g.get("visitor_team") or {}).get("full_name"),
                # Only store scores for finished games — BallDontLie returns 0
                # (not null) for unplayed games, which breaks is_finished detection.
                "home_score": g.get("home_team_score") if is_final else None,
                "away_score": g.get("visitor_team_score") if is_final else None,
            })
        meta = data.get("meta", {}) or {}
        if page >= (meta.get("total_pages") or 1):
            break
        page += 1

    df = pd.DataFrame(rows)
    p = out_dir / f"games_{season}.csv"
    df.to_csv(p, index=False)
    return p
