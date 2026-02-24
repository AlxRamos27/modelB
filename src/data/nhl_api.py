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
    start_year = int(str(season_id)[:4])
    end_date = f"{start_year+1}-06-30"

    # The NHL schedule API returns one week at a time.
    # We iterate using nextStartDate until we reach the end of the season.
    current_date = f"{start_year}-10-01"
    rows = []
    seen_weeks = set()

    while current_date <= end_date:
        try:
            data = client.get_json(f"{BASE}/schedule/{current_date}")
        except Exception:
            break

        weeks = data.get("gameWeek", []) or []
        for w in weeks:
            week_date = w.get("date", "")
            if week_date in seen_weeks:
                continue
            seen_weeks.add(week_date)
            for g in w.get("games", []) or []:
                home = (g.get("homeTeam") or {})
                away = (g.get("awayTeam") or {})
                state = str(g.get("gameState", "")).upper()
                is_final = state in ("OFF", "FINAL", "OVER")
                rows.append({
                    "game_id": g.get("id"),
                    "date": g.get("startTimeUTC"),
                    "status": state,
                    "season": season_id,
                    "home_team": home.get("name", {}).get("default"),
                    "away_team": away.get("name", {}).get("default"),
                    # NHL API returns score=0 for unplayed games — only store when final
                    "home_score": home.get("score") if is_final else None,
                    "away_score": away.get("score") if is_final else None,
                })

        next_date = data.get("nextStartDate", "")
        if not next_date or next_date >= end_date or next_date == current_date:
            break
        current_date = next_date

    df = pd.DataFrame(rows)
    p = out_dir / f"schedule_{season_id}.csv"
    df.to_csv(p, index=False)
    return p
