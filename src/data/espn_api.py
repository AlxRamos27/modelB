"""
espn_api.py
===========
Fetches public ESPN data (standings, injuries) with no API key required.
Supports NBA, NHL, and NFL. Data is used as supplementary context.
"""
from __future__ import annotations
from pathlib import Path
import warnings
import pandas as pd
from .http import HttpClient
from ..config import RAW_DIR
from ..utils import ensure_dir

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports"

SPORT_MAP = {
    "nba": ("basketball", "nba"),
    "nhl": ("hockey", "nhl"),
    "mlb": ("baseball", "mlb"),
}


def fetch_espn_standings(sport: str) -> Path | None:
    """Fetch current standings from ESPN public API. Returns path to CSV or None on error."""
    if sport not in SPORT_MAP:
        return None
    category, league = SPORT_MAP[sport]
    out_dir = RAW_DIR / sport
    ensure_dir(out_dir)

    client = HttpClient(min_delay_s=0.3)
    try:
        data = client.get_json(f"{ESPN_BASE}/{category}/{league}/standings")
    except Exception as exc:
        warnings.warn(f"ESPN standings fetch failed for {sport}: {exc}")
        return None

    rows = []
    for group in data.get("children", []) or []:
        for entry in group.get("standings", {}).get("entries", []) or []:
            team_name = (entry.get("team") or {}).get("displayName", "")
            stats = {s["name"]: s.get("value") for s in entry.get("stats", []) or []}
            rows.append({
                "team": team_name,
                "wins": stats.get("wins"),
                "losses": stats.get("losses"),
                "win_pct": stats.get("winPercent"),
                "points_for": stats.get("pointsFor"),
                "points_against": stats.get("pointsAgainst"),
                "streak": stats.get("streak"),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    p = out_dir / "espn_standings.csv"
    df.to_csv(p, index=False)
    return p


def fetch_espn_injuries(sport: str) -> Path | None:
    """Fetch current injury reports from ESPN public API."""
    if sport not in SPORT_MAP:
        return None
    category, league = SPORT_MAP[sport]
    out_dir = RAW_DIR / sport
    ensure_dir(out_dir)

    client = HttpClient(min_delay_s=0.3)
    try:
        data = client.get_json(f"{ESPN_BASE}/{category}/{league}/injuries")
    except Exception as exc:
        warnings.warn(f"ESPN injuries fetch failed for {sport}: {exc}")
        return None

    rows = []
    for item in data.get("injuries", []) or []:
        team = (item.get("team") or {}).get("displayName", "")
        for inj in item.get("injuries", []) or []:
            athlete = (inj.get("athlete") or {})
            rows.append({
                "team": team,
                "player": athlete.get("displayName", ""),
                "position": (athlete.get("position") or {}).get("abbreviation", ""),
                "status": inj.get("status", ""),
                "date": inj.get("date", ""),
                "type": inj.get("type", ""),
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    p = out_dir / "espn_injuries.csv"
    df.to_csv(p, index=False)
    return p


def fetch_espn_scoreboard_games(sport: str) -> list[dict]:
    """Return today's unplayed games from ESPN public scoreboard (no auth required).

    Returns list of dicts: [{home_team, away_team, date}]
    Only includes games that have NOT started yet (status.type.completed == False
    and status.type.name != STATUS_FINAL / STATUS_IN_PROGRESS).
    """
    if sport not in SPORT_MAP:
        return []
    category, league = SPORT_MAP[sport]
    client = HttpClient(min_delay_s=0.3)
    try:
        data = client.get_json(f"{ESPN_BASE}/{category}/{league}/scoreboard")
    except Exception as exc:
        warnings.warn(f"ESPN scoreboard fetch failed for {sport}: {exc}")
        return []

    games = []
    for event in data.get("events", []) or []:
        comps = event.get("competitions", []) or []
        if not comps:
            continue
        comp = comps[0]
        status_type = (comp.get("status") or {}).get("type") or {}
        # Skip games already completed or in progress
        if status_type.get("completed", False):
            continue
        sname = str(status_type.get("name", "")).upper()
        if "PROGRESS" in sname or "FINAL" in sname:
            continue

        home_team = None
        away_team = None
        for c in comp.get("competitors", []) or []:
            name = ((c.get("team") or {}).get("displayName") or "").strip()
            if c.get("homeAway") == "home":
                home_team = name
            else:
                away_team = name

        if home_team and away_team:
            games.append({
                "home_team": home_team,
                "away_team": away_team,
                "date": event.get("date", ""),
            })
    return games


def get_espn_context(sport: str) -> dict:
    """
    Returns a summary dict with standings and injury context for Claude analysis.
    Does not affect model features — used only for narrative enrichment.
    """
    context: dict = {}

    standings_path = fetch_espn_standings(sport)
    if standings_path and standings_path.exists():
        try:
            df = pd.read_csv(standings_path)
            # all_teams: full record lookup for Claude notes
            context["all_teams"] = df[["team", "wins", "losses", "win_pct"]].to_dict(orient="records")
            # top_teams kept for backward compat
            context["top_teams"] = df.nlargest(5, "win_pct")[["team", "wins", "losses", "win_pct"]].to_dict(orient="records")
        except Exception:
            pass

    injuries_path = fetch_espn_injuries(sport)
    if injuries_path and injuries_path.exists():
        try:
            df = pd.read_csv(injuries_path)
            out_players = df[df["status"].str.lower().isin(["out", "doubtful", "questionable"])]
            context["key_injuries"] = out_players[["team", "player", "status"]].head(10).to_dict(orient="records")
        except Exception:
            pass

    return context
