from __future__ import annotations
import pandas as pd
from pathlib import Path

def load_injuries(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date","league","team","player","status","impact"])
    df = pd.read_csv(path)
    if "impact" not in df.columns:
        df["impact"] = 0.0
    df["impact"] = pd.to_numeric(df["impact"], errors="coerce").fillna(0.0)
    return df

def add_injury_impact(df_games: pd.DataFrame, injuries: pd.DataFrame, league: str, date_col: str, home_col: str, away_col: str) -> pd.DataFrame:
    """Aggregate injury impact per team for the game date (<= date)."""
    if injuries.empty:
        df_games = df_games.copy()
        df_games["inj_home_impact"] = 0.0
        df_games["inj_away_impact"] = 0.0
        return df_games

    g = df_games.copy()
    g[date_col] = pd.to_datetime(g[date_col], utc=True, errors="coerce")
    inj = injuries.copy()
    inj = inj[inj["league"].astype(str).str.lower() == league.lower()]
    inj["date"] = pd.to_datetime(inj["date"], utc=True, errors="coerce")

    # For each game row, sum impacts of injuries with inj.date <= game.date
    # (simple approximation; you can refine to expected return dates later)
    def team_impact(team: str, game_date) -> float:
        m = inj[(inj["team"] == team) & (inj["date"] <= game_date)]
        return float(m["impact"].sum()) if len(m) else 0.0

    g["inj_home_impact"] = [team_impact(t, d) if pd.notna(d) else 0.0 for t, d in zip(g[home_col], g[date_col])]
    g["inj_away_impact"] = [team_impact(t, d) if pd.notna(d) else 0.0 for t, d in zip(g[away_col], g[date_col])]
    g["inj_gap"] = g["inj_home_impact"] - g["inj_away_impact"]
    return g
