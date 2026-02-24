from __future__ import annotations
import warnings
from pathlib import Path
import pandas as pd
from ..config import RAW_DIR

_NEW_COLS = [
    "home_off_rtg", "home_def_rtg", "home_net_rtg", "home_pace",
    "away_off_rtg", "away_def_rtg", "away_net_rtg", "away_pace",
    "net_rtg_diff", "pace_avg",
]


def _load_stats(season: int) -> pd.DataFrame | None:
    p = RAW_DIR / "nba" / f"team_advanced_{season}.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as exc:
        warnings.warn(f"Could not read {p}: {exc}")
        return None


def add_nba_advanced_features(
    df: pd.DataFrame,
    home_col: str = "home_team",
    away_col: str = "away_team",
    season_col: str = "season",
) -> pd.DataFrame:
    """Merge prior-season advanced stats onto each game row (leakage-free).

    For games in season N, uses end-of-season stats from season N-1.
    If the prior-season file is missing, all new columns default to 0.0.

    New columns added:
        home_off_rtg, home_def_rtg, home_net_rtg, home_pace
        away_off_rtg, away_def_rtg, away_net_rtg, away_pace
        net_rtg_diff  (home_net_rtg - away_net_rtg)
        pace_avg      ((home_pace + away_pace) / 2)
    """
    df = df.copy()
    for c in _NEW_COLS:
        df[c] = 0.0

    if season_col not in df.columns:
        warnings.warn(
            f"add_nba_advanced_features: '{season_col}' column not found. "
            "All advanced stat features will be 0.0."
        )
        return df

    seasons = df[season_col].dropna().unique()

    for season in seasons:
        prior = int(season) - 1
        stats = _load_stats(prior)
        if stats is None:
            warnings.warn(
                f"add_nba_advanced_features: No file for prior season {prior} "
                f"(needed for season {int(season)}). "
                f"Fetch with: python -m src.cli fetch --league nba-advanced --season {prior}. "
                "Features will be 0.0 for this season."
            )
            continue

        stats_map: dict[str, dict] = stats.set_index("team_name").to_dict(orient="index")
        mask = df[season_col] == season

        def _get(team: str, key: str) -> float:
            entry = stats_map.get(team, {})
            return float(entry.get(key, 0.0)) if entry else 0.0

        rows = df.loc[mask]
        df.loc[mask, "home_off_rtg"] = [_get(t, "off_rating") for t in rows[home_col]]
        df.loc[mask, "home_def_rtg"] = [_get(t, "def_rating") for t in rows[home_col]]
        df.loc[mask, "home_net_rtg"] = [_get(t, "net_rating") for t in rows[home_col]]
        df.loc[mask, "home_pace"]    = [_get(t, "pace")       for t in rows[home_col]]
        df.loc[mask, "away_off_rtg"] = [_get(t, "off_rating") for t in rows[away_col]]
        df.loc[mask, "away_def_rtg"] = [_get(t, "def_rating") for t in rows[away_col]]
        df.loc[mask, "away_net_rtg"] = [_get(t, "net_rating") for t in rows[away_col]]
        df.loc[mask, "away_pace"]    = [_get(t, "pace")       for t in rows[away_col]]

        df.loc[mask, "net_rtg_diff"] = (
            df.loc[mask, "home_net_rtg"] - df.loc[mask, "away_net_rtg"]
        )
        df.loc[mask, "pace_avg"] = (
            (df.loc[mask, "home_pace"] + df.loc[mask, "away_pace"]) / 2.0
        )

    return df
