from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import numpy as np

from .config import PROCESSED_DIR, MANUAL_DIR
from .utils import ensure_dir
from .features.elo import add_elo_features
from .features.rolling import add_rolling_form
from .features.injuries import load_injuries, add_injury_impact
from .features.nba_advanced import add_nba_advanced_features
from .features.elo_538 import add_elo_538_features

_NBA_EXTRA_COLS = [
    "home_off_rtg", "home_def_rtg", "home_net_rtg", "home_pace",
    "away_off_rtg", "away_def_rtg", "away_net_rtg", "away_pace",
    "net_rtg_diff", "pace_avg", "elo_538_diff_pre",
]

@dataclass
class Dataset:
    league: str
    kind: str
    df: pd.DataFrame
    target_col: str
    classes: list[str]  # e.g. ["H","D","A"] or ["H","A"]

def _parse_date(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    # fallback for some APIs
    df["date"] = df[col]
    return df

def build_dataset_soccer(raw_csv: Path, league: str) -> Dataset:
    df = pd.read_csv(raw_csv)
    df = _parse_date(df, "utc_date")
    # keep finished for training, scheduled for prediction
    df["is_finished"] = df["status"].eq("FINISHED")

    # target
    def outcome(r):
        if pd.isna(r["home_goals"]) or pd.isna(r["away_goals"]):
            return None
        if r["home_goals"] > r["away_goals"]:
            return "H"
        if r["home_goals"] < r["away_goals"]:
            return "A"
        return "D"
    df["y"] = df.apply(outcome, axis=1)

    # features
    df = df.sort_values("date").reset_index(drop=True)
    df = add_elo_features(df, "home_team", "away_team", "home_goals", "away_goals")
    df = add_rolling_form(df, "home_team", "away_team", "home_goals", "away_goals", window=5)

    injuries = load_injuries(MANUAL_DIR / "injuries.csv")
    df = add_injury_impact(df, injuries, league=league, date_col="date", home_col="home_team", away_col="away_team")

    # save processed
    out_dir = PROCESSED_DIR / league
    ensure_dir(out_dir)
    out_csv = out_dir / "games.csv"
    df.to_csv(out_csv, index=False)

    return Dataset(league=league, kind="soccer", df=df, target_col="y", classes=["H","D","A"])

def build_dataset_two_way(raw_csv: Path, league: str, kind: str, date_col: str, home_score: str, away_score: str,
                          home_team: str = "home_team", away_team: str = "away_team") -> Dataset:
    df = pd.read_csv(raw_csv)
    df = _parse_date(df, date_col)
    df["is_finished"] = df[home_score].notna() & df[away_score].notna()

    def outcome(r):
        if pd.isna(r[home_score]) or pd.isna(r[away_score]):
            return None
        return "H" if r[home_score] > r[away_score] else "A"
    df["y"] = df.apply(outcome, axis=1)

    df = df.sort_values("date").reset_index(drop=True)
    df = add_elo_features(df, home_team, away_team, home_score, away_score)
    df = add_rolling_form(df, home_team, away_team, home_score, away_score, window=5)

    injuries = load_injuries(MANUAL_DIR / "injuries.csv")
    df = add_injury_impact(df, injuries, league=league, date_col="date", home_col=home_team, away_col=away_team)

    if kind == "nba":
        df = add_nba_advanced_features(df, home_col=home_team, away_col=away_team, season_col="season")
        df = add_elo_538_features(df, home_col=home_team, away_col=away_team, date_col="date")
    else:
        for c in _NBA_EXTRA_COLS:
            df[c] = 0.0

    out_dir = PROCESSED_DIR / league
    ensure_dir(out_dir)
    out_csv = out_dir / "games.csv"
    df.to_csv(out_csv, index=False)

    return Dataset(league=league, kind=kind, df=df, target_col="y", classes=["H","A"])
