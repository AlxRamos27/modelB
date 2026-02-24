from __future__ import annotations
import warnings
from pathlib import Path
import pandas as pd
from ..config import RAW_DIR

# Mapping from 538 ELO team abbreviations to BallDontLie full names.
# Includes historical abbreviations for relocated/renamed franchises.
NBA_ABBREV_TO_FULL: dict[str, str] = {
    "ATL": "Atlanta Hawks",
    "BOS": "Boston Celtics",
    "BKN": "Brooklyn Nets",
    "NJN": "Brooklyn Nets",
    "CHA": "Charlotte Hornets",
    "CHH": "Charlotte Hornets",
    "CHI": "Chicago Bulls",
    "CLE": "Cleveland Cavaliers",
    "DAL": "Dallas Mavericks",
    "DEN": "Denver Nuggets",
    "DET": "Detroit Pistons",
    "GSW": "Golden State Warriors",
    "HOU": "Houston Rockets",
    "IND": "Indiana Pacers",
    "LAC": "LA Clippers",
    "LAL": "Los Angeles Lakers",
    "MEM": "Memphis Grizzlies",
    "VAN": "Memphis Grizzlies",
    "MIA": "Miami Heat",
    "MIL": "Milwaukee Bucks",
    "MIN": "Minnesota Timberwolves",
    "NOP": "New Orleans Pelicans",
    "NOH": "New Orleans Pelicans",
    "NOK": "New Orleans Pelicans",
    "NYK": "New York Knicks",
    "OKC": "Oklahoma City Thunder",
    "SEA": "Oklahoma City Thunder",
    "ORL": "Orlando Magic",
    "PHI": "Philadelphia 76ers",
    "PHX": "Phoenix Suns",
    "POR": "Portland Trail Blazers",
    "SAC": "Sacramento Kings",
    "SAS": "San Antonio Spurs",
    "TOR": "Toronto Raptors",
    "UTA": "Utah Jazz",
    "WAS": "Washington Wizards",
    "WSB": "Washington Wizards",
}


def _load_elo_538() -> pd.DataFrame | None:
    p = RAW_DIR / "nba" / "elo_538.csv"
    if not p.exists():
        return None
    try:
        return pd.read_csv(p)
    except Exception as exc:
        warnings.warn(f"Could not read 538 ELO file: {exc}")
        return None


def add_elo_538_features(
    df: pd.DataFrame,
    home_col: str = "home_team",
    away_col: str = "away_team",
    date_col: str = "date",
) -> pd.DataFrame:
    """Merge 538 pre-game ELO differential onto each game row.

    Feature added: elo_538_diff_pre = elo1_pre - elo2_pre
    (team1 = home team for non-neutral games in the 538 dataset)

    Matching uses plain date objects to avoid timezone issues.
    Unmatched rows default to 0.0.
    """
    df = df.copy()
    df["elo_538_diff_pre"] = 0.0

    elo_raw = _load_elo_538()
    if elo_raw is None:
        warnings.warn(
            "add_elo_538_features: elo_538.csv not found. "
            "Run: python -m src.cli fetch --league nba-elo. "
            "elo_538_diff_pre will be 0.0 for all rows."
        )
        return df

    # Map abbreviations to full names.
    elo_raw["home_full"] = elo_raw["team1"].map(NBA_ABBREV_TO_FULL)
    elo_raw["away_full"] = elo_raw["team2"].map(NBA_ABBREV_TO_FULL)

    unmapped_abbrevs = (
        set(elo_raw.loc[elo_raw["home_full"].isna(), "team1"].dropna().unique())
        | set(elo_raw.loc[elo_raw["away_full"].isna(), "team2"].dropna().unique())
    )
    if unmapped_abbrevs:
        warnings.warn(
            f"add_elo_538_features: Unmapped abbreviations: {unmapped_abbrevs}. "
            "Add them to NBA_ABBREV_TO_FULL in src/features/elo_538.py."
        )
    elo_raw = elo_raw.dropna(subset=["home_full", "away_full"])

    # Compute ELO diff and normalize date to plain date object for join.
    elo_raw["elo_diff"] = (
        pd.to_numeric(elo_raw["elo1_pre"], errors="coerce")
        - pd.to_numeric(elo_raw["elo2_pre"], errors="coerce")
    )
    elo_raw["join_date"] = pd.to_datetime(elo_raw["date"], errors="coerce").dt.date

    # Build lookup dict: (date, home_full, away_full) -> elo_diff
    lookup: dict = (
        elo_raw.dropna(subset=["elo_diff", "join_date"])
        .set_index(["join_date", "home_full", "away_full"])["elo_diff"]
        .to_dict()
    )

    # Normalize game dates to plain date (strips timezone).
    game_dates = pd.to_datetime(df[date_col], utc=True, errors="coerce").dt.date

    df["elo_538_diff_pre"] = [
        float(lookup.get((gdate, home, away), 0.0))
        for gdate, home, away in zip(game_dates, df[home_col], df[away_col])
    ]
    return df
