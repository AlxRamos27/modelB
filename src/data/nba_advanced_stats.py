from __future__ import annotations
import warnings
from pathlib import Path
import pandas as pd
from ..config import RAW_DIR
from ..utils import ensure_dir


def _season_str(season: int) -> str:
    """Convert end-year integer to nba_api format: 2025 -> '2024-25'."""
    return f"{season - 1}-{str(season)[-2:]}"


def fetch_nba_advanced_stats(season: int) -> Path:
    """Fetch LeagueDashTeamStats (Advanced, PerGame) for a single season.

    Args:
        season: Integer end-year (e.g. 2025 for the 2024-25 season).

    Returns:
        Path to data/raw/nba/team_advanced_{season}.csv

    Raises:
        ImportError: if nba_api is not installed.
        RuntimeError: if the API call fails.
    """
    try:
        from nba_api.stats.endpoints import LeagueDashTeamStats
    except ImportError as exc:
        raise ImportError("nba_api is required: pip install nba_api") from exc

    out_dir = RAW_DIR / "nba"
    ensure_dir(out_dir)

    season_str = _season_str(season)

    try:
        endpoint = LeagueDashTeamStats(
            season=season_str,
            measure_type_detailed_defense="Advanced",
            per_mode_simple="PerGame",
            timeout=60,
        )
        df = endpoint.get_data_frames()[0]
    except Exception as exc:
        raise RuntimeError(
            f"nba_api LeagueDashTeamStats failed for {season_str}: {exc}"
        ) from exc

    keep = {
        "TEAM_NAME": "team_name",
        "OFF_RATING": "off_rating",
        "DEF_RATING": "def_rating",
        "NET_RATING": "net_rating",
        "PACE": "pace",
    }
    missing = [c for c in keep if c not in df.columns]
    if missing:
        warnings.warn(f"nba_api response missing expected columns: {missing}")

    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    df["season"] = season

    p = out_dir / f"team_advanced_{season}.csv"
    df.to_csv(p, index=False)
    return p
