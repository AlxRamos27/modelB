from __future__ import annotations
from pathlib import Path
import requests
from ..config import RAW_DIR
from ..utils import ensure_dir

ELO_538_URL = (
    "https://raw.githubusercontent.com/Neil-Paine-1/NBA-elo/master/NBA_ELO.csv"
)


def fetch_elo_538() -> Path:
    """Download the Neil Paine / 538 NBA ELO CSV from GitHub.

    No API key required. The file covers all historical NBA seasons.

    Returns:
        Path to data/raw/nba/elo_538.csv
    """
    out_dir = RAW_DIR / "nba"
    ensure_dir(out_dir)

    resp = requests.get(ELO_538_URL, timeout=60)
    resp.raise_for_status()

    p = out_dir / "elo_538.csv"
    p.write_bytes(resp.content)
    return p
