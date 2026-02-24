from __future__ import annotations
from pathlib import Path
import pandas as pd
from ..config import RAW_DIR
from ..utils import ensure_dir

def ensure_atp_template() -> Path:
    out_dir = RAW_DIR / "atp"
    ensure_dir(out_dir)
    p = out_dir / "matches_template.csv"
    if not p.exists():
        df = pd.DataFrame([{
            "match_id": "example-1",
            "date": "2026-01-01",
            "tournament": "Example Open",
            "surface": "Hard",
            "player_a": "Player A",
            "player_b": "Player B",
            "winner": "Player A",  # set to blank for upcoming
        }])
        df.to_csv(p, index=False)
    return p
