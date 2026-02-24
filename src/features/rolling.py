from __future__ import annotations
import pandas as pd
import numpy as np

def add_rolling_form(
    df: pd.DataFrame,
    home_col: str,
    away_col: str,
    home_score_col: str,
    away_score_col: str,
    window: int = 5,
) -> pd.DataFrame:
    """Adds rolling winrate and point differential for home/away teams.

    Computes using only past games (no leakage) by processing in chronological order.
    """
    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    stats = {}  # team -> dict of lists for recent outcomes

    def init_team(t):
        if t not in stats:
            stats[t] = {"wins": [], "diff": []}

    home_wr, away_wr = [], []
    home_diff, away_diff = [], []

    for _, r in df.iterrows():
        h, a = r[home_col], r[away_col]
        init_team(h); init_team(a)

        hw = stats[h]["wins"][-window:]
        aw = stats[a]["wins"][-window:]
        hd = stats[h]["diff"][-window:]
        ad = stats[a]["diff"][-window:]

        home_wr.append(float(np.mean(hw)) if hw else 0.5)
        away_wr.append(float(np.mean(aw)) if aw else 0.5)
        home_diff.append(float(np.mean(hd)) if hd else 0.0)
        away_diff.append(float(np.mean(ad)) if ad else 0.0)

        hs, as_ = r[home_score_col], r[away_score_col]
        if pd.isna(hs) or pd.isna(as_):
            continue
        if hs > as_:
            stats[h]["wins"].append(1.0); stats[a]["wins"].append(0.0)
        elif hs < as_:
            stats[h]["wins"].append(0.0); stats[a]["wins"].append(1.0)
        else:
            stats[h]["wins"].append(0.5); stats[a]["wins"].append(0.5)
        stats[h]["diff"].append(float(hs - as_))
        stats[a]["diff"].append(float(as_ - hs))

    df["home_winrate_l5"] = home_wr
    df["away_winrate_l5"] = away_wr
    df["home_diff_l5"] = home_diff
    df["away_diff_l5"] = away_diff
    df["diff_l5_gap"] = df["home_diff_l5"] - df["away_diff_l5"]
    return df
