from __future__ import annotations
import pandas as pd
import numpy as np

def add_elo_features(
    df: pd.DataFrame,
    home_col: str,
    away_col: str,
    home_score_col: str | None,
    away_score_col: str | None,
    k: float = 20.0,
    home_adv: float = 60.0,
    base: float = 1500.0,
) -> pd.DataFrame:
    """Compute pre-game Elo for each team.

    Notes:
    - For soccer and other sports, we treat "win/loss" only for Elo updates.
    - Draws are treated as 0.5/0.5 when both scores exist and equal.
    """
    df = df.copy()
    teams = pd.unique(pd.concat([df[home_col], df[away_col]], ignore_index=True).dropna())
    elo = {t: base for t in teams}

    pre_home = []
    pre_away = []

    for _, r in df.iterrows():
        h = r[home_col]
        a = r[away_col]
        eh = elo.get(h, base)
        ea = elo.get(a, base)
        pre_home.append(eh)
        pre_away.append(ea)

        if home_score_col and away_score_col:
            hs = r[home_score_col]
            as_ = r[away_score_col]
            if pd.isna(hs) or pd.isna(as_):
                continue
            if hs > as_:
                sh, sa = 1.0, 0.0
            elif hs < as_:
                sh, sa = 0.0, 1.0
            else:
                sh, sa = 0.5, 0.5

            # expected with home advantage
            exp_h = 1.0 / (1.0 + 10 ** (((ea) - (eh + home_adv)) / 400.0))
            exp_a = 1.0 - exp_h
            elo[h] = eh + k * (sh - exp_h)
            elo[a] = ea + k * (sa - exp_a)

    df["elo_home_pre"] = np.array(pre_home, dtype=float)
    df["elo_away_pre"] = np.array(pre_away, dtype=float)
    df["elo_diff_pre"] = df["elo_home_pre"] - df["elo_away_pre"]
    return df
