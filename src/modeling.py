from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss
import joblib

from .config import MODELS_DIR
from .utils import ensure_dir
from .pipeline import Dataset

_BASE_FEATURE_COLS = [
    "elo_diff_pre",
    "home_winrate_l5", "away_winrate_l5",
    "diff_l5_gap",
    "inj_gap",
]

_NBA_EXTRA_FEATURE_COLS = [
    "net_rtg_diff",
    "pace_avg",
    "elo_538_diff_pre",
]

# Backward-compatible alias (used by load_model and any external code).
FEATURE_COLS = _BASE_FEATURE_COLS


def get_feature_cols(kind: str) -> list[str]:
    """Return feature columns for the given dataset kind.

    NBA gets base + advanced stats + 538 ELO features.
    All other sports use only the base set.
    """
    if kind == "nba":
        return _BASE_FEATURE_COLS + _NBA_EXTRA_FEATURE_COLS
    return list(_BASE_FEATURE_COLS)

@dataclass
class TrainResult:
    model_path: Path
    metrics: dict

def train(dataset: Dataset) -> TrainResult:
    df = dataset.df.copy()
    train_df = df[df["is_finished"] & df[dataset.target_col].notna()].copy()
    if len(train_df) < 100:
        raise RuntimeError(f"Not enough finished games to train ({len(train_df)}). Fetch more seasons or leagues.")

    feat_cols = get_feature_cols(dataset.kind)
    X = train_df[feat_cols].fillna(0.0).to_numpy()
    y = train_df[dataset.target_col].astype(str).to_numpy()

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    clf = LogisticRegression(
        multi_class="multinomial" if len(dataset.classes) == 3 else "auto",
        max_iter=2000,
        n_jobs=1,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", clf),
    ])

    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_val)
    pred = pipe.predict(X_val)

    acc = float(accuracy_score(y_val, pred))
    ll = float(log_loss(y_val, proba, labels=pipe.named_steps["clf"].classes_))

    # Brier score for the "home win" class in 2-way; for 3-way, average one-vs-rest.
    classes = list(pipe.named_steps["clf"].classes_)
    if len(classes) == 2:
        idx_h = classes.index("H")
        brier = float(brier_score_loss((y_val == "H").astype(int), proba[:, idx_h]))
    else:
        briers = []
        for c in classes:
            i = classes.index(c)
            briers.append(brier_score_loss((y_val == c).astype(int), proba[:, i]))
        brier = float(np.mean(briers))

    out_dir = MODELS_DIR / dataset.league
    ensure_dir(out_dir)
    model_path = out_dir / "model.joblib"
    meta_path = out_dir / "metrics.json"

    joblib.dump({"model": pipe, "classes": classes, "feature_cols": feat_cols}, model_path)
    metrics = {"accuracy": acc, "logloss": ll, "brier": brier, "n_train": int(len(X_train)), "n_val": int(len(X_val)), "classes": classes}
    meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return TrainResult(model_path=model_path, metrics=metrics)

def load_model(league: str):
    p = MODELS_DIR / league / "model.joblib"
    obj = joblib.load(p)
    return obj["model"], obj["classes"], obj["feature_cols"]

def predict(dataset: Dataset, top_n: int = 3, odds_csv: Path | None = None) -> pd.DataFrame:
    df = dataset.df.copy()
    upcoming = df[~df["is_finished"]].copy()
    if upcoming.empty:
        # some APIs may mark future as not finished but missing scores; keep those too
        upcoming = df[df[dataset.target_col].isna()].copy()

    if upcoming.empty:
        raise RuntimeError("No upcoming games found in dataset. Fetch current season and ensure scheduled games exist.")

    model, classes, feat_cols = load_model(dataset.league)
    X = upcoming[feat_cols].fillna(0.0).to_numpy()
    proba = model.predict_proba(X)

    out = upcoming[["date","home_team","away_team"]].copy()
    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, i]

    # Confidence band (very rough): distance from 0.5 for the top class
    out["p_max"] = out[[f"p_{c}" for c in classes]].max(axis=1)
    out["top_pick"] = out[[f"p_{c}" for c in classes]].idxmax(axis=1).str.replace("p_", "", regex=False)

    # Optional odds comparison
    if odds_csv and odds_csv.exists():
        odds = pd.read_csv(odds_csv)
        # Normalize both date columns to plain YYYY-MM-DD strings before merging
        # to avoid UTC-aware datetime vs naive string mismatch.
        out_for_merge = out.copy()
        out_for_merge["date"] = pd.to_datetime(
            out_for_merge["date"], utc=True, errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        odds = odds.copy()
        odds["date"] = pd.to_datetime(
            odds["date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        # Expected columns: date, home_team, away_team, odds_H, odds_D(optional), odds_A
        m = out_for_merge.merge(odds, on=["date", "home_team", "away_team"], how="left")
        # implied probabilities (decimal odds)
        for c in classes:
            col = f"odds_{c}"
            if col in m.columns:
                m[f"imp_{c}"] = 1.0 / pd.to_numeric(m[col], errors="coerce")
                m[f"edge_{c}"] = m[f"p_{c}"] - m[f"imp_{c}"]
        out = m

    out = out.sort_values("p_max", ascending=False).head(top_n).reset_index(drop=True)
    return out
