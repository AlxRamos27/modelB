"""
generate_predictions.py
=======================
Daily orchestration script: fetch → train → predict for each sport.
Writes results to docs/data/predictions.json for the static web frontend.

Usage:
    python scripts/generate_predictions.py

Environment variables (from .env or GitHub Secrets):
    BALLDONTLIE_API_KEY  – required for NBA
    FOOTBALL_DATA_TOKEN  – required for soccer (epl, laliga, ligue1, bundesliga, primeira, ucl)
    ODDS_API_KEY         – optional, used for live odds if available
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from datetime import date, datetime, timezone
from itertools import combinations
from math import prod
from pathlib import Path

# ── Ensure project root is in sys.path ──────────────────────────────────────
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # all relative paths in src/ resolve from here

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Import pipeline components ───────────────────────────────────────────────
from src.config import env
from src.data.football_data import fetch_soccer_matches
from src.data.balldontlie_nba import fetch_nba_games
from src.data.mlb_statsapi import fetch_mlb_schedule
from src.data.nhl_api import fetch_nhl_schedule
from src.pipeline import build_dataset_soccer, build_dataset_two_way
from src.modeling import train as train_model, predict as predict_model

# ── Constants ────────────────────────────────────────────────────────────────
TARGETS = [3, 4, 5, 6, 10, 20]
SOCCER_LEAGUES = ["epl", "laliga", "ligue1", "bundesliga", "primeira", "ucl"]
OUT_PATH = ROOT / "docs" / "data" / "predictions.json"


# ── Season helpers ───────────────────────────────────────────────────────────

def current_nba_season() -> int:
    """Returns BallDontLie season integer (end-year convention).
    NBA 2025-26 → 2025  (season starts Oct, ends June)
    """
    today = date.today()
    # If we're before October, we're in the previous season
    return today.year if today.month >= 10 else today.year - 1


def current_soccer_season() -> int:
    """Soccer seasons start Aug/Sep, so Aug 2025 → season 2025."""
    today = date.today()
    return today.year if today.month >= 8 else today.year - 1


def current_nhl_season() -> int:
    """NHL season starts Oct, ends June. Same convention as NBA."""
    today = date.today()
    return today.year if today.month >= 10 else today.year - 1


def current_mlb_season() -> int:
    """MLB season is April–October of the same year."""
    return date.today().year


# ── Per-sport runner ─────────────────────────────────────────────────────────

def run_nba() -> dict:
    api_key = env("BALLDONTLIE_API_KEY")
    if not api_key:
        return _skip("BALLDONTLIE_API_KEY no configurada")
    season = current_nba_season()
    try:
        raw = fetch_nba_games(season)
        ds = build_dataset_two_way(
            raw, league="nba", kind="nba",
            date_col="date", home_score="home_score", away_score="away_score",
        )
        metrics = train_model(ds).metrics
        picks_df = predict_model(ds, top_n=30)
        return _ok(picks_df, metrics, "nba")
    except Exception as exc:
        return _error(exc)


def run_nhl() -> dict:
    season = current_nhl_season()
    try:
        raw = fetch_nhl_schedule(season)
        ds = build_dataset_two_way(
            raw, league="nhl", kind="nhl",
            date_col="date", home_score="home_score", away_score="away_score",
        )
        metrics = train_model(ds).metrics
        picks_df = predict_model(ds, top_n=30)
        return _ok(picks_df, metrics, "nhl")
    except Exception as exc:
        return _error(exc)


def run_mlb() -> dict:
    today = date.today()
    # MLB is April–October
    if not (4 <= today.month <= 10):
        return _skip("Fuera de temporada (Oct–Mar)")
    season = current_mlb_season()
    try:
        raw = fetch_mlb_schedule(season)
        ds = build_dataset_two_way(
            raw, league="mlb", kind="mlb",
            date_col="date", home_score="home_score", away_score="away_score",
        )
        metrics = train_model(ds).metrics
        picks_df = predict_model(ds, top_n=30)
        return _ok(picks_df, metrics, "mlb")
    except Exception as exc:
        return _error(exc)


def run_soccer(league: str) -> dict:
    api_key = env("FOOTBALL_DATA_TOKEN")
    if not api_key:
        return _skip("FOOTBALL_DATA_TOKEN no configurada")
    season = current_soccer_season()
    try:
        raw = fetch_soccer_matches(league, season)
        ds = build_dataset_soccer(raw, league=league)
        metrics = train_model(ds).metrics
        picks_df = predict_model(ds, top_n=30)
        return _ok(picks_df, metrics, league)
    except Exception as exc:
        return _error(exc)


# ── Result builders ──────────────────────────────────────────────────────────

def _ok(picks_df, metrics: dict, sport: str) -> dict:
    today_str = date.today().isoformat()
    picks = []
    for _, row in picks_df.iterrows():
        # Determine the winning side and probability
        if "p_H" in row and "p_A" in row:
            if row.get("p_H", 0) >= row.get("p_A", 0):
                pick, pick_label, p_win = "H", row["home_team"], float(row["p_H"])
            else:
                pick, pick_label, p_win = "A", row["away_team"], float(row["p_A"])
        else:
            # Soccer 3-way: use top_pick
            pick = str(row.get("top_pick", "H"))
            p_win = float(row.get("p_max", 0.5))
            if pick == "H":
                pick_label = row["home_team"]
            elif pick == "A":
                pick_label = row["away_team"]
            else:
                pick_label = "Empate"

        implied_odds = round(1.0 / max(p_win, 0.01), 2)
        signal = "alta" if p_win >= 0.70 else "media" if p_win >= 0.60 else "baja"

        game_date = ""
        try:
            import pandas as pd
            dt = pd.to_datetime(row["date"], utc=True)
            game_date = dt.strftime("%Y-%m-%d")
        except Exception:
            game_date = today_str

        picks.append({
            "home_team":   str(row["home_team"]),
            "away_team":   str(row["away_team"]),
            "pick":        pick,
            "pick_label":  str(pick_label),
            "p_win":       round(p_win, 4),
            "implied_odds": implied_odds,
            "signal":      signal,
            "date":        game_date,
        })

    # Sort by p_win descending
    picks.sort(key=lambda x: x["p_win"], reverse=True)

    return {
        "status":  "ok",
        "metrics": {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)},
        "picks":   picks,
    }


def _skip(reason: str) -> dict:
    return {"status": "skipped", "reason": reason, "picks": []}


def _error(exc: Exception) -> dict:
    warnings.warn(str(exc))
    return {"status": "error", "reason": str(exc), "picks": []}


# ── Parlay builder ───────────────────────────────────────────────────────────

def build_parlay(picks: list[dict], target: float) -> dict | None:
    """Find the combination of picks whose product of implied_odds is closest to target."""
    if len(picks) < 2:
        return None

    # Use only top 12 picks (highest confidence) to keep runtime fast
    cands = picks[:12]

    best: dict | None = None
    best_diff = float("inf")

    max_legs = min(len(cands), 9)
    for n in range(2, max_legs + 1):
        for combo in combinations(cands, n):
            total_odds = prod(p["implied_odds"] for p in combo)
            diff = abs(total_odds - target)
            combined_prob = prod(p["p_win"] for p in combo)
            if diff < best_diff:
                best = {
                    "legs":         [_leg(p) for p in combo],
                    "total_odds":   round(total_odds, 2),
                    "combined_prob": round(combined_prob, 4),
                    "n_legs":       n,
                }
                best_diff = diff

    return best


def _leg(pick: dict) -> dict:
    return {
        "sport":        pick.get("_sport", ""),
        "match":        f"{pick['home_team']} vs {pick['away_team']}",
        "pick":         pick["pick_label"],
        "p_win":        pick["p_win"],
        "implied_odds": pick["implied_odds"],
    }


def annotate_sport(picks: list[dict], sport: str) -> list[dict]:
    return [{**p, "_sport": sport} for p in picks]


def build_all_parlays(sports_results: dict) -> dict:
    """Build parlays for each sport and combined."""
    # Per-sport parlays
    by_sport: dict = {}

    for sport in ["nba", "nhl", "mlb"]:
        res = sports_results.get(sport, {})
        picks = annotate_sport(res.get("picks", []), sport)
        if picks:
            by_sport[sport] = {f"cuota_{t}": build_parlay(picks, t) for t in TARGETS}

    # Soccer: aggregate all leagues
    soccer_picks: list[dict] = []
    for lg in SOCCER_LEAGUES:
        res = sports_results.get(lg, {})
        soccer_picks += annotate_sport(res.get("picks", []), lg)
    soccer_picks.sort(key=lambda x: x["p_win"], reverse=True)
    if soccer_picks:
        by_sport["soccer"] = {f"cuota_{t}": build_parlay(soccer_picks, t) for t in TARGETS}

    # Combined parlays: all sports together
    all_picks: list[dict] = []
    for sport in ["nba", "nhl", "mlb"]:
        all_picks += annotate_sport(sports_results.get(sport, {}).get("picks", []), sport)
    all_picks += soccer_picks
    all_picks.sort(key=lambda x: x["p_win"], reverse=True)

    combined = {f"cuota_{t}": build_parlay(all_picks, t) for t in TARGETS}

    return {"by_sport": by_sport, "combined": combined}


# ── Main ─────────────────────────────────────────────────────────────────────

SPORT_NAMES = {
    "nba":        "NBA",
    "nhl":        "NHL",
    "mlb":        "MLB",
    "epl":        "Premier League",
    "laliga":     "La Liga",
    "ligue1":     "Ligue 1",
    "bundesliga": "Bundesliga",
    "primeira":   "Primeira Liga",
    "ucl":        "Champions League",
}


def main() -> None:
    print("=" * 60)
    print(f"PronosticoSport — {date.today().isoformat()}")
    print("=" * 60)

    results: dict = {}

    # ── NBA ──
    print("\n[NBA] Procesando…")
    results["nba"] = run_nba()
    print(f"  → {results['nba']['status']} | {len(results['nba'].get('picks', []))} picks")

    # ── NHL ──
    print("[NHL] Procesando…")
    results["nhl"] = run_nhl()
    print(f"  → {results['nhl']['status']} | {len(results['nhl'].get('picks', []))} picks")

    # ── MLB ──
    print("[MLB] Procesando…")
    results["mlb"] = run_mlb()
    print(f"  → {results['mlb']['status']} | {len(results['mlb'].get('picks', []))} picks")

    # ── Soccer leagues ──
    for lg in SOCCER_LEAGUES:
        print(f"[{lg.upper()}] Procesando…")
        results[lg] = run_soccer(lg)
        print(f"  → {results[lg]['status']} | {len(results[lg].get('picks', []))} picks")

    # ── Add display names ──
    for key, name in SPORT_NAMES.items():
        if key in results:
            results[key]["name"] = name

    # ── Build parlays ──
    print("\n[Parlays] Calculando combinaciones…")
    parlays = build_all_parlays(results)
    print("  → Parlays generados para", list(parlays["by_sport"].keys()))

    # ── Compose output ──
    output = {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date":         date.today().isoformat(),
        "sports":       results,
        "parlays":      parlays,
    }

    # ── Write JSON ──
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n✅ Guardado en: {OUT_PATH}")
    print(f"   Deportes activos: {[k for k, v in results.items() if v.get('status') == 'ok']}")


if __name__ == "__main__":
    main()
