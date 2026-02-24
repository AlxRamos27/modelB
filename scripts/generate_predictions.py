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
    ANTHROPIC_API_KEY    – optional, used for Claude AI analysis
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
from src.data.espn_api import get_espn_context
from src.data.odds_api import fetch_live_odds
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

def _has_today_games(picks_df, today: date) -> bool:
    """Return True if picks_df has ≥1 game for today or tomorrow UTC (+1 day for timezone gap)."""
    if picks_df is None or picks_df.empty:
        return False
    import pandas as pd
    for _, row in picks_df.iterrows():
        try:
            dt = pd.to_datetime(row["date"], utc=True)
            days_diff = (dt.date() - today).days
            if 0 <= days_diff <= 1:
                return True
        except Exception:
            pass
    return False


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
        picks_df = predict_model(ds, top_n=200)
        # BallDontLie returns the full season (past + future); picks_df may have
        # hundreds of upcoming games but none for today → fall back to ESPN.
        if not _has_today_games(picks_df, date.today()):
            picks_df = _picks_from_espn_schedule("nba", ds, "nba")
        espn_ctx = get_espn_context("nba")
        return _ok(picks_df, metrics, "nba", espn_ctx)
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
        picks_df = predict_model(ds, top_n=200)
        if not _has_today_games(picks_df, date.today()):
            picks_df = _picks_from_espn_schedule("nhl", ds, "nhl")
        espn_ctx = get_espn_context("nhl")
        return _ok(picks_df, metrics, "nhl", espn_ctx)
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
        picks_df = predict_model(ds, top_n=200)
        espn_ctx = get_espn_context("mlb")
        return _ok(picks_df, metrics, "mlb", espn_ctx)
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
        picks_df = predict_model(ds, top_n=200)
        return _ok(picks_df, metrics, league)  # no ESPN for soccer
    except Exception as exc:
        return _error(exc)


# ── ESPN schedule fallback ───────────────────────────────────────────────────

def _picks_from_espn_schedule(sport: str, ds, league: str):
    """Build prediction rows from ESPN scoreboard when BallDontLie has no upcoming games.

    Looks up each team's most recent historical feature values (ELO, form, etc.)
    and applies the trained model to produce picks for today's ESPN games.
    Returns a DataFrame compatible with predict_model() output, or empty df.
    """
    import pandas as pd
    from src.data.espn_api import fetch_espn_scoreboard_games
    from src.modeling import load_model, get_feature_cols

    games = fetch_espn_scoreboard_games(sport)
    if not games:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    try:
        model, classes, feat_cols = load_model(league)
    except Exception:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    df_hist = ds.df[ds.df["is_finished"]].copy().sort_values("date")

    def _find(df_all, team_name, col):
        """Most recent rows for a team in given column (substring match)."""
        t = team_name.lower()
        mask = df_all[col].str.lower().str.contains(t.split()[-1], regex=False, na=False)
        return df_all[mask]

    def _latest(df_sub, col, fallback=0.0):
        if df_sub.empty or col not in df_sub.columns:
            return fallback
        v = df_sub[col].dropna()
        return float(v.iloc[-1]) if not v.empty else fallback

    rows = []
    for g in games:
        home, away = g["home_team"], g["away_team"]
        home_h = _find(df_hist, home, "home_team")  # home team's home games
        away_a = _find(df_hist, away, "away_team")  # away team's away games

        elo_home = _latest(home_h, "elo_home_pre", 1500.0)
        elo_away = _latest(away_a, "elo_away_pre", 1500.0)

        home_wr = _latest(home_h, "home_winrate_l5", 0.5)
        away_wr = _latest(away_a, "away_winrate_l5", 0.5)

        feat: dict = {
            "elo_diff_pre":    elo_home - elo_away,
            "home_winrate_l5": home_wr,
            "away_winrate_l5": away_wr,
            "diff_l5_gap":     home_wr - away_wr,
            "inj_gap":         0.0,
        }
        if "net_rtg_diff" in feat_cols:
            feat["net_rtg_diff"]     = _latest(home_h, "net_rtg_diff", 0.0)
            feat["pace_avg"]         = _latest(home_h, "pace_avg", 98.5)
            feat["elo_538_diff_pre"] = _latest(home_h, "elo_538_diff_pre", 0.0)

        rows.append({"date": g["date"], "home_team": home, "away_team": away, **feat})

    if not rows:
        return pd.DataFrame(columns=["date", "home_team", "away_team"])

    out = pd.DataFrame(rows)
    X = out[feat_cols].fillna(0.0).to_numpy()
    proba = model.predict_proba(X)
    for i, c in enumerate(classes):
        out[f"p_{c}"] = proba[:, i]
    out["p_max"] = out[[f"p_{c}" for c in classes]].max(axis=1)
    out["top_pick"] = (
        out[[f"p_{c}" for c in classes]]
        .idxmax(axis=1)
        .str.replace("p_", "", regex=False)
    )
    return out


# ── Result builders ──────────────────────────────────────────────────────────

def _match_injuries(injuries_by_team: dict, team_name: str) -> list[dict]:
    """Find injuries for a team using partial name matching."""
    if team_name in injuries_by_team:
        return injuries_by_team[team_name]
    tl = team_name.lower()
    for key, val in injuries_by_team.items():
        if not isinstance(key, str):
            continue
        kl = key.lower()
        if tl in kl or kl in tl:
            return val
        # match by nickname (last word)
        if tl.split()[-1] == kl.split()[-1]:
            return val
    return []


def _model_spread(p_win: float, kind: str) -> float | None:
    """Convert win probability to implied point spread.

    Uses sport-specific standard deviation of margin of victory:
    - NBA: ~14.6 pts  → spread = 14.6 * ln(p/(1-p))
    - NHL: ~1.5 goals → spread = 1.5 * ln(p/(1-p))
    Soccer: not applicable (returns None)
    """
    import math
    if p_win <= 0.01 or p_win >= 0.99:
        return None
    logit = math.log(p_win / (1 - p_win))
    sd = {"nba": 14.6, "nhl": 1.5, "mlb": 1.8}.get(kind)
    if sd is None:
        return None
    return round(logit * sd, 1)


def _build_odds_lookup(sport: str) -> dict:
    """Fetch live odds and return dict keyed by (home_team, away_team)."""
    try:
        path = fetch_live_odds(sport)
        if not path or not path.exists():
            return {}
        import pandas as pd
        df = pd.read_csv(path)
        lookup = {}
        for _, row in df.iterrows():
            key = (str(row.get("home_team", "")), str(row.get("away_team", "")))
            lookup[key] = {
                "odds_H": row.get("odds_H"),
                "odds_A": row.get("odds_A"),
                "spread_home": row.get("spread_home"),
                "spread_away": row.get("spread_away"),
                "total_over": row.get("total_over"),
            }
        return lookup
    except Exception as exc:
        warnings.warn(f"Odds lookup failed for {sport}: {exc}")
        return {}


def _claude_game_notes(sport: str, picks: list[dict], espn_ctx: dict | None = None) -> list[str]:
    """One Claude call → one rich note per pick (records + injuries + O/U + recommendation)."""
    api_key = env("ANTHROPIC_API_KEY")
    if not api_key or not picks:
        return _fallback_notes(picks, espn_ctx)
    try:
        import anthropic, json as _json
        client = anthropic.Anthropic(api_key=api_key)
        label = SPORT_LABELS.get(sport, sport.upper())

        # Build record lookup from ALL teams (not just top 5)
        standings: dict[str, str] = {}
        all_teams = (espn_ctx or {}).get("all_teams") or (espn_ctx or {}).get("top_teams") or []
        for t in all_teams:
            team = str(t.get("team") or "")
            if team:
                w = int(t.get("wins") or 0)
                l = int(t.get("losses") or 0)
                standings[team.lower()] = f"{w}-{l}"

        def get_record(team_name: str) -> str:
            tl = team_name.lower()
            for k, v in standings.items():
                if tl in k or k in tl or tl.split()[-1] == k.split()[-1]:
                    return v
            return ""

        lines = []
        for i, p in enumerate(picks):
            inj_h = "; ".join(f"{x['player']} ({x['status']})" for x in p.get("injuries_home", [])[:3]) or "sin bajas"
            inj_a = "; ".join(f"{x['player']} ({x['status']})" for x in p.get("injuries_away", [])[:3]) or "sin bajas"
            rec_h = get_record(p["home_team"])
            rec_a = get_record(p["away_team"])
            model_s = p.get("model_spread")
            house_s = p.get("house_spread")
            edge = p.get("edge_pct")
            total = p.get("house_total")

            align = ""
            if edge is not None:
                align = "✅ líneas alineadas" if abs(edge) <= 3 else f"⚠️ diferencia {edge:+.1f}%"

            parts = [
                f"{i+1}. {p['home_team']}{' (' + rec_h + ')' if rec_h else ''} vs "
                f"{p['away_team']}{' (' + rec_a + ')' if rec_a else ''}",
                f"pick: {p['pick_label']} | modelo {p['p_win']*100:.0f}% | señal {p['signal']}",
                f"bajas local: {inj_h}",
                f"bajas visitante: {inj_a}",
            ]
            if model_s is not None:
                parts.append(f"spread modelo: {model_s:+.1f} pts")
            if house_s is not None:
                parts.append(f"línea casa: {house_s:+.1f} pts")
            if total is not None:
                parts.append(f"O/U: {total} puntos")
            if align:
                parts.append(align)
            lines.append(" | ".join(parts))

        prompt = (
            f"Eres analista experto de {label}. Para CADA partido genera UNA nota analítica en español de 40-50 palabras.\n\n"
            f"CADA nota DEBE incluir EN ESTE ORDEN:\n"
            f"1. ✅ (modelo y casa alineados, diferencia <3%) o ⚠️ (diferencia significativa ≥3%)\n"
            f"2. Récord de AMBOS equipos (ej: '34-21 vs 28-27')\n"
            f"3. Lesiones importantes si las hay (menciona el jugador y su impacto)\n"
            f"4. Si hay O/U disponible: 'Alta [X] pts' o 'Baja [X] pts' con justificación breve\n"
            f"5. Recomendación final: 'Apostar ML [equipo]', 'Apostar Alta/Baja [X]', o 'SKIP'\n\n"
            f"Sé directo y específico. Usa datos del partido, no generalidades.\n"
            f"Responde ÚNICAMENTE con un array JSON de strings (mismo orden que los partidos).\n\n"
            + "\n".join(lines)
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1600,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        s, e = text.find("["), text.rfind("]") + 1
        if s >= 0 and e > s:
            notes = _json.loads(text[s:e])
            return [str(n) for n in notes]
    except Exception as exc:
        warnings.warn(f"Claude game notes failed for {sport}: {exc}")
    return _fallback_notes(picks, espn_ctx)


def _fallback_notes(picks: list[dict], espn_ctx: dict | None = None) -> list[str]:
    """Generate structured notes from model data when Claude is unavailable.

    Includes team records (from ESPN standings), injuries, O/U line, and
    a clear bet recommendation — everything the Claude prompt would produce.
    """
    # Build standings lookup: team_name.lower() → "W-L"
    standings: dict[str, str] = {}
    all_teams = (espn_ctx or {}).get("all_teams") or (espn_ctx or {}).get("top_teams") or []
    for t in all_teams:
        team = str(t.get("team") or "")
        if team:
            w = int(t.get("wins") or 0)
            l = int(t.get("losses") or 0)
            standings[team.lower()] = f"{w}-{l}"

    def _record(name: str) -> str:
        tl = name.lower()
        for k, v in standings.items():
            if tl in k or k in tl or (tl.split()[-1] == k.split()[-1]):
                return v
        return ""

    notes = []
    for p in picks:
        icon = "✅" if p["signal"] == "alta" else "⚠️" if p["signal"] == "media" else "❌"

        rec_h = _record(p["home_team"])
        rec_a = _record(p["away_team"])
        records = f"{rec_h} vs {rec_a}" if (rec_h and rec_a) else (rec_h or rec_a)

        # Key injuries (up to 2 per side)
        inj_parts = []
        for inj in (p.get("injuries_home") or [])[:2]:
            inj_parts.append(f"{inj.get('player', '')} {inj.get('status', '')} (local)")
        for inj in (p.get("injuries_away") or [])[:2]:
            inj_parts.append(f"{inj.get('player', '')} {inj.get('status', '')} (visit.)")
        inj_text = "; ".join(inj_parts)

        # O/U recommendation based on model spread vs total
        total = p.get("house_total")
        model_s = p.get("model_spread")
        ou_text = ""
        if total is not None:
            # If model spread implies high-scoring game → Alta, else → Baja
            if model_s is not None and abs(model_s) < 3:
                ou_text = f"Alta {total} pts"
            else:
                ou_text = f"O/U {total} pts"

        # Edge
        edge = p.get("edge_pct")
        edge_text = f"Edge {edge:+.1f}%" if edge is not None else f"modelo {p['p_win']*100:.0f}%"

        # Assemble: icon records — injuries — O/U → bet edge
        parts = [f"{icon} {records}" if records else icon]
        if inj_text:
            parts.append(inj_text)
        if ou_text:
            parts.append(ou_text)
        parts.append(f"→ Apostar ML {p['pick_label']} ({edge_text})")

        notes.append(" — ".join(parts))
    return notes


def _ok(picks_df, metrics: dict, sport: str, espn_ctx: dict | None = None) -> dict:
    import pandas as pd
    from src.pipeline import Dataset  # just for kind detection
    today = date.today()
    today_str = today.isoformat()

    # Determine sport kind for spread calc
    kind = sport if sport in ("nba", "nhl", "mlb") else "soccer"

    # Fetch live odds (optional — silently skipped if no key)
    odds_lookup = _build_odds_lookup(sport) if kind != "soccer" else {}

    # Build injury lookup by team name
    injuries_by_team: dict = {}
    if espn_ctx and espn_ctx.get("key_injuries"):
        for inj in espn_ctx["key_injuries"]:
            team = inj.get("team") or ""
            # Skip NaN / float values that pandas inserts for empty CSV cells
            if not isinstance(team, str) or not team.strip():
                continue
            injuries_by_team.setdefault(team, []).append({
                "player": str(inj.get("player") or ""),
                "status": str(inj.get("status") or ""),
            })

    picks = []
    for _, row in picks_df.iterrows():
        # ── Parse game date first so we can filter ──
        game_date = today_str
        game_date_obj = today
        try:
            dt = pd.to_datetime(row["date"], utc=True)
            game_date = dt.strftime("%Y-%m-%d")
            game_date_obj = dt.date()
        except Exception:
            pass

        # Filter: only keep today's games.
        # +1 day margin covers UTC/local timezone gap (e.g. 10 PM ET = next day UTC).
        days_diff = (game_date_obj - today).days
        if days_diff < 0 or days_diff > 1:
            continue

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
        # Base signal on rounded % so display (70%) matches badge (Alta)
        conf_pct = round(p_win * 100)
        signal = "alta" if conf_pct >= 70 else "media" if conf_pct >= 60 else "baja"

        home_team = str(row["home_team"])
        away_team = str(row["away_team"])

        # House odds lookup
        house = odds_lookup.get((home_team, away_team), {})
        house_ml_h = house.get("odds_H")   # decimal moneyline home
        house_ml_a = house.get("odds_A")   # decimal moneyline away
        house_spread_home = house.get("spread_home")
        house_spread_away = house.get("spread_away")
        house_total = house.get("total_over")  # O/U line

        # Model spread (implied from p_win)
        model_s = _model_spread(p_win if pick == "H" else 1 - p_win, kind)
        if model_s is not None and pick == "A":
            model_s = -model_s  # flip sign for away pick

        # House spread for the picked side
        house_s = house_spread_home if pick == "H" else house_spread_away

        # Edge = model% - house implied%
        house_implied = None
        if pick == "H" and house_ml_h:
            house_implied = round(1.0 / house_ml_h * 100, 1)
        elif pick == "A" and house_ml_a:
            house_implied = round(1.0 / house_ml_a * 100, 1)
        edge_pct = round(p_win * 100 - house_implied, 1) if house_implied else None

        picks.append({
            "home_team":        home_team,
            "away_team":        away_team,
            "pick":             pick,
            "pick_label":       str(pick_label),
            "p_win":            round(p_win, 4),
            "implied_odds":     implied_odds,
            "signal":           signal,
            "date":             game_date,
            "model_spread":     model_s,
            "house_spread":     house_s,
            "house_odds":       house_ml_h if pick == "H" else house_ml_a,
            "house_implied_pct": house_implied,
            "edge_pct":         edge_pct,
            "house_total":      house_total,
            "injuries_home":    _match_injuries(injuries_by_team, home_team),
            "injuries_away":    _match_injuries(injuries_by_team, away_team),
            "note":             "",
        })

    # Generate per-game Claude notes
    if picks:
        notes = _claude_game_notes(sport, picks, espn_ctx)
        for i, note in enumerate(notes):
            if i < len(picks):
                picks[i]["note"] = note

    # Sort by p_win descending (highest confidence first)
    picks.sort(key=lambda x: -x["p_win"])

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


# ── Claude AI analysis ───────────────────────────────────────────────────────

SPORT_LABELS = {
    "nba": "NBA", "nhl": "NHL", "mlb": "MLB",
    "epl": "Premier League", "laliga": "La Liga", "ligue1": "Ligue 1",
    "bundesliga": "Bundesliga", "primeira": "Primeira Liga", "ucl": "Champions League",
}


def claude_analysis(sport: str, picks: list[dict], metrics: dict) -> str:
    """Call Claude to generate a brief narrative analysis for a sport's picks."""
    api_key = env("ANTHROPIC_API_KEY")
    if not api_key or not picks:
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        top = picks[:5]
        picks_summary = "\n".join(
            f"- {p['home_team']} vs {p['away_team']}: pick={p['pick_label']} ({p['p_win']*100:.1f}% confianza, señal {p['signal']})"
            for p in top
        )
        acc = metrics.get("accuracy", 0)
        label = SPORT_LABELS.get(sport, sport.upper())

        # Enrich with ESPN context for North American sports
        espn_ctx = {}
        if sport in ("nba", "nhl", "mlb"):
            try:
                espn_ctx = get_espn_context(sport)
            except Exception:
                pass

        context_lines = []
        if espn_ctx.get("top_teams"):
            top_teams = ", ".join(t["team"] for t in espn_ctx["top_teams"][:3])
            context_lines.append(f"Líderes actuales: {top_teams}.")
        if espn_ctx.get("key_injuries"):
            inj = espn_ctx["key_injuries"][:3]
            inj_str = ", ".join(f"{i['player']} ({i['team']}, {i['status']})" for i in inj)
            context_lines.append(f"Lesiones clave: {inj_str}.")
        context_block = " ".join(context_lines)

        prompt = (
            f"Eres un analista deportivo. Genera un análisis breve (2-3 oraciones en español) "
            f"para los picks de {label} de hoy. "
            f"El modelo tiene {acc*100:.1f}% de precisión histórica. "
            f"Picks destacados:\n{picks_summary}\n"
            + (f"Contexto adicional (ESPN): {context_block}\n" if context_block else "")
            + "Menciona los partidos más interesantes y la confianza general del modelo hoy. "
            f"No garantices ganancias. Sé directo y concreto."
        )

        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=250,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as exc:
        warnings.warn(f"Claude analysis failed for {sport}: {exc}")
        return ""


def add_claude_analysis(results: dict) -> None:
    """Add Claude analysis field to each sport result in-place."""
    api_key = env("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [Claude] ANTHROPIC_API_KEY no configurada — análisis omitido")
        return
    all_sports = list(SPORT_LABELS.keys())
    for sport in all_sports:
        res = results.get(sport, {})
        if res.get("status") != "ok" or not res.get("picks"):
            continue
        print(f"  [Claude] Analizando {SPORT_LABELS[sport]}…")
        res["analysis"] = claude_analysis(sport, res["picks"], res.get("metrics", {}))


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

    # ── Claude AI analysis ──
    print("\n[Claude] Generando análisis…")
    add_claude_analysis(results)

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
