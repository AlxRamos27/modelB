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
from datetime import date, datetime, timezone, timedelta
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
from src.data.espn_api import get_espn_context, fetch_espn_scoreboard_games  # noqa: F401 (scoreboard used inside _picks_from_espn_schedule)
from src.data.odds_api import fetch_live_odds
from src.pipeline import build_dataset_soccer, build_dataset_two_way
from src.modeling import train as train_model, predict as predict_model

# ── Constants ────────────────────────────────────────────────────────────────
# NBA/NHL games are played in US timezones.  Use Pacific Time (UTC-8 in winter,
# UTC-7 in summer) as the reference so a 10 PM PT game on Feb 24 stays on
# Feb 24 rather than being labelled Feb 25 (UTC).
_PT = timezone(timedelta(hours=-8))   # PST — close enough year-round for NBA

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


# ── Standings from local dataset (ESPN standings API no longer public) ────────

def _standings_from_dataset(ds) -> list[dict]:
    """Compute W-L records from the historical dataset already in memory.

    ESPN's standings endpoint no longer returns usable data.  We compute
    team records from the BallDontLie / NHL / MLB history instead.
    Returns a list of {team, wins, losses, win_pct} dicts sorted by win_pct.
    """
    import pandas as pd
    df = ds.df[ds.df["is_finished"]].copy()
    if df.empty or "home_team" not in df.columns or ds.target_col not in df.columns:
        return []

    wins: dict[str, int] = {}
    losses: dict[str, int] = {}
    for _, row in df.iterrows():
        ht = str(row["home_team"])
        at = str(row["away_team"])
        result = str(row.get(ds.target_col, ""))
        wins.setdefault(ht, 0); losses.setdefault(ht, 0)
        wins.setdefault(at, 0); losses.setdefault(at, 0)
        if result == "H":
            wins[ht] += 1; losses[at] += 1
        elif result == "A":
            wins[at] += 1; losses[ht] += 1

    teams = []
    for team in wins:
        w, l = wins[team], losses[team]
        total = w + l
        teams.append({"team": team, "wins": w, "losses": l,
                      "win_pct": round(w / total, 3) if total else 0.0})
    return sorted(teams, key=lambda x: -x["win_pct"])


# ── Per-sport runner ─────────────────────────────────────────────────────────

def _parse_game_date(date_val) -> date:
    """Parse a game date to a US Pacific date object.

    - ESPN timestamps ("2026-02-25T03:30Z") are UTC → convert to PT so a
      7:30 PM PT game on Feb 24 is stored as Feb 24, not Feb 25.
    - BallDontLie / NHL plain dates ("2026-02-24") have no time component
      and are already in US local time → keep as-is.
    """
    import pandas as pd
    s = str(date_val)
    if "T" in s or "Z" in s:
        dt_utc = pd.to_datetime(s, utc=True)
        return dt_utc.astimezone(_PT).date()
    return pd.to_datetime(s).date()



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
        # BallDontLie is for training only — its schedule data is unreliable.
        # ALWAYS use ESPN scoreboard as the authoritative source for today's games.
        picks_df = _picks_from_espn_schedule("nba", ds, "nba")
        espn_ctx = get_espn_context("nba")
        if not espn_ctx.get("all_teams"):
            espn_ctx["all_teams"] = _standings_from_dataset(ds)
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
        picks_df = _picks_from_espn_schedule("nhl", ds, "nhl")
        espn_ctx = get_espn_context("nhl")
        if not espn_ctx.get("all_teams"):
            espn_ctx["all_teams"] = _standings_from_dataset(ds)
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
        if not espn_ctx.get("all_teams"):
            espn_ctx["all_teams"] = _standings_from_dataset(ds)
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

    def _avg_pts(df_sub: "pd.DataFrame", score_col: str, n: int = 10):
        """Average points in score_col over last n rows; None if insufficient data."""
        if df_sub.empty or score_col not in df_sub.columns:
            return None
        vals = df_sub[score_col].dropna().tail(n)
        return round(float(vals.mean()), 1) if len(vals) >= 3 else None

    rows = []
    for g in games:
        home, away = g["home_team"], g["away_team"]
        home_h = _find(df_hist, home, "home_team")  # home team's home games
        away_a = _find(df_hist, away, "away_team")  # away team's away games
        # Also grab away-team's road games for allowed-pts lookup
        home_a = _find(df_hist, home, "away_team")
        away_h = _find(df_hist, away, "home_team")

        elo_home = _latest(home_h, "elo_home_pre", 1500.0)
        elo_away = _latest(away_a, "elo_away_pre", 1500.0)

        home_wr = _latest(home_h, "home_winrate_l5", 0.5)
        away_wr = _latest(away_a, "away_winrate_l5", 0.5)

        pace_avg = _latest(home_h, "pace_avg", 98.5)

        # ── Scoring stats for O/U estimation ──────────────────────────────────
        # home team: pts scored at home, pts allowed at home
        pts_home_scored  = _avg_pts(home_h, "home_score")
        pts_home_allowed = _avg_pts(home_h, "away_score")
        # away team: pts scored on road, pts allowed on road
        pts_away_scored  = _avg_pts(away_a, "away_score")
        pts_away_allowed = _avg_pts(away_a, "home_score")
        # combined recent avg (both teams' scoring)
        pts_home_all = _avg_pts(
            pd.concat([home_h[["home_score"]].rename(columns={"home_score": "_s"}),
                       home_a[["away_score"]].rename(columns={"away_score": "_s"})]),
            "_s", n=10,
        ) if not home_h.empty or not home_a.empty else None
        pts_away_all = _avg_pts(
            pd.concat([away_h[["home_score"]].rename(columns={"home_score": "_s"}),
                       away_a[["away_score"]].rename(columns={"away_score": "_s"})]),
            "_s", n=10,
        ) if not away_h.empty or not away_a.empty else None

        feat: dict = {
            "elo_diff_pre":       elo_home - elo_away,
            "home_winrate_l5":    home_wr,
            "away_winrate_l5":    away_wr,
            "diff_l5_gap":        home_wr - away_wr,
            "inj_gap":            0.0,
            # O/U context — kept outside feat_cols, used by _ok() and Claude
            "pace_avg":           pace_avg,
            "pts_home_scored":    pts_home_scored  or 0.0,
            "pts_home_allowed":   pts_home_allowed or 0.0,
            "pts_away_scored":    pts_away_scored  or 0.0,
            "pts_away_allowed":   pts_away_allowed or 0.0,
            "pts_home_avg":       pts_home_all     or 0.0,
            "pts_away_avg":       pts_away_all     or 0.0,
        }
        if "net_rtg_diff" in feat_cols:
            feat["net_rtg_diff"]     = _latest(home_h, "net_rtg_diff", 0.0)
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


def _claude_game_notes(sport: str, picks: list[dict], espn_ctx: dict | None = None) -> list[dict]:
    """One Claude call → one rich note + O/U pick per game.

    Returns list of {"note": str, "ou_pick": "over"|"under"|None}.
    Claude decides Over/Under based on team tendencies, injuries and pace —
    more accurate than the pace-formula fallback.
    """
    api_key = env("ANTHROPIC_API_KEY")
    if not api_key or not picks:
        return _fallback_notes(picks, espn_ctx)
    try:
        import anthropic, json as _json
        client = anthropic.Anthropic(api_key=api_key)
        label = SPORT_LABELS.get(sport, sport.upper())

        # Build record lookup
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
            edge = p.get("edge_pct")
            ou_line = p.get("ou_line") or p.get("house_total")
            model_s = p.get("model_spread")
            house_s = p.get("house_spread")

            # Scoring context for O/U
            pts_hs = p.get("pts_home_scored")   # home team avg pts scored at home
            pts_ha = p.get("pts_home_allowed")  # home team avg pts allowed at home
            pts_as = p.get("pts_away_scored")   # away team avg pts scored away
            pts_aa = p.get("pts_away_allowed")  # away team avg pts allowed away

            align = ""
            if edge is not None:
                align = "✅ alineado" if abs(edge) <= 3 else f"⚠️ diferencia {edge:+.1f}%"

            parts = [
                f"{i+1}. {p['home_team']}{' (' + rec_h + ')' if rec_h else ''} vs "
                f"{p['away_team']}{' (' + rec_a + ')' if rec_a else ''}",
                f"pick: {p['pick_label']} | modelo {p['p_win']*100:.0f}% | señal {p['signal']}",
                f"bajas local: {inj_h}",
                f"bajas visitante: {inj_a}",
            ]
            if model_s is not None:
                parts.append(f"spread modelo: {model_s:+.1f}")
            if house_s is not None:
                parts.append(f"línea casa: {house_s:+.1f}")
            if ou_line is not None:
                parts.append(f"total estimado: {ou_line} pts")
            # Scoring detail for Claude's O/U analysis
            if pts_hs and pts_hs > 0:
                parts.append(f"local anota {pts_hs} pts/partido en casa")
            if pts_ha and pts_ha > 0:
                parts.append(f"local permite {pts_ha} pts/partido en casa")
            if pts_as and pts_as > 0:
                parts.append(f"visit. anota {pts_as} pts/partido fuera")
            if pts_aa and pts_aa > 0:
                parts.append(f"visit. permite {pts_aa} pts/partido fuera")
            if align:
                parts.append(align)
            lines.append(" | ".join(parts))

        prompt = (
            f"Eres analista experto de {label} con profundo conocimiento de tendencias de apuestas.\n"
            f"Para CADA partido genera una nota analítica en español de 80-100 palabras y una predicción Over/Under.\n\n"
            f"Para decidir Over/Under considera EN ESTE ORDEN:\n"
            f"1. Promedios de puntos anotados/permitidos de cada equipo (los datos te los doy)\n"
            f"2. Lesiones de jugadores clave que afecten el ataque o defensa\n"
            f"3. Ventaja de localía (los locales suelen anotar más en casa)\n"
            f"4. Ritmo de juego: equipos de ritmo alto → Over; defensivos → Under\n"
            f"5. Racha reciente: ¿los últimos partidos fueron de muchos o pocos puntos?\n\n"
            f"Estructura OBLIGATORIA de la nota:\n"
            f"- ✅ / ⚠️ + récords con narrativa (ej: 'Bulls 31-25 en forma; Hornets 18-38 en caída')\n"
            f"- Lesiones clave por nombre\n"
            f"- O/U: explica con los datos (ej: 'local anota 118/g, visita permite 121/g → Over claro')\n"
            f"- Veredicto ML: 'ML [equipo] es la jugada' o 'SKIP'\n\n"
            f"IMPORTANTE: Usa los datos numéricos exactos que te doy. No inventes estadísticas.\n\n"
            f"Responde ÚNICAMENTE con un array JSON de objetos (mismo orden que los partidos):\n"
            f'[{{"note": "texto aquí", "ou": "over"}}, {{"note": "...", "ou": "under"}}, ...]\n'
            f'Valores válidos para "ou": "over" o "under"\n\n'
            + "\n".join(lines)
        )
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2800,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text.strip()
        s, e = text.find("["), text.rfind("]") + 1
        if s >= 0 and e > s:
            results = _json.loads(text[s:e])
            out = []
            for r in results:
                if isinstance(r, dict):
                    ou_raw = str(r.get("ou") or "").lower().strip()
                    ou_pick = "over" if ou_raw == "over" else "under" if ou_raw == "under" else None
                    out.append({"note": str(r.get("note", "")), "ou_pick": ou_pick})
                else:
                    # Fallback: Claude returned a plain string (old format)
                    out.append({"note": str(r), "ou_pick": None})
            return out
    except Exception as exc:
        warnings.warn(f"Claude game notes failed for {sport}: {exc}")
    return _fallback_notes(picks, espn_ctx)


def _fallback_notes(picks: list[dict], espn_ctx: dict | None = None) -> list[dict]:
    """Generate structured notes when Claude is unavailable.

    Returns list of {"note": str, "ou_pick": "over"|"under"|None}.
    Uses pace-based O/U direction since Claude is not available.
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
        # 1. Alignment indicator
        edge = p.get("edge_pct")
        model_s = p.get("model_spread")
        house_s = p.get("house_spread")
        if edge is not None:
            if abs(edge) <= 3:
                align_text = "✅ Alineado"
            else:
                icon = "✅" if p["signal"] == "alta" else "⚠️"
                align_text = f"{icon} Discrepancia {edge:+.1f}%"
        else:
            icon = "✅" if p["signal"] == "alta" else "⚠️" if p["signal"] == "media" else "❌"
            align_text = f"{icon} Modelo {p['p_win']*100:.0f}%"

        # 2. Records with narrative context
        rec_h = _record(p["home_team"])
        rec_a = _record(p["away_team"])
        home_nick = p["home_team"].split()[-1]
        away_nick = p["away_team"].split()[-1]
        if rec_h and rec_a:
            records_text = f"{home_nick} {rec_h} vs {away_nick} {rec_a}"
        elif rec_h:
            records_text = f"{home_nick} {rec_h}"
        elif rec_a:
            records_text = f"{away_nick} {rec_a}"
        else:
            records_text = ""

        # 3. Key injuries by player name
        inj_parts = []
        for inj in (p.get("injuries_home") or [])[:2]:
            player = str(inj.get("player") or "").strip()
            status = str(inj.get("status") or "").strip()
            if player:
                inj_parts.append(f"{player} fuera ({status})")
        for inj in (p.get("injuries_away") or [])[:2]:
            player = str(inj.get("player") or "").strip()
            status = str(inj.get("status") or "").strip()
            if player:
                inj_parts.append(f"{player} fuera (visit.)")
        inj_text = "; ".join(inj_parts) if inj_parts else "Sin bajas importantes"

        # 4. O/U verdict
        ou_line = p.get("ou_line") or p.get("house_total")
        ou_pick = p.get("ou_pick")
        if ou_line is not None:
            direction = "Over" if ou_pick == "over" else "Under" if ou_pick == "under" else "O/U"
            reason    = "ritmo alto" if ou_pick == "over" else "ritmo lento" if ou_pick == "under" else ""
            ou_text = f"O/U en {ou_line} pts — {direction} es la jugada" + (f" ({reason})" if reason else "")
        else:
            ou_text = ""

        # 5. ML verdict
        pct = p["p_win"] * 100
        pick_nick = p["pick_label"].split()[-1]
        if p["signal"] == "baja":
            verdict = f"SKIP — sin edge claro ({pct:.0f}%)"
        elif p["signal"] == "media":
            verdict = f"ML {pick_nick} con cautela (modelo {pct:.0f}%)"
        else:
            verdict = f"ML {pick_nick} es la jugada segura (modelo {pct:.0f}%)"

        # Assemble as readable prose
        parts = [align_text]
        if records_text:
            parts.append(records_text)
        parts.append(inj_text)
        if ou_text:
            parts.append(ou_text)
        parts.append(verdict)

        notes.append({"note": ". ".join(parts) + ".", "ou_pick": ou_pick})
    return notes


def _ok(picks_df, metrics: dict, sport: str, espn_ctx: dict | None = None) -> dict:
    import pandas as pd
    from src.pipeline import Dataset  # just for kind detection
    # Use Pacific Time as the reference so NBA/NHL night games on Feb 24 PT
    # don't bleed into Feb 25 (UTC) and appear as "tomorrow".
    today = datetime.now(_PT).date()
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
        # ── Parse game date in PT so ESPN UTC timestamps (e.g. "2026-02-25T03:30Z")
        #    are stored as their US local date ("2026-02-24") rather than UTC date.
        game_date = today_str
        game_date_obj = today
        try:
            game_date_obj = _parse_game_date(row["date"])
            game_date = game_date_obj.isoformat()
        except Exception:
            pass

        # Filter: only today's PT games (timezone is already resolved by _parse_game_date).
        if game_date_obj != today:
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
        house_total = house.get("total_over")  # O/U line from bookmaker

        # Estimated O/U from model pace when bookmaker line unavailable.
        # NBA: pace_avg (possessions/game) × 2.3 ≈ total points expected.
        # NHL: pace is not applicable; use None.
        # Soccer: goals are low-count, different logic.
        ou_line = house_total  # prefer live bookmaker line
        ou_pick = None         # "over" | "under" | None — overridden by Claude later
        if ou_line is None and kind == "nba":
            try:
                # Best estimate: sum of each team's recent scoring averages.
                # pts_home_scored = home team avg pts at home (last 10 games)
                # pts_away_scored = away team avg pts on road (last 10 games)
                pts_h = float(row.get("pts_home_scored") or 0)
                pts_a = float(row.get("pts_away_scored") or 0)
                if pts_h >= 80 and pts_a >= 80:
                    ou_line = round(pts_h + pts_a, 1)
                else:
                    # Fallback: pace × 2.3 (possessions × ~2.3 pts/possession)
                    pace = float(row.get("pace_avg") or 0)
                    if pace < 50:
                        pace = 98.5
                    ou_line = round(pace * 2.3, 1)
            except Exception:
                ou_line = 226.5

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

        def _f(col): return float(row.get(col) or 0) or None  # None if 0/missing

        picks.append({
            "home_team":         home_team,
            "away_team":         away_team,
            "pick":              pick,
            "pick_label":        str(pick_label),
            "p_win":             round(p_win, 4),
            "implied_odds":      implied_odds,
            "signal":            signal,
            "date":              game_date,
            "model_spread":      model_s,
            "house_spread":      house_s,
            "house_odds":        house_ml_h if pick == "H" else house_ml_a,
            "house_implied_pct": house_implied,
            "edge_pct":          edge_pct,
            "house_total":       house_total,
            "ou_line":           ou_line,
            "ou_pick":           ou_pick,    # overridden by Claude below
            # Scoring stats passed to Claude for O/U analysis
            "pts_home_scored":   _f("pts_home_scored"),
            "pts_home_allowed":  _f("pts_home_allowed"),
            "pts_away_scored":   _f("pts_away_scored"),
            "pts_away_allowed":  _f("pts_away_allowed"),
            "injuries_home":     _match_injuries(injuries_by_team, home_team),
            "injuries_away":     _match_injuries(injuries_by_team, away_team),
            "note":              "",
        })

    # Sort by p_win descending (highest confidence first)
    picks.sort(key=lambda x: -x["p_win"])

    # Generate per-game notes (Claude or fallback).
    # Claude also decides Over/Under direction — overrides the pace estimate.
    if picks:
        note_data = _claude_game_notes(sport, picks, espn_ctx)
        for i, nd in enumerate(note_data):
            if i < len(picks):
                picks[i]["note"] = nd["note"]
                if nd.get("ou_pick"):          # Claude's O/U overrides pace estimate
                    picks[i]["ou_pick"] = nd["ou_pick"]

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
