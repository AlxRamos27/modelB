from __future__ import annotations
import argparse
from pathlib import Path
from rich import print

from .leagues import LEAGUES
from .data.football_data import fetch_soccer_matches
from .data.balldontlie_nba import fetch_nba_games
from .data.nba_advanced_stats import fetch_nba_advanced_stats
from .data.elo_538 import fetch_elo_538
from .data.odds_api import fetch_live_odds_nba
from .data.mlb_statsapi import fetch_mlb_schedule
from .data.nhl_api import fetch_nhl_schedule
from .data.manual_atp import ensure_atp_template

from .pipeline import build_dataset_soccer, build_dataset_two_way
from .modeling import train as train_model, predict as predict_model

def cmd_fetch(args):
    league = args.league.lower()
    _extra = {"nba-advanced", "nba-elo"}
    if league not in LEAGUES and league not in _extra:
        raise SystemExit(
            f"Unknown league: {league}. "
            f"Options: {sorted(LEAGUES.keys())} + nba-advanced, nba-elo"
        )
    if league in ("epl","laliga","ligue1","bundesliga","primeira","ucl"):
        p = fetch_soccer_matches(league, args.season)
    elif league == "nba":
        p = fetch_nba_games(args.season)
    elif league == "nba-advanced":
        if not args.season:
            raise SystemExit("--season is required for --league nba-advanced")
        p = fetch_nba_advanced_stats(args.season)
    elif league == "nba-elo":
        p = fetch_elo_538()
    elif league == "mlb":
        p = fetch_mlb_schedule(args.season)
    elif league == "nhl":
        p = fetch_nhl_schedule(args.season)
    elif league == "atp":
        p = ensure_atp_template()
    else:
        raise SystemExit("Adapter not implemented yet.")
    print(f"[green]Saved:[/green] {p}")

def _build_dataset_for_league(league: str, season: int | None = None):
    league = league.lower()
    if league in ("epl","laliga","ligue1","bundesliga","primeira","ucl"):
        # choose latest raw file if season not provided
        raw_dir = Path("data/raw") / league
        if season is None:
            files = sorted(raw_dir.glob("matches_*.csv"))
            if not files:
                raise SystemExit("No raw soccer data found. Run fetch first.")
            raw = files[-1]
        else:
            raw = raw_dir / f"matches_{season}.csv"
        return build_dataset_soccer(raw, league=league)

    if league == "nba":
        raw_dir = Path("data/raw/nba")
        raw = (raw_dir / f"games_{season}.csv") if season else sorted(raw_dir.glob("games_*.csv"))[-1]
        return build_dataset_two_way(raw, league="nba", kind="nba", date_col="date", home_score="home_score", away_score="away_score")

    if league == "mlb":
        raw_dir = Path("data/raw/mlb")
        raw = (raw_dir / f"schedule_{season}.csv") if season else sorted(raw_dir.glob("schedule_*.csv"))[-1]
        return build_dataset_two_way(raw, league="mlb", kind="mlb", date_col="date", home_score="home_score", away_score="away_score")

    if league == "nhl":
        raw_dir = Path("data/raw/nhl")
        raw = (raw_dir / f"schedule_{season}.csv") if season else sorted(raw_dir.glob("schedule_*.csv"))[-1]
        return build_dataset_two_way(raw, league="nhl", kind="nhl", date_col="date", home_score="home_score", away_score="away_score")

    if league == "atp":
        raise SystemExit("ATP adapter is manual for now. Put matches in data/raw/atp/ and implement build_dataset_tennis.")
    raise SystemExit(f"No dataset builder for league: {league}")

def cmd_train(args):
    ds = _build_dataset_for_league(args.league, args.season)
    res = train_model(ds)
    print("[bold green]Trained successfully[/bold green]")
    print(res.metrics)

def cmd_predict(args):
    from datetime import date as _date

    odds_path = Path(args.odds) if args.odds else None

    if getattr(args, "auto_odds", False):
        if args.league.lower() != "nba":
            print("[yellow]Warning: --auto-odds is only supported for --league nba.[/yellow]")
        else:
            try:
                odds_path = fetch_live_odds_nba(today=_date.today())
                print(f"[green]Live odds saved:[/green] {odds_path}")
            except Exception as exc:
                print(f"[yellow]Warning: Could not fetch live odds: {exc}. Proceeding without odds.[/yellow]")
                odds_path = None

    ds = _build_dataset_for_league(args.league, args.season)
    out = predict_model(ds, top_n=args.top, odds_csv=odds_path)
    print(out.to_string(index=False))

def main():
    ap = argparse.ArgumentParser(prog="sports-prob-model")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("fetch", help="Download raw data for a league/season")
    a.add_argument("--league", required=True)
    a.add_argument("--season", type=int, required=True)
    a.set_defaults(fn=cmd_fetch)

    b = sub.add_parser("train", help="Train a model for a league (uses latest downloaded season by default)")
    b.add_argument("--league", required=True)
    b.add_argument("--season", type=int, default=None)
    b.set_defaults(fn=cmd_train)

    c = sub.add_parser("predict", help="Predict upcoming games and print top picks")
    c.add_argument("--league", required=True)
    c.add_argument("--season", type=int, default=None)
    c.add_argument("--top", type=int, default=3)
    c.add_argument("--odds", type=str, default=None, help="Optional odds CSV (decimal odds).")
    c.add_argument("--auto-odds", action="store_true", default=False,
                   help="Auto-fetch live NBA odds from The Odds API (requires ODDS_API_KEY in .env).")
    c.set_defaults(fn=cmd_predict)

    args = ap.parse_args()
    args.fn(args)

if __name__ == "__main__":
    main()
