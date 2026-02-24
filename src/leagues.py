from __future__ import annotations
from .config import LeagueConfig

LEAGUES: dict[str, LeagueConfig] = {
    # soccer (football-data.org competition codes are different; we map our short code -> API competition code)
    "epl": LeagueConfig(code="epl", name="Premier League (England)", kind="soccer"),
    "laliga": LeagueConfig(code="laliga", name="La Liga (Spain)", kind="soccer"),
    "ligue1": LeagueConfig(code="ligue1", name="Ligue 1 (France)", kind="soccer"),
    "bundesliga": LeagueConfig(code="bundesliga", name="Bundesliga (Germany)", kind="soccer"),
    "primeira": LeagueConfig(code="primeira", name="Primeira Liga (Portugal)", kind="soccer"),
    "ucl": LeagueConfig(code="ucl", name="UEFA Champions League", kind="soccer"),

    "nba": LeagueConfig(code="nba", name="NBA", kind="nba"),
    "mlb": LeagueConfig(code="mlb", name="MLB", kind="mlb"),
    "nhl": LeagueConfig(code="nhl", name="NHL", kind="nhl"),

    # Tennis: manual CSV adapter to start (ATP matches are often behind paywalls; you can plug in your data source later)
    "atp": LeagueConfig(code="atp", name="ATP (Tennis)", kind="tennis"),
}

SOCCER_COMPETITION_CODES = {
    "epl": "PL",
    "laliga": "PD",
    "ligue1": "FL1",
    "bundesliga": "BL1",
    "primeira": "PPL",
    "ucl": "CL",
}
