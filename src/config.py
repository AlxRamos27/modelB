from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MANUAL_DIR = DATA_DIR / "manual"
MODELS_DIR = ROOT / "models"

load_dotenv(ROOT / ".env", override=False)

def env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)

@dataclass(frozen=True)
class LeagueConfig:
    code: str
    name: str
    kind: str  # "soccer" | "nba" | "mlb" | "nhl" | "tennis"
