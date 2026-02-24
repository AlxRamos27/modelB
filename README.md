# Sports Probability Model (Multi‑League) — Python + Claude Code

> **Purpose (important):** this project is for **education / analytics**. It estimates win probabilities from historical performance signals.
> It **does not guarantee profit** and should not be treated as financial advice.

Supported leagues (initial adapters):
- Soccer: La Liga, Premier League, Ligue 1, Bundesliga, Primeira Liga, UEFA Champions League (via football-data.org)
- NBA (via balldontlie NBA API)
- MLB (via MLB Stats API)
- NHL (via NHL public endpoints)
- ATP (initially supported as a **CSV/manual** adapter; see `data/manual/`)

## 1) Quick start (VS Code)

### Prereqs
- Python 3.10+
- Git
- (Optional) Claude Code CLI for agent workflows (see section 6)

### Setup
```bash
# from the repo root
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate

pip install -r requirements.txt
```

### Configure API keys
Copy the example env:
```bash
cp .env.example .env
```

Then edit `.env` (or export env vars) with at least:
- `FOOTBALL_DATA_TOKEN` (football-data.org) — required for soccer
- `BALLDONTLIE_API_KEY` — required for NBA
- MLB/NHL adapters are public (no key)

## 2) Fetch data

```bash
python -m src.cli fetch --league epl --season 2025
python -m src.cli fetch --league laliga --season 2025
python -m src.cli fetch --league nba --season 2025
python -m src.cli fetch --league mlb --season 2025
python -m src.cli fetch --league nhl --season 2025
```

Fetched data is stored in `data/raw/<league>/`.

## 3) Add injuries (optional but recommended)

Injuries are one of the most important features, but sources vary widely.
This repo supports **manual injuries input**:

- Fill `data/manual/injuries.csv` with expected absences and an impact score (0–1).
- The model aggregates impact per team and uses it as a feature.

Template is already provided.

## 4) Train

```bash
python -m src.cli train --league epl
python -m src.cli train --league nba
```

Models and metrics go to `models/<league>/`.

## 5) Predict upcoming games + print top picks

```bash
python -m src.cli predict --league epl --top 3
python -m src.cli predict --league nba --top 3
```

Output includes probability + a confidence band and key drivers.

> If you want **edge vs odds**, you can pass odds CSV:
```bash
python -m src.cli predict --league epl --odds data/manual/odds_example.csv --top 3
```

## 6) Claude Code agent workflows (optional)

Claude Code can operate on this repository (edit files, run commands, etc.). Docs:
- Overview/setup/headless are in Anthropic's Claude Code docs.

This project ships:
- `CLAUDE.md` — project context and conventions for the agent
- `agent/tasks/` — ready-to-run task prompts

Example workflow:
```bash
# In a terminal (after installing Claude Code):
claude
# then ask:
# "Run the epl fetch+train pipeline and summarize model performance."
```

## 7) How the model works (high level)

We build a per‑league dataset of games with:
- **Form**: rolling winrate, goal/run/point differential, streaks
- **Elo ratings** with home advantage
- **Rest/fatigue** (where schedule data supports it)
- **Injury impact** (manual CSV)
- **Season strength** (standings rank / points pace)

Then we train:
- Soccer (3‑way): multinomial logistic regression (win/draw/loss)
- NBA/MLB/NHL (2‑way): logistic regression (win/loss)

These are baseline, explainable models. You can switch to XGBoost/LightGBM later.

---

## League codes
- Soccer: `epl`, `laliga`, `ligue1`, `bundesliga`, `primeira`, `ucl`
- Others: `nba`, `mlb`, `nhl`, `atp` (manual)

