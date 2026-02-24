# Project instructions for Claude Code

You are working in a Python repo that estimates sports win probabilities.
Your goals:
- Keep the pipeline reproducible and explainable (avoid black-box code changes without metrics).
- Prefer small, testable changes.
- Do not claim betting profitability or guarantee wins.

Conventions:
- CLI entrypoint is `python -m src.cli ...`
- Data files:
  - raw downloads: `data/raw/<league>/`
  - processed tables: `data/processed/<league>/`
  - manual inputs: `data/manual/`
- Models and reports: `models/<league>/`

When asked to improve the model:
1) Run `fetch` (if needed), `train`, then `predict --top 3` and report:
   - Accuracy, log loss
   - Calibration (Brier score)
   - Any data issues
2) Propose one improvement at a time with justification and expected impact.

Safe-usage reminder:
- This project is for analytics/education; avoid advice like "guaranteed wins".
