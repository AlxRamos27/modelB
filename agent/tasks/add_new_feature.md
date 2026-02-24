Task: Add a new feature and evaluate it.

Rules:
- Add feature to `src/features/*.py`
- Ensure it is computed for all rows (no leakage)
- Retrain and compare metrics. If worse, revert.

Example feature ideas:
- Recent head-to-head (last 3 meetings)
- Travel distance proxy (for NBA/NHL: back-to-back + road trip length)
- Goalkeeper/starting pitcher proxy (if available)
