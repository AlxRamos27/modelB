Task: Run an end-to-end pipeline for a league and summarize results.

Steps:
1) `python -m src.cli fetch --league <league> --season <season>`
2) `python -m src.cli train --league <league>`
3) `python -m src.cli predict --league <league> --top 3`
4) Summarize:
   - dataset size
   - train/validation metrics (accuracy, logloss, brier)
   - top 3 upcoming games by predicted probability
   - any warnings/data gaps
