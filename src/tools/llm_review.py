from __future__ import annotations
"""Optional: Use Claude (Anthropic API) to review training metrics and flag issues.

This is NOT required to run the model.
"""
import json
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)

def review_metrics(metrics_path: Path) -> str:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    client = Anthropic()
    model = (Path(ROOT / ".env").read_text() if False else None)  # no-op; model comes from env via SDK defaults
    prompt = f"""You are a QA-minded ML reviewer.
Given these metrics, list potential risks: data leakage, small sample, poor calibration, class imbalance.
Then suggest 3 concrete improvements. Keep it concise.

Metrics:
{json.dumps(metrics, indent=2)}
"""
    msg = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL","claude-sonnet-4-6"),
        max_tokens=400,
        messages=[{"role":"user","content":prompt}],
    )
    return msg.content[0].text

if __name__ == "__main__":
    import os, sys
    league = sys.argv[1] if len(sys.argv) > 1 else "epl"
    p = ROOT / "models" / league / "metrics.json"
    print(review_metrics(p))
