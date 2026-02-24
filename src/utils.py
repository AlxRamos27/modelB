from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Any, Dict

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def sha1_dict(d: Dict[str, Any]) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(s).hexdigest()
