from __future__ import annotations
import time
import requests
from typing import Any, Dict, Optional

class HttpClient:
    def __init__(self, timeout_s: int = 30, min_delay_s: float = 0.25):
        self.timeout_s = timeout_s
        self.min_delay_s = min_delay_s
        self._last = 0.0

    def get_json(self, url: str, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Any:
        now = time.time()
        dt = now - self._last
        if dt < self.min_delay_s:
            time.sleep(self.min_delay_s - dt)
        r = requests.get(url, headers=headers, params=params, timeout=self.timeout_s)
        self._last = time.time()
        r.raise_for_status()
        return r.json()
