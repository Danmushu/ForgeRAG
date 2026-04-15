"""
Benchmark report utilities.

- Config sanitization (strip secrets)
- Report JSON assembly for download
"""

from __future__ import annotations

import re
from typing import Any

# Fields that should be redacted in the config snapshot
_SECRET_PATTERNS = re.compile(r"(api_key|password|secret|token|credential)", re.IGNORECASE)


def sanitize_config(cfg) -> dict:
    """Dump AppConfig to dict with all secret-like fields redacted."""
    raw = cfg.model_dump()
    return _redact(raw)


def _redact(obj: Any) -> Any:
    if isinstance(obj, dict):
        result = {}
        for k, v in obj.items():
            if _SECRET_PATTERNS.search(k) and isinstance(v, str) and v:
                result[k] = "***"
            else:
                result[k] = _redact(v)
        return result
    if isinstance(obj, list):
        return [_redact(item) for item in obj]
    return obj


def build_report(status: dict) -> dict:
    """Build the full downloadable report from a benchmark status dict."""
    return {
        "benchmark_report": {
            "run_id": status.get("run_id", ""),
            "status": status.get("status", ""),
            "elapsed_ms": status.get("elapsed_ms", 0),
            "scores": status.get("scores", {}),
            "method": status.get("method", ""),
        },
        "items": status.get("items", []),
        "config_snapshot": status.get("config_snapshot", {}),
    }
