from __future__ import annotations

import json
from typing import Any


def parse_json_object(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    parsed = _json_loads(candidate)
    if isinstance(parsed, dict):
        return parsed
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        parsed = _json_loads(candidate[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    return None


def _json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None
