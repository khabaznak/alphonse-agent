from __future__ import annotations

from typing import Any


def triage_gap_text(text: str) -> dict[str, Any]:
    normalized = str(text or "").strip()
    if not normalized:
        return {"category": "intent_missing", "suggested_intent": None, "confidence": 0.1}
    # Intentionally heuristic-light: route unknowns into reflection/proposal pipeline.
    return {"category": "intent_missing", "suggested_intent": None, "confidence": 0.3}


def detect_language(text: str) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return "und"
    return "und"
