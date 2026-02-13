from __future__ import annotations

from collections import Counter
from typing import Any

def build_daily_report(locale: str, gaps: list[dict[str, Any]]) -> str:
    _ = locale
    total = len(gaps)
    if total == 0:
        return "No capability gaps were reported today."
    reasons = Counter([gap.get("reason") or "unknown" for gap in gaps])
    open_count = sum(1 for gap in gaps if gap.get("status") == "open")
    header = f"Daily capability gaps: total={total}, open={open_count}"
    lines = []
    for reason, count in reasons.most_common(3):
        example = _example_for_reason(gaps, reason)
        line = f"- reason={_label_for_reason(reason)}, count={count}, example={example}"
        lines.append(line)
    return "\n".join([header, *lines])


def _example_for_reason(gaps: list[dict[str, Any]], reason: str) -> str:
    for gap in gaps:
        if gap.get("reason") == reason:
            text = str(gap.get("user_text") or "").strip()
            if text:
                return _snippet(text)
    return "-"


def _label_for_reason(reason: str) -> str:
    labels = {
        "unknown_intent": "unknown intent",
        "missing_slots": "missing slots",
        "no_tool": "no tool",
        "policy_denied": "policy denied",
    }
    return labels.get(reason, reason)


def _snippet(text: str, limit: int = 80) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."
