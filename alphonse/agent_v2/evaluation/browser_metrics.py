"""Privacy-preserving summary of already-recorded browser workflow telemetry.

This is an offline projection, not a replacement for instrumenting missing spans.
Only fixed categories and numeric durations cross the projection boundary.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

_INFERENCE = {"phase_planning", "tactical_action"}
_BROWSER = {
    "begin", "reacquire", "observe", "observe_result", "request_value", "fill",
    "submit", "reconcile", "verify_submission",
}


def summarize(events: Iterable[dict[str, Any]], *, task_id: str) -> dict[str, Any]:
    """Project one task's measured spans; never copy payloads, identifiers or errors.

    Router timings cover the whole provider adapter invocation, including local
    adapter work. Tool timings cover the invocation boundary, including validation
    and local work. Neither is an exact remote provider/browser execution time.
    Input events may contain sensitive data and must themselves remain protected.
    """
    if not isinstance(task_id, str) or not task_id:
        raise ValueError("task_id_required")
    totals: dict[str, dict[str, int]] = defaultdict(lambda: {"count": 0, "failed": 0, "elapsed_ms": 0})
    for event in events:
        if not isinstance(event, dict) or event.get("task_id") != task_id:
            continue
        event_type = event.get("event_type")
        category = None
        if event_type == "inference" and isinstance(event.get("purpose"), str) and event["purpose"] in _INFERENCE:
            category = "inference." + event["purpose"]
        elif event_type == "v3_measurement" and event.get("operation") in ("phase_planning", "tactical_composition"):
            category = "orchestration." + event["operation"]
        elif event_type == "tool" and isinstance(event.get("tool_id"), str):
            prefix = "native.browser_workflow_"
            tool_id = event["tool_id"]
            operation = tool_id[len(prefix):] if tool_id.startswith(prefix) else ""
            if operation in _BROWSER:
                category = "browser." + operation
        duration = event.get("duration_ms")
        status = event.get("status")
        valid_status = {"returned", "raised"} if event_type == "v3_measurement" else {"success", "failed"}
        if category is None or type(duration) is not int or not 0 <= duration <= 86_400_000 or not isinstance(status, str) or status not in valid_status:
            continue
        row = totals[category]
        row["count"] += 1
        row["failed"] += int(status in {"failed", "raised"})
        row["elapsed_ms"] += duration
    return {"schema_version": 1, "spans": dict(sorted(totals.items()))}
