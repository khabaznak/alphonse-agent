from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def emit_transition_event(
    state: dict[str, Any],
    phase: str,
    detail: dict[str, Any] | None = None,
) -> None:
    events = state.get("events")
    if not isinstance(events, list):
        events = []
        state["events"] = events
    event: dict[str, Any] = {
        "type": "agent.transition",
        "phase": str(phase),
        "at": datetime.now(timezone.utc).isoformat(),
        "correlation_id": state.get("correlation_id"),
    }
    if isinstance(detail, dict) and detail:
        event["detail"] = detail
    if events:
        last = events[-1]
        if (
            isinstance(last, dict)
            and last.get("type") == "agent.transition"
            and last.get("phase") == phase
            and (last.get("detail") or {}) == (event.get("detail") or {})
        ):
            return
    events.append(event)
    sink = state.get("_transition_sink")
    if callable(sink):
        try:
            sink(event)
        except Exception:
            return


def emit_presence_transition_event(
    state: dict[str, Any],
    *,
    event_family: str,
    phase: str,
    detail: dict[str, Any] | None = None,
) -> None:
    merged_detail = dict(detail) if isinstance(detail, dict) else {}
    merged_detail["presence_event_family"] = str(event_family or "").strip()
    emit_transition_event(state, str(phase or "").strip(), merged_detail)
