from __future__ import annotations

from typing import Any

from alphonse.agent.cortex.transitions import emit_transition_event


def emit_brain_state(
    *,
    state: dict[str, Any],
    node: str,
    updates: dict[str, Any] | None = None,
    stage: str = "done",
    error_type: str | None = None,
) -> dict[str, Any]:
    merged = dict(state)
    if isinstance(updates, dict):
        merged.update(updates)
    detail: dict[str, Any] = {
        "node": node,
        "stage": stage,
        "route_decision": merged.get("route_decision"),
        "intent": merged.get("intent"),
        "has_response_text": bool(merged.get("response_text")),
        "has_pending_interaction": bool(merged.get("pending_interaction")),
        "plans_count": len(merged.get("plans") or [])
    }
    if error_type:
        detail["error_type"] = error_type
    emit_transition_event(
        state,
        "cortex.state",
        detail,
    )
    if not isinstance(updates, dict):
        return {"events": state.get("events") or []}
    updates.setdefault("events", state.get("events") or [])
    return updates
