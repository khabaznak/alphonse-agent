from __future__ import annotations

from datetime import datetime, timezone
from alphonse.agent.observability.log_manager import get_component_logger
from typing import Any, Callable

from alphonse.agent.actions.session_context import IncomingContext

logger = get_component_logger("actions.transitions")


def emit_agent_transitions_from_meta(
    *,
    incoming: IncomingContext,
    meta: dict[str, Any],
    emit_presence_event: Callable[[IncomingContext, dict[str, Any]], None],
    skip_phases: set[str] | None = None,
) -> None:
    phases_to_skip = {str(item).lower() for item in (skip_phases or set())}
    events = meta.get("events")
    if not isinstance(events, list):
        return
    for event in events:
        if not isinstance(event, dict):
            continue
        if str(event.get("type") or "") != "agent.transition":
            continue
        phase = str(event.get("phase") or "").strip()
        if not phase:
            continue
        if phase.lower() in phases_to_skip:
            continue
        emit_presence_event(incoming, event)


def chat_action_for_phase(phase: str) -> str | None:
    mapped = {
        "acknowledged": "typing",
        "thinking": "typing",
        "executing": "typing",
    }
    return mapped.get(str(phase or "").strip().lower())


def reaction_for_phase(phase: str) -> str | None:
    mapped = {
        "acknowledged": "👀",
        "thinking": "🤔",
        "executing": "🤔",
        "waiting_user": "❓",
        "done": "👍",
        "failed": "👎",
    }
    return mapped.get(str(phase or "").strip().lower())


def phase_from_transition_event(event: dict[str, Any]) -> str | None:
    phase = str(event.get("phase") or "").strip().lower()
    if phase in {"acknowledged", "thinking", "executing", "waiting_user", "done", "failed", "wip_update"}:
        return phase
    if phase != "cortex.state":
        return None
    detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
    stage = str(detail.get("stage") or "").strip().lower()
    node = str(detail.get("node") or "").strip().lower()
    has_pending = bool(detail.get("has_pending_interaction"))
    if stage == "start":
        if node in {"next_step_node", "progress_critic_node", "act_node", "apology_node"}:
            return "thinking"
        if node == "respond_node":
            return "executing"
    if stage == "done" and has_pending:
        return "waiting_user"
    return None


def presence_event_from_transition_event(event: dict[str, Any]) -> dict[str, Any] | None:
    phase = phase_from_transition_event(event)
    if not phase:
        return None
    detail = event.get("detail") if isinstance(event.get("detail"), dict) else {}
    family = _presence_family_for_phase(phase)
    emitted_at = str(event.get("at") or "").strip() or datetime.now(timezone.utc).isoformat()
    hint = _presence_hint(detail)
    tool_name = _presence_tool_name(detail)
    payload: dict[str, Any] = {
        "event_family": family,
        "correlation_id": str(event.get("correlation_id") or "").strip() or None,
        "ts": emitted_at,
        "phase": "thinking" if phase == "wip_update" else phase,
    }
    if hint:
        payload["hint"] = hint
    if tool_name:
        payload["tool_name"] = tool_name
    return payload


def projectable_phase_for_presence_event(presence_event: dict[str, Any]) -> str | None:
    phase = str(presence_event.get("phase") or "").strip().lower()
    if phase in {"acknowledged", "thinking", "executing", "waiting_user", "done", "failed"}:
        return phase
    return None


def _presence_family_for_phase(phase: str) -> str:
    phase_value = str(phase or "").strip().lower()
    if phase_value == "wip_update":
        return "presence.progress"
    if phase_value == "waiting_user":
        return "presence.waiting_input"
    if phase_value == "done":
        return "presence.completed"
    if phase_value == "failed":
        return "presence.failed"
    return "presence.phase_changed"


def _presence_hint(detail: dict[str, Any]) -> str | None:
    for key in ("hint", "planner_intent", "text", "reason"):
        value = str(detail.get(key) or "").strip()
        if value:
            return value[:160]
    return None


def _presence_tool_name(detail: dict[str, Any]) -> str | None:
    for key in ("tool_name", "tool"):
        value = str(detail.get(key) or "").strip()
        if value:
            return value
    return None
