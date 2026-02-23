from __future__ import annotations

from alphonse.agent.observability.log_manager import get_component_logger
from typing import Any, Callable

from alphonse.agent.actions.session_context import IncomingContext

logger = get_component_logger("actions.transitions")


def emit_agent_transitions_from_meta(
    *,
    incoming: IncomingContext,
    meta: dict[str, Any],
    emit_transition: Callable[[IncomingContext, str], None],
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
        emit_transition(incoming, phase)
