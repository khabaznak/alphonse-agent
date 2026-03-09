from __future__ import annotations

from typing import Any

from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("services.cognition_lifecycle")


def flush_cognition_state_if_task_succeeded(
    cognition_state: dict[str, Any],
    *,
    correlation_id: str | None = None,
) -> dict[str, Any]:
    task_state = cognition_state.get("task_state")
    if not isinstance(task_state, dict):
        return dict(cognition_state)
    status = str(task_state.get("status") or "").strip().lower()
    if status != "done" or not task_state.get("outcome"):
        return dict(cognition_state)
    logger.info(
        "flush_after_task_success correlation_id=%s",
        str(correlation_id or ""),
    )
    return {}
