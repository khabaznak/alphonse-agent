from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.observability.log_manager import get_log_manager
from alphonse.agent.cortex.task_mode.task_record import TaskRecord


def log_task_event(
    *,
    logger: logging.Logger,
    state: dict[str, Any],
    node: str,
    event: str,
    task_record: TaskRecord | None = None,
    cycle_index: int | None = None,
    status: str | None = None,
    level: str = "info",
    **extra: Any,
) -> None:
    effective_cycle = 0 if cycle_index is None else int(cycle_index)
    effective_status = str(status or (task_record.status if isinstance(task_record, TaskRecord) else "") or "")
    manager = get_log_manager()
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": str(level).lower(),
        "event": event,
        "component": "task_mode",
        "correlation_id": state.get("correlation_id"),
        "channel": state.get("channel_type"),
        "user_id": state.get("actor_person_id"),
        "node": node,
        "cycle": effective_cycle,
        "status": effective_status,
    }
    if extra:
        payload.update({k: v for k, v in extra.items() if v is not None})
    message = f"task_mode_event {json.dumps(payload, ensure_ascii=False, separators=(',', ':'), default=str)}"
    manager.emit(
        level=str(level).lower(),
        event=event,
        message=message,
        component="task_mode",
        correlation_id=_as_text_or_none(payload.get("correlation_id")),
        channel=_as_text_or_none(payload.get("channel")),
        user_id=_as_text_or_none(payload.get("user_id")),
        node=_as_text_or_none(payload.get("node")),
        cycle=_as_int_or_none(payload.get("cycle")),
        status=_as_text_or_none(payload.get("status")),
        tool=_as_text_or_none(payload.get("tool")),
        error_code=_as_text_or_none(payload.get("error_code")),
        latency_ms=_as_int_or_none(payload.get("latency_ms")),
        payload=payload,
    )


def _as_text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
