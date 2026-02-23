from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.observability.log_manager import get_log_manager


def log_task_event(
    *,
    logger: logging.Logger,
    state: dict[str, Any],
    task_state: dict[str, Any],
    node: str,
    event: str,
    level: str = "info",
    **extra: Any,
) -> None:
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
        "cycle": int(task_state.get("cycle_index") or 0),
        "status": str(task_state.get("status") or ""),
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
