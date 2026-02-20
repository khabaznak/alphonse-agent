from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.observability.store import write_task_event


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
    payload: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "level": str(level).lower(),
        "event": event,
        "correlation_id": state.get("correlation_id"),
        "channel": state.get("channel_type"),
        "user_id": state.get("actor_person_id"),
        "node": node,
        "cycle": int(task_state.get("cycle_index") or 0),
        "status": str(task_state.get("status") or ""),
    }
    if extra:
        payload.update({k: v for k, v in extra.items() if v is not None})
    line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)
    if str(level).lower() == "debug":
        logger.debug("task_mode_event %s", line)
    elif str(level).lower() == "warning":
        logger.warning("task_mode_event %s", line)
    elif str(level).lower() == "error":
        logger.error("task_mode_event %s", line)
    else:
        logger.info("task_mode_event %s", line)
    try:
        write_task_event(payload)
    except Exception:
        return
