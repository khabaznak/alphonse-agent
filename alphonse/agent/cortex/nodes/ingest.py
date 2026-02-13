from __future__ import annotations

import json
import logging
from typing import Any

from alphonse.agent.cortex.transitions import emit_transition_event

logger = logging.getLogger(__name__)


def ingest_node(state: dict[str, Any]) -> dict[str, Any]:
    message = state.get("last_user_message")
    if not _is_payload_cannonical(message):
        error = ValueError("last_user_message is not cannonical JSON payload")
        logger.exception(
            "cortex ingest invalid payload chat_id=%s payload=%s error=%s",
            state.get("chat_id"),
            _snippet(str(message)),
            error,
        )
        raise error
    logger.info(
        "cortex ingest chat_id=%s text=%s",
        state.get("chat_id"),
        _snippet(str(message)),
    )
    emit_transition_event(state, "acknowledged")
    return {
        "last_user_message": message,
        "events": state.get("events") or [],
    }


def _is_payload_cannonical(payload: Any) -> bool:
    try:
        if isinstance(payload, str):
            return bool(payload.strip())
        json.dumps(payload, ensure_ascii=False)
        return True
    except Exception:
        return False


def _snippet(text: str, limit: int = 140) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."
