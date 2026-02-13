from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any


def build_cognition_state(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "slots_collected": state.get("slots") or {},
        "last_intent": state.get("intent"),
        "locale": state.get("locale"),
        "autonomy_level": state.get("autonomy_level"),
        "planning_mode": state.get("planning_mode"),
        "intent_category": state.get("intent_category"),
        "route_decision": state.get("route_decision"),
        "pending_interaction": state.get("pending_interaction"),
        "ability_state": state.get("ability_state"),
        "planning_context": state.get("planning_context"),
        "last_updated_at": datetime.now(timezone.utc).isoformat(),
    }


def build_meta(state: dict[str, Any]) -> dict[str, Any]:
    return {
        "intent": state.get("intent"),
        "intent_confidence": state.get("intent_confidence"),
        "correlation_id": state.get("correlation_id"),
        "chat_id": state.get("chat_id"),
        "response_key": state.get("response_key"),
        "response_vars": state.get("response_vars"),
        "autonomy_level": state.get("autonomy_level"),
        "planning_mode": state.get("planning_mode"),
        "intent_category": state.get("intent_category"),
        "route_decision": state.get("route_decision"),
        "events": state.get("events") or [],
    }


def safe_json(value: Any, limit: int = 1200) -> str:
    try:
        rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except Exception:
        rendered = str(value)
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[:limit]}..."
