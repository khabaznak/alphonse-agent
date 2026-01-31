from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any


_EVENTS: dict[str, dict[str, Any]] = {}


def list_family_events(limit: int = 200) -> list[dict[str, Any]]:
    return list(_EVENTS.values())[:limit]


def get_family_event(event_id: str) -> dict[str, Any] | None:
    return _EVENTS.get(event_id)


def create_family_event(payload: dict[str, Any]) -> dict[str, Any]:
    event_id = str(uuid.uuid4())
    record = {"id": event_id, **payload}
    record.setdefault("created_at", _timestamp())
    record.setdefault("updated_at", _timestamp())
    _EVENTS[event_id] = record
    return record


def update_family_event(event_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    record = _EVENTS.get(event_id)
    if not record:
        return None
    record.update(payload)
    record["updated_at"] = _timestamp()
    return record


def delete_family_event(event_id: str) -> dict[str, Any] | None:
    return _EVENTS.pop(event_id, None)


def _timestamp() -> str:
    return datetime.now().isoformat()
