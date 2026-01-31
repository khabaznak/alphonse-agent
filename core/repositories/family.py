from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any


_FAMILY: dict[str, dict[str, Any]] = {}


def list_family_members(limit: int = 200) -> list[dict[str, Any]]:
    return list(_FAMILY.values())[:limit]


def get_family_member(member_id: str) -> dict[str, Any] | None:
    return _FAMILY.get(member_id)


def create_family_member(payload: dict[str, Any]) -> dict[str, Any]:
    member_id = str(uuid.uuid4())
    record = {
        "id": member_id,
        "name": payload.get("name"),
        "role": payload.get("role", "member"),
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
    }
    _FAMILY[member_id] = record
    return record


def update_family_member(member_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    record = _FAMILY.get(member_id)
    if not record:
        return None
    record.update(payload)
    record["updated_at"] = _timestamp()
    return record


def _timestamp() -> str:
    return datetime.now().isoformat()
