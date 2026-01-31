from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any


_DEVICES: dict[str, dict[str, Any]] = {}


def list_push_devices(limit: int = 200) -> list[dict[str, Any]]:
    return list(_DEVICES.values())[:limit]


def list_active_push_devices(target_group: str | None = None, platforms: list[str] | None = None) -> list[dict[str, Any]]:
    devices = [device for device in _DEVICES.values() if device.get("active", True)]
    if platforms:
        devices = [device for device in devices if device.get("platform") in platforms]
    return devices


def upsert_push_device(payload: dict[str, Any]) -> dict[str, Any]:
    device_id = payload.get("id") or str(uuid.uuid4())
    record = _DEVICES.get(device_id, {})
    record.update(payload)
    record.setdefault("id", device_id)
    record.setdefault("created_at", _timestamp())
    record["updated_at"] = _timestamp()
    _DEVICES[device_id] = record
    return record


def deactivate_push_device(device_id: str) -> dict[str, Any] | None:
    record = _DEVICES.get(device_id)
    if not record:
        return None
    record["active"] = False
    record["updated_at"] = _timestamp()
    return record


def _timestamp() -> str:
    return datetime.now().isoformat()
