from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from alphonse.config import settings

_SETTINGS: dict[str, dict[str, Any]] = {}


def init_db() -> None:
    return None


def list_settings() -> list[dict[str, Any]]:
    return list(_SETTINGS.values())


def get_setting(setting_id: str) -> dict[str, Any] | None:
    return _SETTINGS.get(setting_id)


def get_setting_by_name(name: str) -> dict[str, Any] | None:
    for setting in _SETTINGS.values():
        if setting.get("name") == name:
            return setting
    return None


def create_setting(payload: dict[str, Any]) -> dict[str, Any]:
    setting_id = str(uuid.uuid4())
    record = {
        "id": setting_id,
        "name": payload.get("name"),
        "config": payload.get("config"),
        "created_at": _timestamp(),
        "updated_at": _timestamp(),
    }
    _SETTINGS[setting_id] = record
    return record


def update_setting(setting_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    record = _SETTINGS.get(setting_id)
    if not record:
        return None
    record.update(payload)
    record["updated_at"] = _timestamp()
    return record


def delete_setting(setting_id: str) -> dict[str, Any] | None:
    return _SETTINGS.pop(setting_id, None)


def get_timezone() -> str:
    setting = get_setting_by_name("timezone")
    if not setting:
        return settings.get_timezone()
    config = setting.get("config")
    if isinstance(config, dict):
        tz_name = config.get("tz")
        if isinstance(tz_name, str) and tz_name:
            return tz_name
    return settings.get_timezone()


def _timestamp() -> str:
    return datetime.now().isoformat()


def _local_timezone() -> str:
    tzinfo = datetime.now().astimezone().tzinfo
    if tzinfo is not None and hasattr(tzinfo, "key"):
        return str(tzinfo.key)
    return settings.get_timezone()
