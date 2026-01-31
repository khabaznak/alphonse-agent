from __future__ import annotations

import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def list_persons(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM persons ORDER BY person_id ASC LIMIT ?", (limit,))


def get_person(person_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM persons WHERE person_id = ?", (person_id,))


def create_person(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO persons (person_id, display_name, relationship, timezone, is_active)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            payload.get("person_id"),
            payload.get("display_name"),
            payload.get("relationship"),
            payload.get("timezone"),
            _bool(payload.get("is_active", True)),
        ),
    )
    return payload


def update_person(person_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    _execute(
        """
        UPDATE persons
        SET display_name = ?, relationship = ?, timezone = ?, is_active = ?
        WHERE person_id = ?
        """,
        (
            payload.get("display_name"),
            payload.get("relationship"),
            payload.get("timezone"),
            _bool(payload.get("is_active", True)),
            person_id,
        ),
    )
    return get_person(person_id)


def delete_person(person_id: str) -> dict[str, Any] | None:
    return _delete("persons", "person_id", person_id)


def list_groups(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM groups ORDER BY group_id ASC LIMIT ?", (limit,))


def get_group(group_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM groups WHERE group_id = ?", (group_id,))


def create_group(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO groups (group_id, name, is_active)
        VALUES (?, ?, ?)
        """,
        (
            payload.get("group_id"),
            payload.get("name"),
            _bool(payload.get("is_active", True)),
        ),
    )
    return payload


def update_group(group_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    _execute(
        """
        UPDATE groups
        SET name = ?, is_active = ?
        WHERE group_id = ?
        """,
        (
            payload.get("name"),
            _bool(payload.get("is_active", True)),
            group_id,
        ),
    )
    return get_group(group_id)


def delete_group(group_id: str) -> dict[str, Any] | None:
    return _delete("groups", "group_id", group_id)


def list_channels(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM channels ORDER BY channel_type ASC, priority ASC LIMIT ?", (limit,))


def get_channel(channel_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM channels WHERE channel_id = ?", (channel_id,))


def create_channel(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO channels (channel_id, channel_type, person_id, address, is_enabled, priority)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("channel_id"),
            payload.get("channel_type"),
            payload.get("person_id"),
            payload.get("address"),
            _bool(payload.get("is_enabled", True)),
            payload.get("priority", 100),
        ),
    )
    return payload


def update_channel(channel_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    _execute(
        """
        UPDATE channels
        SET channel_type = ?, person_id = ?, address = ?, is_enabled = ?, priority = ?
        WHERE channel_id = ?
        """,
        (
            payload.get("channel_type"),
            payload.get("person_id"),
            payload.get("address"),
            _bool(payload.get("is_enabled", True)),
            payload.get("priority", 100),
            channel_id,
        ),
    )
    return get_channel(channel_id)


def delete_channel(channel_id: str) -> dict[str, Any] | None:
    return _delete("channels", "channel_id", channel_id)


def list_prefs(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM communication_prefs ORDER BY prefs_id ASC LIMIT ?", (limit,))


def get_prefs(prefs_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM communication_prefs WHERE prefs_id = ?", (prefs_id,))


def create_prefs(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO communication_prefs
          (prefs_id, scope_type, scope_id, language_preference, tone, formality, emoji, verbosity_cap,
           quiet_hours_start, quiet_hours_end, allow_push, allow_telegram, allow_web, allow_cli, model_budget_policy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("prefs_id"),
            payload.get("scope_type"),
            payload.get("scope_id"),
            payload.get("language_preference"),
            payload.get("tone"),
            payload.get("formality"),
            payload.get("emoji"),
            payload.get("verbosity_cap"),
            payload.get("quiet_hours_start"),
            payload.get("quiet_hours_end"),
            _bool(payload.get("allow_push", True)),
            _bool(payload.get("allow_telegram", True)),
            _bool(payload.get("allow_web", True)),
            _bool(payload.get("allow_cli", True)),
            payload.get("model_budget_policy"),
        ),
    )
    return payload


def update_prefs(prefs_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    _execute(
        """
        UPDATE communication_prefs
        SET scope_type = ?, scope_id = ?, language_preference = ?, tone = ?, formality = ?, emoji = ?,
            verbosity_cap = ?, quiet_hours_start = ?, quiet_hours_end = ?, allow_push = ?, allow_telegram = ?,
            allow_web = ?, allow_cli = ?, model_budget_policy = ?
        WHERE prefs_id = ?
        """,
        (
            payload.get("scope_type"),
            payload.get("scope_id"),
            payload.get("language_preference"),
            payload.get("tone"),
            payload.get("formality"),
            payload.get("emoji"),
            payload.get("verbosity_cap"),
            payload.get("quiet_hours_start"),
            payload.get("quiet_hours_end"),
            _bool(payload.get("allow_push", True)),
            _bool(payload.get("allow_telegram", True)),
            _bool(payload.get("allow_web", True)),
            _bool(payload.get("allow_cli", True)),
            payload.get("model_budget_policy"),
            prefs_id,
        ),
    )
    return get_prefs(prefs_id)


def delete_prefs(prefs_id: str) -> dict[str, Any] | None:
    return _delete("communication_prefs", "prefs_id", prefs_id)


def list_presence(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM presence_state ORDER BY updated_at DESC LIMIT ?", (limit,))


def get_presence(person_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM presence_state WHERE person_id = ?", (person_id,))


def upsert_presence(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO presence_state (person_id, in_meeting, location_hint, updated_at)
        VALUES (?, ?, ?, datetime('now'))
        ON CONFLICT(person_id) DO UPDATE SET
          in_meeting = excluded.in_meeting,
          location_hint = excluded.location_hint,
          updated_at = datetime('now')
        """,
        (
            payload.get("person_id"),
            _bool(payload.get("in_meeting", False)),
            payload.get("location_hint"),
        ),
    )
    return get_presence(payload.get("person_id")) or {}


def _fetch_all(query: str, params: tuple) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]


def _fetch_one(query: str, params: tuple) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(query, params).fetchone()
    return dict(row) if row else None


def _execute(query: str, params: tuple) -> None:
    with _connect() as conn:
        conn.execute(query, params)
        conn.commit()


def _delete(table: str, key: str, record_id: str) -> dict[str, Any] | None:
    record = _fetch_one(f"SELECT * FROM {table} WHERE {key} = ?", (record_id,))
    if not record:
        return None
    _execute(f"DELETE FROM {table} WHERE {key} = ?", (record_id,))
    return record


def _connect() -> sqlite3.Connection:
    path = resolve_nervous_system_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _bool(value: object) -> int:
    if isinstance(value, str):
        return 1 if value.lower() in {"1", "true", "yes", "on"} else 0
    return 1 if bool(value) else 0
