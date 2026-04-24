from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

logger = get_component_logger("cognition.preferences.store")


def get_user_preference(user_id: str, preference_name: str) -> Any | None:
    normalized_user_id = str(user_id or "").strip()
    normalized_name = str(preference_name or "").strip()
    if not normalized_user_id or not normalized_name:
        return None
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT up.value_json
            FROM user_preferences up
            JOIN preferences p ON p.preference_id = up.preference_id
            WHERE up.user_id = ? AND p.name = ?
            LIMIT 1
            """,
            (normalized_user_id, normalized_name),
        ).fetchone()
    if not row:
        _log_preference_read(normalized_user_id, normalized_name, None)
        return None
    try:
        parsed = json.loads(row[0])
    except (TypeError, json.JSONDecodeError):
        parsed = None
    _log_preference_read(normalized_user_id, normalized_name, parsed)
    return parsed


def set_user_preference(
    user_id: str,
    preference_name: str,
    value: Any,
    source: str = "user",
    *,
    description: str | None = None,
    value_kind: str = "json",
) -> None:
    normalized_user_id = str(user_id or "").strip()
    normalized_name = str(preference_name or "").strip()
    if not normalized_user_id or not normalized_name:
        return None
    db_path = resolve_nervous_system_db_path()
    now = _timestamp()
    value_json = json.dumps(value)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT preference_id FROM preferences WHERE name = ? LIMIT 1",
            (normalized_name,),
        ).fetchone()
        preference_id = str(row[0]) if row and row[0] is not None else str(uuid.uuid4())
        if not row:
            conn.execute(
                """
                INSERT INTO preferences (
                  preference_id, name, description, value_kind, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    preference_id,
                    normalized_name,
                    description,
                    value_kind,
                    now,
                    now,
                ),
            )
        conn.execute(
            """
            INSERT INTO user_preferences (
              user_preference_id, preference_id, user_id, value_json, source, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, preference_id)
            DO UPDATE SET
              value_json = excluded.value_json,
              source = excluded.source,
              updated_at = excluded.updated_at
            """,
            (
                str(uuid.uuid4()),
                preference_id,
                normalized_user_id,
                value_json,
                source,
                now,
                now,
            ),
        )
    _log_preference_write(normalized_user_id, normalized_name, value, source)


def delete_user_preference(user_id: str, preference_name: str) -> bool:
    normalized_user_id = str(user_id or "").strip()
    normalized_name = str(preference_name or "").strip()
    if not normalized_user_id or not normalized_name:
        return False
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            """
            DELETE FROM user_preferences
            WHERE user_id = ?
              AND preference_id IN (
                SELECT preference_id FROM preferences WHERE name = ?
              )
            """,
            (normalized_user_id, normalized_name),
        )
        conn.commit()
    return cur.rowcount > 0


def get_with_fallback(user_id: str | None, preference_name: str, default: Any) -> Any:
    if not user_id:
        return default
    value = get_user_preference(user_id, preference_name)
    return default if value is None else value


def get_preference(user_id: str, preference_name: str) -> Any | None:
    return get_user_preference(user_id, preference_name)


def set_preference(user_id: str, preference_name: str, value: Any, source: str = "user") -> None:
    set_user_preference(user_id, preference_name, value, source=source)


def delete_preference(user_id: str, preference_name: str) -> bool:
    return delete_user_preference(user_id, preference_name)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_preference_read(user_id: str, key: str, value: Any | None) -> None:
    logger.info(
        "preferences read user_id=%s key=%s value=%s",
        user_id,
        key,
        value,
    )


def _log_preference_write(user_id: str, key: str, value: Any, source: str) -> None:
    logger.info(
        "preferences write user_id=%s key=%s source=%s value=%s",
        user_id,
        key,
        source,
        value,
    )
