from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

logger = logging.getLogger(__name__)


def get_or_create_principal_for_channel(
    channel_type: str, channel_id: str
) -> str | None:
    if not channel_type or not channel_id:
        return None
    db_path = resolve_nervous_system_db_path()
    now = _timestamp()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT principal_id
            FROM principals
            WHERE channel_type = ? AND channel_id = ?
            """,
            (channel_type, channel_id),
        ).fetchone()
        if row:
            return str(row[0])
        principal_id = str(uuid.uuid4())
        conn.execute(
            """
            INSERT INTO principals (
              principal_id, principal_type, channel_type, channel_id, created_at, updated_at
            ) VALUES (?, 'channel_chat', ?, ?, ?, ?)
            """,
            (principal_id, channel_type, channel_id, now, now),
        )
        return principal_id


def get_preference(principal_id: str, key: str) -> Any | None:
    if not principal_id or not key:
        return None
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT value_json
            FROM preferences
            WHERE principal_id = ? AND key = ?
            """,
            (principal_id, key),
        ).fetchone()
    if not row:
        _log_preference_read(principal_id, key, None)
        return None
    value_json = row[0]
    try:
        parsed = json.loads(value_json)
    except (TypeError, json.JSONDecodeError):
        parsed = None
    _log_preference_read(principal_id, key, parsed)
    return parsed


def set_preference(
    principal_id: str, key: str, value: Any, source: str = "user"
) -> None:
    if not principal_id or not key:
        return None
    db_path = resolve_nervous_system_db_path()
    now = _timestamp()
    value_json = json.dumps(value)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO preferences (
              preference_id, principal_id, key, value_json, source, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(principal_id, key)
            DO UPDATE SET
              value_json = excluded.value_json,
              source = excluded.source,
              updated_at = excluded.updated_at
            """,
            (str(uuid.uuid4()), principal_id, key, value_json, source, now, now),
        )
    _log_preference_write(principal_id, key, value, source)


def get_with_fallback(principal_id: str, key: str, default: Any) -> Any:
    value = get_preference(principal_id, key)
    return default if value is None else value


def list_principals_with_preference(
    key: str, value: Any | None = None
) -> list[dict[str, str]]:
    if not key:
        return []
    db_path = resolve_nervous_system_db_path()
    params: list[Any] = [key]
    value_clause = ""
    if value is not None:
        value_clause = "AND preferences.value_json = ?"
        params.append(json.dumps(value))
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT principals.principal_id, principals.principal_type,
                   principals.channel_type, principals.channel_id
            FROM preferences
            JOIN principals ON principals.principal_id = preferences.principal_id
            WHERE preferences.key = ? {value_clause}
            """,
            tuple(params),
        ).fetchall()
    return [
        {
            "principal_id": str(row[0]),
            "principal_type": str(row[1]),
            "channel_type": str(row[2]) if row[2] is not None else "",
            "channel_id": str(row[3]) if row[3] is not None else "",
        }
        for row in rows
    ]


def get_or_create_principal_for_conversation(conversation_key: str) -> str | None:
    if not conversation_key:
        return None
    return get_or_create_principal_for_channel("conversation", conversation_key)


def set_preference_for_conversation(conversation_key: str, key: str, value: Any, source: str = "user") -> None:
    principal_id = get_or_create_principal_for_conversation(conversation_key)
    if principal_id:
        set_preference(principal_id, key, value, source=source)


def get_preference_for_conversation(conversation_key: str, key: str) -> Any | None:
    principal_id = get_or_create_principal_for_conversation(conversation_key)
    if not principal_id:
        return None
    return get_preference(principal_id, key)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log_preference_read(principal_id: str, key: str, value: Any | None) -> None:
    principal_type = _principal_type(principal_id)
    logger.info(
        "preferences read principal_type=%s principal_id=%s key=%s value=%s",
        principal_type,
        principal_id,
        key,
        value,
    )


def _log_preference_write(principal_id: str, key: str, value: Any, source: str) -> None:
    principal_type = _principal_type(principal_id)
    logger.info(
        "preferences write principal_type=%s principal_id=%s key=%s value=%s source=%s",
        principal_type,
        principal_id,
        key,
        value,
        source,
    )


def _principal_type(principal_id: str) -> str:
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT principal_type
            FROM principals
            WHERE principal_id = ?
            """,
            (principal_id,),
        ).fetchone()
    return str(row[0]) if row else "unknown"
