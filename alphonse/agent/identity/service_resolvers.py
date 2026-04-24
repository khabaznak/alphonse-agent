from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.services import TELEGRAM_CHANNEL_ID
from alphonse.agent.nervous_system.services import get_channel_by_key


def resolve_channel_user_id(*, user_id: str, channel_id: int) -> str | None:
    value = str(user_id or "").strip()
    if not value:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT channel_user_id
            FROM channels_users
            WHERE user_id = ? AND channel_id = ? AND is_active = 1
            LIMIT 1
            """,
            (value, int(channel_id)),
        ).fetchone()
    return str(row[0]) if row and row[0] is not None else None


def resolve_user_id_by_channel_user_id(*, channel_id: int, channel_user_id: str) -> str | None:
    value = str(channel_user_id or "").strip()
    if not value:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT user_id
            FROM channels_users
            WHERE channel_id = ? AND channel_user_id = ? AND is_active = 1
            LIMIT 1
            """,
            (int(channel_id), value),
        ).fetchone()
    return str(row[0]) if row and row[0] is not None else None


def resolve_channel_id_by_channel_type(channel_type: str | None) -> int | None:
    rendered = str(channel_type or "").strip().lower()
    if not rendered:
        return None
    channel = get_channel_by_key(rendered)
    if not isinstance(channel, dict):
        return None
    value = channel.get("channel_id")
    return int(value) if value is not None else None


def resolve_channel_key_by_channel_user_id(channel_user_id: str) -> str | None:
    value = str(channel_user_id or "").strip()
    if not value:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT c.channel_key
            FROM channels_users cu
            JOIN channels c ON c.channel_id = cu.channel_id
            WHERE cu.channel_user_id = ? AND cu.is_active = 1
            ORDER BY cu.updated_at DESC
            LIMIT 1
            """,
            (value,),
        ).fetchone()
    return str(row[0]).strip().lower() if row and row[0] is not None else None


def upsert_channel_mapping(
    *,
    user_id: str,
    channel_id: int,
    channel_user_id: str,
    is_active: bool = True,
) -> str:
    normalized_user = str(user_id or "").strip()
    normalized_channel_user = str(channel_user_id or "").strip()
    if not normalized_user:
        raise ValueError("user_id is required")
    if not normalized_channel_user:
        raise ValueError("channel_user_id is required")
    now = _now_iso()
    mapping_id = str(uuid.uuid4())
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO channels_users (
              mapping_id, user_id, channel_id, channel_user_id, is_active, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, channel_id) DO UPDATE SET
              channel_user_id = excluded.channel_user_id,
              is_active = excluded.is_active,
              updated_at = excluded.updated_at
            """,
            (
                mapping_id,
                normalized_user,
                int(channel_id),
                normalized_channel_user,
                1 if is_active else 0,
                now,
                now,
            ),
        )
        conn.commit()
    return mapping_id


def resolve_service_user_id(*, user_id: str, service_id: int) -> str | None:
    return resolve_channel_user_id(user_id=user_id, channel_id=service_id)


def resolve_user_id_by_service_user_id(*, service_id: int, service_user_id: str) -> str | None:
    return resolve_user_id_by_channel_user_id(channel_id=service_id, channel_user_id=service_user_id)


def resolve_service_id_by_channel_type(channel_type: str | None) -> int | None:
    return resolve_channel_id_by_channel_type(channel_type)


def resolve_service_key_by_service_user_id(service_user_id: str) -> str | None:
    return resolve_channel_key_by_channel_user_id(service_user_id)


def upsert_service_resolver(
    *,
    user_id: str,
    service_id: int,
    service_user_id: str,
    is_active: bool = True,
) -> str:
    return upsert_channel_mapping(
        user_id=user_id,
        channel_id=service_id,
        channel_user_id=service_user_id,
        is_active=is_active,
    )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
