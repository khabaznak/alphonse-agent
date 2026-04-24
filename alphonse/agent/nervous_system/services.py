from __future__ import annotations

import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

TELEGRAM_CHANNEL_ID = 2
TELEGRAM_SERVICE_ID = TELEGRAM_CHANNEL_ID


def get_channel(channel_id: int) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        try:
            row = conn.execute(
                """
                SELECT channel_id, channel_key, provider, channel_type, raw_user_key_field,
                       name, description, created_at, updated_at
                FROM channels
                WHERE channel_id = ?
                LIMIT 1
                """,
                (int(channel_id),),
            ).fetchone()
        except sqlite3.OperationalError:
            row = conn.execute(
                """
                SELECT service_id, service_key, service_key, 'interactive', raw_user_key_field,
                       name, description, created_at, updated_at
                FROM services
                WHERE service_id = ?
                LIMIT 1
                """,
                (int(channel_id),),
            ).fetchone()
    if not row:
        return None
    return {
        "channel_id": row[0],
        "channel_key": row[1],
        "provider": row[2],
        "channel_type": row[3],
        "raw_user_key_field": row[4],
        "name": row[5],
        "description": row[6],
        "created_at": row[7],
        "updated_at": row[8],
    }


def get_channel_by_key(channel_key: str) -> dict[str, Any] | None:
    key = str(channel_key or "").strip().lower()
    if not key:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        try:
            row = conn.execute(
                """
                SELECT channel_id, channel_key, provider, channel_type, raw_user_key_field,
                       name, description, created_at, updated_at
                FROM channels
                WHERE lower(channel_key) = lower(?)
                LIMIT 1
                """,
                (key,),
            ).fetchone()
        except sqlite3.OperationalError:
            row = conn.execute(
                """
                SELECT service_id, service_key, service_key, 'interactive', raw_user_key_field,
                       name, description, created_at, updated_at
                FROM services
                WHERE lower(service_key) = lower(?)
                LIMIT 1
                """,
                (key,),
            ).fetchone()
    if not row:
        return None
    return {
        "channel_id": row[0],
        "channel_key": row[1],
        "provider": row[2],
        "channel_type": row[3],
        "raw_user_key_field": row[4],
        "name": row[5],
        "description": row[6],
        "created_at": row[7],
        "updated_at": row[8],
    }


def get_service(service_id: int) -> dict[str, Any] | None:
    channel = get_channel(service_id)
    if not channel:
        return None
    return {
        "service_id": channel["channel_id"],
        "service_key": channel["channel_key"],
        "raw_user_key_field": channel["raw_user_key_field"],
        "name": channel["name"],
        "description": channel["description"],
        "created_at": channel["created_at"],
        "updated_at": channel["updated_at"],
    }


def get_service_by_key(service_key: str) -> dict[str, Any] | None:
    channel = get_channel_by_key(service_key)
    if not channel:
        return None
    return {
        "service_id": channel["channel_id"],
        "service_key": channel["channel_key"],
        "raw_user_key_field": channel["raw_user_key_field"],
        "name": channel["name"],
        "description": channel["description"],
        "created_at": channel["created_at"],
        "updated_at": channel["updated_at"],
    }
