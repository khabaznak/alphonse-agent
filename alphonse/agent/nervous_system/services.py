from __future__ import annotations

import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

TELEGRAM_SERVICE_ID = 2


def get_service(service_id: int) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT service_id, service_key, raw_user_key_field, name, description, created_at, updated_at
            FROM services
            WHERE service_id = ?
            LIMIT 1
            """,
            (int(service_id),),
        ).fetchone()
    if not row:
        return None
    return {
        "service_id": row[0],
        "service_key": row[1],
        "raw_user_key_field": row[2],
        "name": row[3],
        "description": row[4],
        "created_at": row[5],
        "updated_at": row[6],
    }


def get_service_by_key(service_key: str) -> dict[str, Any] | None:
    key = str(service_key or "").strip().lower()
    if not key:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT service_id, service_key, raw_user_key_field, name, description, created_at, updated_at
            FROM services
            WHERE lower(service_key) = lower(?)
            LIMIT 1
            """,
            (key,),
        ).fetchone()
    if not row:
        return None
    return {
        "service_id": row[0],
        "service_key": row[1],
        "raw_user_key_field": row[2],
        "name": row[3],
        "description": row[4],
        "created_at": row[5],
        "updated_at": row[6],
    }

