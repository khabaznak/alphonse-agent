from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone

from alphonse.agent.identity import users as users_store
from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.services import get_service_by_key
from alphonse.agent.nervous_system.services import TELEGRAM_SERVICE_ID


def resolve_service_user_id(*, user_id: str, service_id: int) -> str | None:
    value = str(user_id or "").strip()
    if not value:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT service_user_id
            FROM user_service_resolvers
            WHERE user_id = ? AND service_id = ? AND is_active = 1
            LIMIT 1
            """,
            (value, int(service_id)),
        ).fetchone()
    return str(row[0]) if row and row[0] is not None else None

# This function is correct but it has a little tech debt as it only considers 1 id per service_id
def resolve_user_id_by_service_user_id(*, service_id: int, service_user_id: str) -> str | None:
    value = str(service_user_id or "").strip()
    if not value:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT user_id
            FROM user_service_resolvers
            WHERE service_id = ? AND service_user_id = ? AND is_active = 1
            LIMIT 1
            """,
            (int(service_id), value),
        ).fetchone()
    return str(row[0]) if row and row[0] is not None else None

def resolve_service_id_by_channel_type(channel_type: str | None) -> int | None:
    """Resolve a channel type using services.service_key in the current schema."""
    rendered = str(channel_type or "").strip().lower()
    if not rendered:
        return None
    service = get_service_by_key(rendered)
    if not isinstance(service, dict):
        return None
    value = service.get("service_id")
    return int(value) if value is not None else None


def resolve_service_key_by_service_user_id(service_user_id: str) -> str | None:
    value = str(service_user_id or "").strip()
    if not value:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT s.service_key
            FROM user_service_resolvers r
            JOIN services s ON s.service_id = r.service_id
            WHERE r.service_user_id = ? AND r.is_active = 1
            ORDER BY r.updated_at DESC
            LIMIT 1
            """,
            (value,),
        ).fetchone()
    # TODO: service_user_id is not globally unique; choose an explicit conflict policy
    # instead of returning the most recently updated resolver.
    return str(row[0]).strip().lower() if row and row[0] is not None else None


def upsert_service_resolver(
    *,
    user_id: str,
    service_id: int,
    service_user_id: str,
    is_active: bool = True,
) -> str:
    normalized_user = str(user_id or "").strip()
    normalized_service_user = str(service_user_id or "").strip()
    if not normalized_user:
        raise ValueError("user_id is required")
    if not normalized_service_user:
        raise ValueError("service_user_id is required")
    now = _now_iso()
    resolver_id = str(uuid.uuid4())
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO user_service_resolvers (
              resolver_id, user_id, service_id, service_user_id, is_active, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, service_id) DO UPDATE SET
              service_user_id = excluded.service_user_id,
              is_active = excluded.is_active,
              updated_at = excluded.updated_at
            """,
            (
                resolver_id,
                normalized_user,
                int(service_id),
                normalized_service_user,
                1 if is_active else 0,
                now,
                now,
            ),
        )
        conn.commit()
    return resolver_id


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
