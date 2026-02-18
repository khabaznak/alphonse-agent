from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.services import TELEGRAM_SERVICE_ID
from alphonse.agent.nervous_system import users as users_store


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


def resolve_telegram_chat_id_for_user(user_ref: str) -> str | None:
    candidate = str(user_ref or "").strip()
    if not candidate:
        return None
    if _is_numeric_identifier(candidate):
        return candidate

    # First treat ref as internal Alphonse user_id.
    mapped = resolve_service_user_id(user_id=candidate, service_id=TELEGRAM_SERVICE_ID)
    if mapped:
        return mapped

    # Then try ref as display name -> internal user_id -> resolver table.
    matched = users_store.get_user_by_display_name(candidate)
    if isinstance(matched, dict):
        internal_user_id = str(matched.get("user_id") or "").strip()
        if internal_user_id:
            mapped = resolve_service_user_id(user_id=internal_user_id, service_id=TELEGRAM_SERVICE_ID)
            if mapped:
                return mapped
    return None


def resolve_internal_user_by_telegram_id(telegram_user_id: str) -> str | None:
    return resolve_user_id_by_service_user_id(
        service_id=TELEGRAM_SERVICE_ID,
        service_user_id=str(telegram_user_id or "").strip(),
    )


def _is_numeric_identifier(value: str) -> bool:
    rendered = str(value or "").strip()
    if not rendered:
        return False
    if rendered.startswith("-"):
        return rendered[1:].isdigit()
    return rendered.isdigit()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
