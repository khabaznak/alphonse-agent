from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.nervous_system.services import TELEGRAM_SERVICE_ID
from alphonse.agent.nervous_system.user_service_resolvers import (
    resolve_internal_user_by_telegram_id,
    resolve_service_user_id,
)


@dataclass(frozen=True)
class TelegramInboundAccessDecision:
    allowed: bool
    reason: str
    emit_invite: bool = False
    leave_chat: bool = False
    access: dict[str, Any] | None = None


def upsert_chat_access(record: dict[str, Any]) -> dict[str, Any]:
    chat_id = str(record.get("chat_id") or "").strip()
    if not chat_id:
        raise ValueError("chat_id is required")
    chat_type = _normalize_chat_type(record.get("chat_type"))
    status = _normalize_status(record.get("status"))
    policy = _normalize_policy(record.get("policy"), chat_type=chat_type)
    owner_user_id = str(record.get("owner_user_id") or "").strip() or None
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO telegram_chat_access (
              chat_id, chat_type, status, owner_user_id, policy,
              created_at, updated_at, revoked_at, revoke_reason
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chat_id) DO UPDATE SET
              chat_type = excluded.chat_type,
              status = excluded.status,
              owner_user_id = excluded.owner_user_id,
              policy = excluded.policy,
              updated_at = excluded.updated_at,
              revoked_at = excluded.revoked_at,
              revoke_reason = excluded.revoke_reason
            """,
            (
                chat_id,
                chat_type,
                status,
                owner_user_id,
                policy,
                now,
                now,
                record.get("revoked_at"),
                record.get("revoke_reason"),
            ),
        )
        conn.commit()
    return get_chat_access(chat_id) or {}


def get_chat_access(chat_id: str) -> dict[str, Any] | None:
    normalized = str(chat_id or "").strip()
    if not normalized:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT chat_id, chat_type, status, owner_user_id, policy,
                   created_at, updated_at, revoked_at, revoke_reason
            FROM telegram_chat_access
            WHERE chat_id = ?
            LIMIT 1
            """,
            (normalized,),
        ).fetchone()
    return _row_to_chat_access(row) if row else None


def revoke_chat_access(chat_id: str, reason: str) -> dict[str, Any] | None:
    normalized = str(chat_id or "").strip()
    if not normalized:
        return None
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            UPDATE telegram_chat_access
            SET status = 'revoked',
                revoked_at = ?,
                revoke_reason = ?,
                updated_at = ?
            WHERE chat_id = ?
            """,
            (now, str(reason or "revoked"), now, normalized),
        )
        conn.commit()
    return get_chat_access(normalized)


def provision_from_invite(invite: dict[str, Any], *, status: str) -> dict[str, Any] | None:
    if str(status or "").strip().lower() != "approved":
        return None
    chat_id = str(invite.get("chat_id") or "").strip()
    if not chat_id:
        return None
    raw_chat_type = str(invite.get("chat_type") or "").strip()
    chat_type = _normalize_chat_type(raw_chat_type) if raw_chat_type else _guess_chat_type(chat_id)
    owner_user_id = None
    policy = "registered_private" if chat_type == "private" else "owner_managed_group"
    if chat_type in {"group", "supergroup"}:
        owner_service_user_id = str(invite.get("from_user_id") or "").strip()
        if owner_service_user_id:
            owner_user_id = resolve_internal_user_by_telegram_id(owner_service_user_id)
    return upsert_chat_access(
        {
            "chat_id": chat_id,
            "chat_type": chat_type,
            "status": "active",
            "owner_user_id": owner_user_id,
            "policy": policy,
            "revoked_at": None,
            "revoke_reason": None,
        }
    )


def evaluate_inbound_access(
    *,
    chat_id: str,
    chat_type: str,
    from_user_id: str | None,
) -> TelegramInboundAccessDecision:
    normalized_chat_id = str(chat_id or "").strip()
    normalized_chat_type = _normalize_chat_type(chat_type)
    sender = str(from_user_id or "").strip()
    if not normalized_chat_id:
        return TelegramInboundAccessDecision(allowed=False, reason="missing_chat_id")

    if normalized_chat_type == "private":
        private_id = normalized_chat_id or sender
        if _is_registered_private_chat(private_id):
            return TelegramInboundAccessDecision(allowed=True, reason="registered_private")
        return TelegramInboundAccessDecision(
            allowed=False,
            reason="private_not_registered",
            emit_invite=True,
        )

    access = get_chat_access(normalized_chat_id)
    if not access:
        return TelegramInboundAccessDecision(
            allowed=False,
            reason="group_not_approved",
            emit_invite=True,
        )
    if str(access.get("status") or "").strip().lower() != "active":
        return TelegramInboundAccessDecision(
            allowed=False,
            reason="group_not_active",
            leave_chat=True,
            access=access,
        )
    return TelegramInboundAccessDecision(
        allowed=True,
        reason="group_active",
        access=access,
    )


def owner_telegram_user_id(access: dict[str, Any] | None) -> str | None:
    if not isinstance(access, dict):
        return None
    owner_user_id = str(access.get("owner_user_id") or "").strip()
    if not owner_user_id:
        return None
    return resolve_service_user_id(user_id=owner_user_id, service_id=TELEGRAM_SERVICE_ID)


def can_deliver_to_chat(chat_id: str) -> bool:
    normalized = str(chat_id or "").strip()
    if not normalized:
        return False
    access = get_chat_access(normalized)
    if access:
        return str(access.get("status") or "").strip().lower() == "active"
    return _is_registered_private_chat(normalized)


def _is_registered_private_chat(telegram_chat_id: str) -> bool:
    value = str(telegram_chat_id or "").strip()
    if not value:
        return False
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT u.user_id
            FROM user_service_resolvers r
            JOIN users u ON u.user_id = r.user_id
            WHERE r.service_id = ?
              AND r.service_user_id = ?
              AND r.is_active = 1
              AND u.is_active = 1
            LIMIT 1
            """,
            (TELEGRAM_SERVICE_ID, value),
        ).fetchone()
    return bool(row)


def _row_to_chat_access(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "chat_id": row[0],
        "chat_type": row[1],
        "status": row[2],
        "owner_user_id": row[3],
        "policy": row[4],
        "created_at": row[5],
        "updated_at": row[6],
        "revoked_at": row[7],
        "revoke_reason": row[8],
    }


def _normalize_chat_type(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value in {"private", "group", "supergroup"}:
        return value
    return "private"


def _normalize_status(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value in {"active", "revoked", "pending"}:
        return value
    return "active"


def _normalize_policy(raw: Any, *, chat_type: str) -> str:
    value = str(raw or "").strip().lower()
    if value in {"registered_private", "owner_managed_group"}:
        return value
    if chat_type == "private":
        return "registered_private"
    return "owner_managed_group"


def _guess_chat_type(chat_id: str) -> str:
    rendered = str(chat_id or "").strip()
    if rendered.startswith("-"):
        return "group"
    return "private"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
