from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


VALID_KINDS = {"user", "chat"}
VALID_STATUSES = {"pending", "approved", "denied", "claimed"}


def upsert_access_request(record: dict[str, Any]) -> str:
    kind = _normalize_kind(record.get("kind"))
    provider_key = str(record.get("provider_key") or "telegram").strip().lower()
    if not provider_key:
        raise ValueError("provider_key is required")
    provider_user_id = _clean_optional(record.get("provider_user_id"))
    channel_target = _clean_optional(record.get("channel_target"))
    request_id = _clean_optional(record.get("request_id")) or _stable_request_id(
        kind=kind,
        provider_key=provider_key,
        provider_user_id=provider_user_id,
        channel_target=channel_target,
    )
    now = _now_iso()
    metadata = record.get("metadata")
    metadata_json = json.dumps(metadata if isinstance(metadata, dict) else {}, sort_keys=True)
    status = _normalize_status(record.get("status") or "pending")
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO access_requests (
              request_id, kind, provider_key, provider_user_id, channel_target,
              display_name, status, created_by_user_id, claimed_user_id, reason,
              expires_at, claimed_at, metadata_json, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(request_id) DO UPDATE SET
              kind = excluded.kind,
              provider_key = excluded.provider_key,
              provider_user_id = COALESCE(excluded.provider_user_id, access_requests.provider_user_id),
              channel_target = COALESCE(excluded.channel_target, access_requests.channel_target),
              display_name = COALESCE(excluded.display_name, access_requests.display_name),
              status = excluded.status,
              created_by_user_id = COALESCE(excluded.created_by_user_id, access_requests.created_by_user_id),
              claimed_user_id = COALESCE(excluded.claimed_user_id, access_requests.claimed_user_id),
              reason = COALESCE(excluded.reason, access_requests.reason),
              expires_at = COALESCE(excluded.expires_at, access_requests.expires_at),
              claimed_at = COALESCE(excluded.claimed_at, access_requests.claimed_at),
              metadata_json = excluded.metadata_json,
              updated_at = excluded.updated_at
            """,
            (
                request_id,
                kind,
                provider_key,
                provider_user_id,
                channel_target,
                _clean_optional(record.get("display_name")),
                status,
                _clean_optional(record.get("created_by_user_id")),
                _clean_optional(record.get("claimed_user_id")),
                _clean_optional(record.get("reason")),
                _clean_optional(record.get("expires_at")),
                _clean_optional(record.get("claimed_at")),
                metadata_json,
                now,
                now,
            ),
        )
        conn.commit()
    return request_id


def list_access_requests(
    *,
    status: str | None = None,
    kind: str | None = None,
    provider_key: str | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    filters: list[str] = []
    params: list[Any] = []
    if status:
        filters.append("status = ?")
        params.append(_normalize_status(status))
    if kind:
        filters.append("kind = ?")
        params.append(_normalize_kind(kind))
    provider = str(provider_key or "").strip().lower()
    if provider:
        filters.append("provider_key = ?")
        params.append(provider)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    params.append(max(1, min(int(limit or 50), 200)))
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(
            f"""
            SELECT request_id, kind, provider_key, provider_user_id, channel_target,
                   display_name, status, created_by_user_id, claimed_user_id, reason,
                   expires_at, claimed_at, metadata_json, created_at, updated_at
            FROM access_requests
            {where}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            tuple(params),
        ).fetchall()
    return [_row_to_request(row) for row in rows]


def get_access_request(request_id: str) -> dict[str, Any] | None:
    key = str(request_id or "").strip()
    if not key:
        return None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT request_id, kind, provider_key, provider_user_id, channel_target,
                   display_name, status, created_by_user_id, claimed_user_id, reason,
                   expires_at, claimed_at, metadata_json, created_at, updated_at
            FROM access_requests
            WHERE request_id = ?
            LIMIT 1
            """,
            (key,),
        ).fetchone()
    return _row_to_request(row) if row else None


def update_access_request(
    request_id: str,
    *,
    status: str,
    reason: str | None = None,
    claimed_user_id: str | None = None,
) -> dict[str, Any] | None:
    key = str(request_id or "").strip()
    if not key:
        return None
    normalized_status = _normalize_status(status)
    now = _now_iso()
    claimed_at = now if normalized_status in {"approved", "claimed"} and claimed_user_id else None
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            UPDATE access_requests
            SET status = ?,
                reason = COALESCE(?, reason),
                claimed_user_id = COALESCE(?, claimed_user_id),
                claimed_at = COALESCE(?, claimed_at),
                updated_at = ?
            WHERE request_id = ?
            """,
            (normalized_status, _clean_optional(reason), _clean_optional(claimed_user_id), claimed_at, now, key),
        )
        conn.commit()
    return get_access_request(key)


def _stable_request_id(
    *,
    kind: str,
    provider_key: str,
    provider_user_id: str | None,
    channel_target: str | None,
) -> str:
    subject = channel_target if kind == "chat" else provider_user_id or channel_target
    if subject:
        return f"{kind}:{provider_key}:{subject}"
    return f"{kind}:{provider_key}:{uuid.uuid4()}"


def _normalize_kind(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value not in VALID_KINDS:
        raise ValueError("invalid_access_request_kind")
    return value


def _normalize_status(raw: Any) -> str:
    value = str(raw or "").strip().lower()
    if value not in VALID_STATUSES:
        raise ValueError("invalid_access_request_status")
    return value


def _clean_optional(value: Any) -> str | None:
    rendered = str(value or "").strip()
    return rendered or None


def _row_to_request(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    metadata_raw = str(row[12] or "{}")
    try:
        metadata = json.loads(metadata_raw)
    except json.JSONDecodeError:
        metadata = {}
    return {
        "request_id": row[0],
        "kind": row[1],
        "provider_key": row[2],
        "provider_user_id": row[3],
        "channel_target": row[4],
        "display_name": row[5],
        "status": row[6],
        "created_by_user_id": row[7],
        "claimed_user_id": row[8],
        "reason": row[9],
        "expires_at": row[10],
        "claimed_at": row[11],
        "metadata": metadata if isinstance(metadata, dict) else {},
        "created_at": row[13],
        "updated_at": row[14],
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
