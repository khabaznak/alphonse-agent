from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


@dataclass(frozen=True)
class PairingRequest:
    pairing_id: str
    device_name: str | None
    challenge: str
    otp_hash: str
    status: str
    expires_at: datetime
    approved_via: str | None
    approved_at: datetime | None
    created_at: datetime


def create_pairing_request(device_name: str | None, ttl_minutes: int) -> tuple[PairingRequest, str]:
    pairing_id = str(uuid.uuid4())
    challenge = _random_code(8)
    otp = _random_code(6)
    now = _utcnow()
    expires_at = now + timedelta(minutes=ttl_minutes)
    otp_hash = _hash_token(otp)
    _execute(
        """
        INSERT INTO pairing_requests
          (pairing_id, device_name, challenge, otp_hash, status, expires_at, created_at)
        VALUES (?, ?, ?, ?, 'pending', ?, ?)
        """,
        (
            pairing_id,
            device_name,
            challenge,
            otp_hash,
            expires_at.isoformat(),
            now.isoformat(),
        ),
    )
    append_audit("pairing.created", pairing_id, {"device_name": device_name})
    request = PairingRequest(
        pairing_id=pairing_id,
        device_name=device_name,
        challenge=challenge,
        otp_hash=otp_hash,
        status="pending",
        expires_at=expires_at,
        approved_via=None,
        approved_at=None,
        created_at=now,
    )
    return request, otp


def get_pairing_request(pairing_id: str) -> PairingRequest | None:
    row = _fetch_one(
        """
        SELECT pairing_id, device_name, challenge, otp_hash, status, expires_at, approved_via, approved_at, created_at
        FROM pairing_requests WHERE pairing_id = ?
        """,
        (pairing_id,),
    )
    if not row:
        return None
    return _row_to_request(row)


def mark_expired(pairing_id: str) -> None:
    expired = _execute(
        """
        UPDATE pairing_requests SET status = 'expired'
        WHERE pairing_id = ? AND status = 'pending' AND expires_at <= ?
        """,
        (pairing_id, _utcnow().isoformat()),
    )
    if expired:
        append_audit("pairing.expired", pairing_id, {})


def approve_pairing(pairing_id: str, otp: str, via: str) -> bool:
    otp_hash = _hash_token(otp)
    now = _utcnow().isoformat()
    cur = _execute(
        """
        UPDATE pairing_requests
        SET status = 'approved', approved_via = ?, approved_at = ?
        WHERE pairing_id = ? AND status = 'pending' AND otp_hash = ? AND expires_at > ?
        """,
        (via, now, pairing_id, otp_hash, now),
    )
    if cur:
        append_audit("pairing.approved", pairing_id, {"via": via})
    return cur


def deny_pairing(pairing_id: str, via: str) -> bool:
    now = _utcnow().isoformat()
    cur = _execute(
        """
        UPDATE pairing_requests
        SET status = 'denied', approved_via = ?, approved_at = ?
        WHERE pairing_id = ? AND status = 'pending'
        """,
        (via, now, pairing_id),
    )
    if cur:
        append_audit("pairing.denied", pairing_id, {"via": via})
    return cur


def create_delivery_receipt(pairing_id: str, channel: str, status: str, details: dict[str, Any]) -> None:
    _execute(
        """
        INSERT INTO delivery_receipts (
          receipt_id, run_id, pairing_id, stage_id, action_id, skill, channel, status,
          details_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            None,
            pairing_id,
            None,
            None,
            channel,
            channel,
            status,
            json.dumps(details),
            _utcnow().isoformat(),
        ),
    )


def append_audit(event_type: str, correlation_id: str | None, payload: dict[str, Any]) -> None:
    _execute(
        """
        INSERT INTO audit_log (id, event_type, correlation_id, payload_json, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            str(uuid.uuid4()),
            event_type,
            correlation_id,
            json.dumps(payload),
            _utcnow().isoformat(),
        ),
    )


def _row_to_request(row: dict[str, Any]) -> PairingRequest:
    return PairingRequest(
        pairing_id=row["pairing_id"],
        device_name=row.get("device_name"),
        challenge=row.get("challenge"),
        otp_hash=row.get("otp_hash"),
        status=row.get("status"),
        expires_at=_parse_timestamp(row.get("expires_at")) or _utcnow(),
        approved_via=row.get("approved_via"),
        approved_at=_parse_timestamp(row.get("approved_at")),
        created_at=_parse_timestamp(row.get("created_at")) or _utcnow(),
    )


def _random_code(length: int = 6) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _connect() -> sqlite3.Connection:
    path = resolve_nervous_system_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_one(query: str, params: tuple[Any, ...]) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(query, params).fetchone()
    return dict(row) if row else None


def _execute(query: str, params: tuple[Any, ...]) -> bool:
    with _connect() as conn:
        cur = conn.execute(query, params)
        conn.commit()
    return cur.rowcount > 0
