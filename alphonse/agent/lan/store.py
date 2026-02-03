from __future__ import annotations

import json
import secrets
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


@dataclass(frozen=True)
class PairingCode:
    code: str
    expires_at: datetime


@dataclass(frozen=True)
class PairedDevice:
    device_id: str
    device_name: str | None
    paired_at: datetime
    allowed_scopes: list[str]
    last_seen_at: datetime | None
    last_status: dict[str, Any] | None
    last_status_at: datetime | None


DEFAULT_ALLOWED_SCOPES = ["request_status", "send_status"]


def generate_pairing_code(ttl_minutes: int = 15) -> PairingCode:
    expires_at = _utcnow() + timedelta(minutes=ttl_minutes)
    for _ in range(5):
        code = _random_code()
        if _insert_pairing_code(code, expires_at):
            return PairingCode(code=code, expires_at=expires_at)
    raise RuntimeError("Unable to allocate unique pairing code")


def consume_pairing_code(code: str) -> PairingCode | None:
    record = _fetch_one(
        "SELECT code, expires_at FROM pairing_codes WHERE code = ?",
        (code,),
    )
    if not record:
        return None
    expires_at = _parse_timestamp(record.get("expires_at"))
    if not expires_at or expires_at <= _utcnow():
        _execute("DELETE FROM pairing_codes WHERE code = ?", (code,))
        return None
    _execute("DELETE FROM pairing_codes WHERE code = ?", (code,))
    return PairingCode(code=code, expires_at=expires_at)


def get_pairing_code(code: str) -> PairingCode | None:
    record = _fetch_one(
        "SELECT code, expires_at FROM pairing_codes WHERE code = ?",
        (code,),
    )
    if not record:
        return None
    expires_at = _parse_timestamp(record.get("expires_at"))
    if not expires_at or expires_at <= _utcnow():
        return None
    return PairingCode(code=code, expires_at=expires_at)


def register_device(
    device_id: str,
    device_name: str | None,
    allowed_scopes: list[str] | None = None,
) -> PairedDevice:
    allowed_scopes = allowed_scopes or list(DEFAULT_ALLOWED_SCOPES)
    now = _utcnow()
    _execute(
        """
        INSERT INTO paired_devices
          (device_id, device_name, paired_at, allowed_scopes, last_seen_at, last_status, last_status_at)
        VALUES (?, ?, ?, ?, NULL, NULL, NULL)
        """,
        (
            device_id,
            device_name,
            now.isoformat(),
            json.dumps(allowed_scopes),
        ),
    )
    return PairedDevice(
        device_id=device_id,
        device_name=device_name,
        paired_at=now,
        allowed_scopes=allowed_scopes,
        last_seen_at=None,
        last_status=None,
        last_status_at=None,
    )


def get_paired_device(device_id: str) -> PairedDevice | None:
    record = _fetch_one(
        """
        SELECT device_id, device_name, paired_at, allowed_scopes, last_seen_at, last_status, last_status_at
        FROM paired_devices
        WHERE device_id = ?
        """,
        (device_id,),
    )
    if not record:
        return None
    return _row_to_device(record)


def update_device_last_seen(device_id: str) -> None:
    _execute(
        "UPDATE paired_devices SET last_seen_at = ? WHERE device_id = ?",
        (_utcnow().isoformat(), device_id),
    )


def update_device_status(device_id: str, payload: dict[str, Any]) -> None:
    _execute(
        """
        UPDATE paired_devices
        SET last_seen_at = ?, last_status = ?, last_status_at = ?
        WHERE device_id = ?
        """,
        (
            _utcnow().isoformat(),
            json.dumps(payload),
            _utcnow().isoformat(),
            device_id,
        ),
    )


def get_latest_last_seen() -> datetime | None:
    record = _fetch_one(
        "SELECT last_seen_at FROM paired_devices WHERE last_seen_at IS NOT NULL ORDER BY last_seen_at DESC LIMIT 1",
        (),
    )
    if not record:
        return None
    return _parse_timestamp(record.get("last_seen_at"))


def is_paired_device(device_id: str) -> bool:
    record = _fetch_one(
        "SELECT 1 FROM paired_devices WHERE device_id = ? LIMIT 1",
        (device_id,),
    )
    return bool(record)


def list_paired_devices(limit: int = 100) -> list[PairedDevice]:
    rows = _fetch_all(
        """
        SELECT device_id, device_name, paired_at, allowed_scopes, last_seen_at, last_status, last_status_at
        FROM paired_devices
        ORDER BY paired_at DESC
        LIMIT ?
        """,
        (limit,),
    )
    return [_row_to_device(row) for row in rows]


def _row_to_device(record: dict[str, Any]) -> PairedDevice:
    allowed_raw = record.get("allowed_scopes") or "[]"
    try:
        allowed_scopes = list(json.loads(allowed_raw))
    except (TypeError, json.JSONDecodeError):
        allowed_scopes = []
    last_status = record.get("last_status")
    if isinstance(last_status, str):
        try:
            last_status = json.loads(last_status)
        except json.JSONDecodeError:
            last_status = None
    return PairedDevice(
        device_id=str(record.get("device_id")),
        device_name=record.get("device_name"),
        paired_at=_parse_timestamp(record.get("paired_at")) or _utcnow(),
        allowed_scopes=allowed_scopes,
        last_seen_at=_parse_timestamp(record.get("last_seen_at")),
        last_status=last_status if isinstance(last_status, dict) else None,
        last_status_at=_parse_timestamp(record.get("last_status_at")),
    )


def _insert_pairing_code(code: str, expires_at: datetime) -> bool:
    try:
        _execute(
            "INSERT INTO pairing_codes (code, expires_at, created_at) VALUES (?, ?, ?)",
            (code, expires_at.isoformat(), _utcnow().isoformat()),
        )
    except sqlite3.IntegrityError:
        return False
    return True


def _random_code(length: int = 6) -> str:
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(length))


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
    return _row_to_dict(row)


def _fetch_all(query: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(row) for row in rows]

def _execute(query: str, params: tuple[Any, ...]) -> None:
    with _connect() as conn:
        conn.execute(query, params)
        conn.commit()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)
