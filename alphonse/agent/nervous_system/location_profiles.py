from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def upsert_location_profile(record: dict[str, Any]) -> str:
    location_id = str(record.get("location_id") or uuid.uuid4())
    principal_id = str(record.get("principal_id") or "")
    if not principal_id:
        raise ValueError("principal_id is required")
    label = str(record.get("label") or "other")
    if label not in {"home", "work", "other"}:
        raise ValueError("label must be one of: home, work, other")
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        _ensure_principal_exists(conn, principal_id, now)
        conn.execute(
            """
            INSERT INTO location_profiles (
              location_id, principal_id, label, address_text, latitude, longitude,
              source, confidence, is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(location_id) DO UPDATE SET
              principal_id = excluded.principal_id,
              label = excluded.label,
              address_text = excluded.address_text,
              latitude = excluded.latitude,
              longitude = excluded.longitude,
              source = excluded.source,
              confidence = excluded.confidence,
              is_active = excluded.is_active,
              updated_at = excluded.updated_at
            """,
            (
                location_id,
                principal_id,
                label,
                record.get("address_text"),
                record.get("latitude"),
                record.get("longitude"),
                record.get("source") or "user",
                record.get("confidence"),
                1 if bool(record.get("is_active", True)) else 0,
                now,
                now,
            ),
        )
        conn.commit()
    return location_id


def list_location_profiles(
    *,
    principal_id: str | None = None,
    label: str | None = None,
    active_only: bool = False,
    limit: int = 100,
) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if principal_id:
        filters.append("principal_id = ?")
        values.append(principal_id)
    if label:
        filters.append("label = ?")
        values.append(label)
    if active_only:
        filters.append("is_active = 1")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT location_id, principal_id, label, address_text, latitude, longitude, "
        "source, confidence, is_active, created_at, updated_at "
        f"FROM location_profiles {where} ORDER BY updated_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_location_profile(row) for row in rows]


def get_location_profile(location_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT location_id, principal_id, label, address_text, latitude, longitude,
                   source, confidence, is_active, created_at, updated_at
            FROM location_profiles
            WHERE location_id = ?
            """,
            (location_id,),
        ).fetchone()
    return _row_to_location_profile(row) if row else None


def delete_location_profile(location_id: str) -> bool:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute(
            "DELETE FROM location_profiles WHERE location_id = ?",
            (location_id,),
        )
        conn.commit()
    return cur.rowcount > 0


def insert_device_location(record: dict[str, Any]) -> str:
    entry_id = str(record.get("id") or uuid.uuid4())
    now = _now_iso()
    observed_at = str(record.get("observed_at") or now)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        principal_id = record.get("principal_id")
        if isinstance(principal_id, str) and principal_id:
            _ensure_principal_exists(conn, principal_id, now)
        conn.execute(
            """
            INSERT INTO device_locations (
              id, principal_id, device_id, latitude, longitude, accuracy_meters,
              source, observed_at, metadata_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry_id,
                record.get("principal_id"),
                record.get("device_id"),
                record.get("latitude"),
                record.get("longitude"),
                record.get("accuracy_meters"),
                record.get("source") or "device",
                observed_at,
                _to_json(record.get("metadata")),
                now,
            ),
        )
        conn.commit()
    return entry_id


def list_device_locations(
    *,
    principal_id: str | None = None,
    device_id: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if principal_id:
        filters.append("principal_id = ?")
        values.append(principal_id)
    if device_id:
        filters.append("device_id = ?")
        values.append(device_id)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT id, principal_id, device_id, latitude, longitude, accuracy_meters, "
        "source, observed_at, metadata_json, created_at "
        f"FROM device_locations {where} ORDER BY observed_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_device_location(row) for row in rows]


def _row_to_location_profile(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "location_id": row[0],
        "principal_id": row[1],
        "label": row[2],
        "address_text": row[3],
        "latitude": row[4],
        "longitude": row[5],
        "source": row[6],
        "confidence": row[7],
        "is_active": bool(row[8]),
        "created_at": row[9],
        "updated_at": row[10],
    }


def _row_to_device_location(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "id": row[0],
        "principal_id": row[1],
        "device_id": row[2],
        "latitude": row[3],
        "longitude": row[4],
        "accuracy_meters": row[5],
        "source": row[6],
        "observed_at": row[7],
        "metadata": _parse_json(row[8]) or {},
        "created_at": row[9],
    }


def _to_json(value: Any) -> str | None:
    if value is None:
        return None
    return json.dumps(value)


def _parse_json(value: str | None) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_principal_exists(conn: sqlite3.Connection, principal_id: str, now: str) -> None:
    conn.execute(
        """
        INSERT OR IGNORE INTO principals (
          principal_id, principal_type, created_at, updated_at
        ) VALUES (?, 'person', ?, ?)
        """,
        (principal_id, now, now),
    )
