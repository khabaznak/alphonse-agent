from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone, timedelta
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.cognition.capability_gaps.triage import triage_gap_text


def insert_gap(record: dict[str, Any]) -> str:
    gap_id = str(record.get("gap_id") or uuid.uuid4())
    created_at = record.get("created_at") or _now_iso()
    missing_slots = record.get("missing_slots")
    metadata = _merge_triage_metadata(
        record.get("metadata"),
        record.get("user_text"),
        record.get("reason"),
    )
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO capability_gaps (
              gap_id, created_at, principal_type, principal_id, channel_type, channel_id,
              correlation_id, user_text, intent, confidence, missing_slots, reason, status,
              resolution_notes, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                gap_id,
                created_at,
                record.get("principal_type"),
                record.get("principal_id"),
                record.get("channel_type"),
                record.get("channel_id"),
                record.get("correlation_id"),
                record.get("user_text"),
                record.get("intent"),
                record.get("confidence"),
                _to_json(missing_slots),
                record.get("reason"),
                record.get("status") or "open",
                record.get("resolution_notes"),
                _to_json(metadata),
            ),
        )
        conn.commit()
    return gap_id


def list_gaps(
    *,
    status: str | None = None,
    limit: int = 50,
    include_all: bool = False,
) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if status:
        filters.append("status = ?")
        values.append(status)
    if not include_all and not status:
        filters.append("status = 'open'")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT gap_id, created_at, principal_type, principal_id, channel_type, channel_id, "
        "correlation_id, user_text, intent, confidence, missing_slots, reason, status, "
        "resolution_notes, metadata "
        f"FROM capability_gaps {where} ORDER BY created_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_gap(row) for row in rows]


def get_gap(gap_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT gap_id, created_at, principal_type, principal_id, channel_type, channel_id,
                   correlation_id, user_text, intent, confidence, missing_slots, reason, status,
                   resolution_notes, metadata
            FROM capability_gaps WHERE gap_id = ?
            """,
            (gap_id,),
        ).fetchone()
    return _row_to_gap(row) if row else None


def update_gap_status(gap_id: str, status: str, note: str | None = None) -> bool:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute(
            """
            UPDATE capability_gaps
            SET status = ?, resolution_notes = ?, created_at = created_at
            WHERE gap_id = ?
            """,
            (status, note, gap_id),
        )
        conn.commit()
    return cur.rowcount > 0


def list_recent_gaps(hours: int = 24) -> list[dict[str, Any]]:
    since = datetime.now(timezone.utc) - timedelta(hours=hours)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(
            """
            SELECT gap_id, created_at, principal_type, principal_id, channel_type, channel_id,
                   correlation_id, user_text, intent, confidence, missing_slots, reason, status,
                   resolution_notes, metadata
            FROM capability_gaps
            WHERE created_at >= ?
            ORDER BY created_at DESC
            """,
            (since.isoformat(),),
        ).fetchall()
    return [_row_to_gap(row) for row in rows]


def _row_to_gap(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "gap_id": row[0],
        "created_at": row[1],
        "principal_type": row[2],
        "principal_id": row[3],
        "channel_type": row[4],
        "channel_id": row[5],
        "correlation_id": row[6],
        "user_text": row[7],
        "intent": row[8],
        "confidence": row[9],
        "missing_slots": _parse_json(row[10]) or [],
        "reason": row[11],
        "status": row[12],
        "resolution_notes": row[13],
        "metadata": _parse_json(row[14]) or {},
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


def _merge_triage_metadata(
    metadata: dict[str, Any] | None,
    user_text: str | None,
    reason: str | None = None,
) -> dict[str, Any] | None:
    if not user_text:
        return metadata
    triage = triage_gap_text(str(user_text))
    if not triage.get("suggested_intent") and reason:
        triage["category"] = reason
    merged = dict(metadata or {})
    merged.setdefault("triage", triage)
    return merged
