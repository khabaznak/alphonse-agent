from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def insert_gap_proposal(record: dict[str, Any]) -> str:
    proposal_id = str(record.get("id") or uuid.uuid4())
    created_at = record.get("created_at") or _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO gap_proposals (
              id, gap_id, created_at, status, proposed_category, confidence,
              proposed_next_action, proposed_intent_name, proposed_clarifying_question,
              notes, language, reviewer, reviewed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposal_id,
                record.get("gap_id"),
                created_at,
                record.get("status") or "pending",
                record.get("proposed_category"),
                record.get("confidence"),
                record.get("proposed_next_action"),
                record.get("proposed_intent_name"),
                record.get("proposed_clarifying_question"),
                record.get("notes"),
                record.get("language"),
                record.get("reviewer"),
                record.get("reviewed_at"),
            ),
        )
        conn.commit()
    return proposal_id


def list_gap_proposals(*, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if status:
        filters.append("status = ?")
        values.append(status)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT id, gap_id, created_at, status, proposed_category, confidence, "
        "proposed_next_action, proposed_intent_name, proposed_clarifying_question, "
        "notes, language, reviewer, reviewed_at "
        f"FROM gap_proposals {where} ORDER BY created_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_proposal(row) for row in rows]


def get_gap_proposal(proposal_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT id, gap_id, created_at, status, proposed_category, confidence,
                   proposed_next_action, proposed_intent_name, proposed_clarifying_question,
                   notes, language, reviewer, reviewed_at
            FROM gap_proposals WHERE id = ?
            """,
            (proposal_id,),
        ).fetchone()
    return _row_to_proposal(row) if row else None


def get_pending_proposal_for_gap(gap_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT id, gap_id, created_at, status, proposed_category, confidence,
                   proposed_next_action, proposed_intent_name, proposed_clarifying_question,
                   notes, language, reviewer, reviewed_at
            FROM gap_proposals WHERE gap_id = ? AND status = 'pending'
            ORDER BY created_at DESC LIMIT 1
            """,
            (gap_id,),
        ).fetchone()
    return _row_to_proposal(row) if row else None


def update_gap_proposal_status(
    proposal_id: str,
    status: str,
    *,
    reviewer: str | None = None,
    reviewed_at: str | None = None,
    notes: str | None = None,
) -> bool:
    reviewed_at = reviewed_at or _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute(
            """
            UPDATE gap_proposals
            SET status = ?, reviewer = ?, reviewed_at = ?, notes = ?
            WHERE id = ?
            """,
            (status, reviewer, reviewed_at, notes, proposal_id),
        )
        conn.commit()
    return cur.rowcount > 0


def _row_to_proposal(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "id": row[0],
        "gap_id": row[1],
        "created_at": row[2],
        "status": row[3],
        "proposed_category": row[4],
        "confidence": row[5],
        "proposed_next_action": row[6],
        "proposed_intent_name": row[7],
        "proposed_clarifying_question": row[8],
        "notes": row[9],
        "language": row[10],
        "reviewer": row[11],
        "reviewed_at": row[12],
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
