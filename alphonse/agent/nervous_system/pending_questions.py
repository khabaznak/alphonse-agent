from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

_DEFAULT_EXPIRY_SECONDS = 24 * 60 * 60


def create_pending_question(
    *,
    task_id: str,
    originator_user_id: str,
    originator_conversation_key: str,
    respondent_user_id: str,
    respondent_conversation_key: str,
    question_text: str,
    expires_in_seconds: int | None = None,
) -> dict[str, Any]:
    now = datetime.now(timezone.utc)
    ttl = max(int(expires_in_seconds or _DEFAULT_EXPIRY_SECONDS), 60)
    record = {
        "question_id": str(uuid.uuid4()),
        "task_id": str(task_id or "").strip(),
        "originator_user_id": str(originator_user_id or "").strip(),
        "originator_conversation_key": str(originator_conversation_key or "").strip(),
        "respondent_user_id": str(respondent_user_id or "").strip(),
        "respondent_conversation_key": str(respondent_conversation_key or "").strip(),
        "question_text": str(question_text or "").strip(),
        "status": "pending",
        "created_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=ttl)).isoformat(),
        "updated_at": now.isoformat(),
    }
    required = (
        "task_id",
        "originator_user_id",
        "originator_conversation_key",
        "respondent_user_id",
        "respondent_conversation_key",
        "question_text",
    )
    if any(not record[key] for key in required):
        raise ValueError("pending_question_missing_required_field")
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO pending_questions (
              question_id, task_id, originator_user_id, originator_conversation_key,
              respondent_user_id, respondent_conversation_key, question_text, status,
              created_at, expires_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?)
            """,
            (
                record["question_id"], record["task_id"], record["originator_user_id"],
                record["originator_conversation_key"], record["respondent_user_id"],
                record["respondent_conversation_key"], record["question_text"],
                record["created_at"], record["expires_at"], record["updated_at"],
            ),
        )
    return record


def bind_outbound_message(*, question_id: str, provider_message_id: str | None) -> bool:
    message_id = str(provider_message_id or "").strip() or None
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE pending_questions SET outbound_provider_message_id = ?, updated_at = ? "
            "WHERE question_id = ? AND status = 'pending'",
            (message_id, _now_iso(), str(question_id or "").strip()),
        )
        return cursor.rowcount == 1


def get_pending_question(question_id: str) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM pending_questions WHERE question_id = ?",
            (str(question_id or "").strip(),),
        ).fetchone()
    return dict(row) if row is not None else None


def list_pending_for_respondent(respondent_user_id: str) -> list[dict[str, Any]]:
    expire_pending_questions()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM pending_questions WHERE respondent_user_id = ? AND status = 'pending' "
            "ORDER BY created_at",
            (str(respondent_user_id or "").strip(),),
        ).fetchall()
    return [dict(row) for row in rows]


def find_pending_by_reply(*, respondent_user_id: str, reply_to_provider_message_id: str) -> dict[str, Any] | None:
    reply_id = str(reply_to_provider_message_id or "").strip()
    if not reply_id:
        return None
    expire_pending_questions()
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM pending_questions WHERE respondent_user_id = ? "
            "AND outbound_provider_message_id = ? AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
            (str(respondent_user_id or "").strip(), reply_id),
        ).fetchone()
    return dict(row) if row is not None else None


def answer_pending_question(
    *,
    question_id: str,
    respondent_user_id: str,
    answer_text: str,
    inbound_provider_message_id: str | None,
) -> dict[str, Any] | None:
    now = _now_iso()
    with _connect() as conn:
        cursor = conn.execute(
            """
            UPDATE pending_questions
            SET status = 'answered', answer_text = ?, inbound_provider_message_id = ?,
                answered_at = ?, updated_at = ?
            WHERE question_id = ? AND respondent_user_id = ? AND status = 'pending'
            """,
            (
                str(answer_text or "").strip(),
                str(inbound_provider_message_id or "").strip() or None,
                now,
                now,
                str(question_id or "").strip(),
                str(respondent_user_id or "").strip(),
            ),
        )
        if cursor.rowcount != 1:
            return None
        row = conn.execute("SELECT * FROM pending_questions WHERE question_id = ?", (question_id,)).fetchone()
    return dict(row) if row is not None else None


def expire_pending_questions(*, now: str | None = None) -> list[dict[str, Any]]:
    current = str(now or _now_iso()).strip()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM pending_questions WHERE status = 'pending' AND expires_at <= ?",
            (current,),
        ).fetchall()
        if rows:
            conn.execute(
                "UPDATE pending_questions SET status = 'expired', updated_at = ? "
                "WHERE status = 'pending' AND expires_at <= ?",
                (current, current),
            )
    return [dict(row) for row in rows]


def cancel_questions_for_task(task_id: str) -> int:
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE pending_questions SET status = 'cancelled', updated_at = ? "
            "WHERE task_id = ? AND status = 'pending'",
            (_now_iso(), str(task_id or "").strip()),
        )
        return int(cursor.rowcount or 0)


def cancel_pending_question(question_id: str) -> bool:
    with _connect() as conn:
        cursor = conn.execute(
            "UPDATE pending_questions SET status = 'cancelled', updated_at = ? "
            "WHERE question_id = ? AND status = 'pending'",
            (_now_iso(), str(question_id or "").strip()),
        )
        return cursor.rowcount == 1


def _connect() -> sqlite3.Connection:
    path = resolve_nervous_system_db_path()
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_questions (
          question_id TEXT PRIMARY KEY, task_id TEXT NOT NULL,
          originator_user_id TEXT NOT NULL, originator_conversation_key TEXT NOT NULL,
          respondent_user_id TEXT NOT NULL, respondent_conversation_key TEXT NOT NULL,
          question_text TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
          outbound_provider_message_id TEXT, answer_text TEXT, inbound_provider_message_id TEXT,
          created_at TEXT NOT NULL, answered_at TEXT, expires_at TEXT NOT NULL, updated_at TEXT NOT NULL,
          CHECK (status IN ('pending', 'answered', 'expired', 'cancelled'))
        ) STRICT
        """
    )
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
