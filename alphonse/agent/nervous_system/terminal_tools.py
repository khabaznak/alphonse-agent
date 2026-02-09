from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def upsert_terminal_sandbox(record: dict[str, Any]) -> str:
    sandbox_id = str(record.get("sandbox_id") or uuid.uuid4())
    owner_principal_id = str(record.get("owner_principal_id") or "").strip()
    label = str(record.get("label") or "").strip()
    path = str(record.get("path") or "").strip()
    if not owner_principal_id or not label or not path:
        raise ValueError("owner_principal_id, label, and path are required")
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO terminal_sandboxes (
              sandbox_id, owner_principal_id, label, path, is_active, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sandbox_id) DO UPDATE SET
              owner_principal_id = excluded.owner_principal_id,
              label = excluded.label,
              path = excluded.path,
              is_active = excluded.is_active,
              updated_at = excluded.updated_at
            """,
            (
                sandbox_id,
                owner_principal_id,
                label,
                path,
                1 if bool(record.get("is_active", True)) else 0,
                now,
                now,
            ),
        )
        conn.commit()
    return sandbox_id


def list_terminal_sandboxes(
    *,
    owner_principal_id: str | None = None,
    active_only: bool = False,
    limit: int = 200,
) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if owner_principal_id:
        filters.append("owner_principal_id = ?")
        values.append(owner_principal_id)
    if active_only:
        filters.append("is_active = 1")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT sandbox_id, owner_principal_id, label, path, is_active, created_at, updated_at "
        f"FROM terminal_sandboxes {where} ORDER BY updated_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_sandbox(row) for row in rows]


def get_terminal_sandbox(sandbox_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT sandbox_id, owner_principal_id, label, path, is_active, created_at, updated_at
            FROM terminal_sandboxes
            WHERE sandbox_id = ?
            """,
            (sandbox_id,),
        ).fetchone()
    return _row_to_sandbox(row) if row else None


def patch_terminal_sandbox(sandbox_id: str, updates: dict[str, Any]) -> dict[str, Any] | None:
    current = get_terminal_sandbox(sandbox_id)
    if not current:
        return None
    merged = dict(current)
    merged.update({k: v for k, v in updates.items() if v is not None})
    merged["sandbox_id"] = sandbox_id
    upsert_terminal_sandbox(merged)
    return get_terminal_sandbox(sandbox_id)


def delete_terminal_sandbox(sandbox_id: str) -> bool:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        cur = conn.execute("DELETE FROM terminal_sandboxes WHERE sandbox_id = ?", (sandbox_id,))
        conn.commit()
    return cur.rowcount > 0


def create_terminal_session(record: dict[str, Any]) -> str:
    session_id = str(record.get("session_id") or uuid.uuid4())
    principal_id = str(record.get("principal_id") or "").strip()
    sandbox_id = str(record.get("sandbox_id") or "").strip()
    if not principal_id or not sandbox_id:
        raise ValueError("principal_id and sandbox_id are required")
    status = str(record.get("status") or "pending")
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO terminal_sessions (
              session_id, principal_id, sandbox_id, status, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, principal_id, sandbox_id, status, now, now),
        )
        conn.commit()
    return session_id


def ensure_terminal_session(*, principal_id: str, sandbox_id: str) -> str:
    """Return existing active session for principal + sandbox or create one."""
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT session_id
            FROM terminal_sessions
            WHERE principal_id = ? AND sandbox_id = ?
            ORDER BY updated_at DESC
            LIMIT 1
            """,
            (principal_id, sandbox_id),
        ).fetchone()
    if row:
        return row[0]
    return create_terminal_session(
        {
            "principal_id": principal_id,
            "sandbox_id": sandbox_id,
            "status": "pending",
        }
    )


def update_terminal_session_status(session_id: str, status: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            "UPDATE terminal_sessions SET status = ?, updated_at = ? WHERE session_id = ?",
            (status, _now_iso(), session_id),
        )
        conn.commit()
    return get_terminal_session(session_id)


def get_terminal_session(session_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT session_id, principal_id, sandbox_id, status, created_at, updated_at
            FROM terminal_sessions
            WHERE session_id = ?
            """,
            (session_id,),
        ).fetchone()
    return _row_to_session(row) if row else None


def list_terminal_sessions(
    *,
    principal_id: str | None = None,
    sandbox_id: str | None = None,
    status: str | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if principal_id:
        filters.append("principal_id = ?")
        values.append(principal_id)
    if sandbox_id:
        filters.append("sandbox_id = ?")
        values.append(sandbox_id)
    if status:
        filters.append("status = ?")
        values.append(status)
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT session_id, principal_id, sandbox_id, status, created_at, updated_at "
        f"FROM terminal_sessions {where} ORDER BY updated_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_session(row) for row in rows]


def create_terminal_command(record: dict[str, Any]) -> str:
    command_id = str(record.get("command_id") or uuid.uuid4())
    session_id = str(record.get("session_id") or "").strip()
    command = str(record.get("command") or "").strip()
    cwd = str(record.get("cwd") or "").strip()
    if not session_id or not command or not cwd:
        raise ValueError("session_id, command, and cwd are required")
    status = str(record.get("status") or "pending")
    requested_by = record.get("requested_by")
    approved_by = record.get("approved_by")
    timeout_seconds = record.get("timeout_seconds")
    if timeout_seconds is not None:
        try:
            timeout_seconds = float(timeout_seconds)
        except (TypeError, ValueError):
            timeout_seconds = None
    now = _now_iso()
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            INSERT INTO terminal_commands (
              command_id, session_id, command, cwd, status, stdout, stderr, exit_code,
              timeout_seconds, requested_by, approved_by, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                command_id,
                session_id,
                command,
                cwd,
                status,
                record.get("stdout"),
                record.get("stderr"),
                record.get("exit_code"),
                timeout_seconds,
                requested_by,
                approved_by,
                now,
                now,
            ),
        )
        conn.commit()
    return command_id


def update_terminal_command_status(command_id: str, status: str, *, approved_by: str | None = None) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            UPDATE terminal_commands
            SET status = ?, approved_by = COALESCE(?, approved_by), updated_at = ?
            WHERE command_id = ?
            """,
            (status, approved_by, _now_iso(), command_id),
        )
        conn.commit()
    return get_terminal_command(command_id)


def record_terminal_command_output(
    command_id: str,
    *,
    stdout: str | None,
    stderr: str | None,
    exit_code: int | None,
    status: str,
) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        conn.execute(
            """
            UPDATE terminal_commands
            SET stdout = ?, stderr = ?, exit_code = ?, status = ?, updated_at = ?
            WHERE command_id = ?
            """,
            (stdout, stderr, exit_code, status, _now_iso(), command_id),
        )
        conn.commit()
    return get_terminal_command(command_id)


def get_terminal_command(command_id: str) -> dict[str, Any] | None:
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        row = conn.execute(
            """
            SELECT command_id, session_id, command, cwd, status, stdout, stderr, exit_code,
                   timeout_seconds, requested_by, approved_by, created_at, updated_at
            FROM terminal_commands
            WHERE command_id = ?
            """,
            (command_id,),
        ).fetchone()
    return _row_to_command(row) if row else None


def list_terminal_commands(
    *,
    session_id: str | None = None,
    status: str | None = None,
    approved_only: bool | None = None,
    limit: int = 200,
) -> list[dict[str, Any]]:
    filters: list[str] = []
    values: list[Any] = []
    if session_id:
        filters.append("session_id = ?")
        values.append(session_id)
    if status:
        filters.append("status = ?")
        values.append(status)
    if approved_only:
        filters.append("status = 'approved'")
    where = f"WHERE {' AND '.join(filters)}" if filters else ""
    query = (
        "SELECT command_id, session_id, command, cwd, status, stdout, stderr, exit_code, "
        "timeout_seconds, requested_by, approved_by, created_at, updated_at "
        f"FROM terminal_commands {where} ORDER BY updated_at DESC LIMIT ?"
    )
    values.append(limit)
    with sqlite3.connect(resolve_nervous_system_db_path()) as conn:
        rows = conn.execute(query, tuple(values)).fetchall()
    return [_row_to_command(row) for row in rows]


def _row_to_sandbox(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "sandbox_id": row[0],
        "owner_principal_id": row[1],
        "label": row[2],
        "path": row[3],
        "is_active": bool(row[4]),
        "created_at": row[5],
        "updated_at": row[6],
    }


def _row_to_session(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "session_id": row[0],
        "principal_id": row[1],
        "sandbox_id": row[2],
        "status": row[3],
        "created_at": row[4],
        "updated_at": row[5],
    }


def _row_to_command(row: sqlite3.Row | tuple | None) -> dict[str, Any]:
    if row is None:
        return {}
    if not isinstance(row, tuple):
        row = tuple(row)
    return {
        "command_id": row[0],
        "session_id": row[1],
        "command": row[2],
        "cwd": row[3],
        "status": row[4],
        "stdout": row[5],
        "stderr": row[6],
        "exit_code": row[7],
        "timeout_seconds": row[8],
        "requested_by": row[9],
        "approved_by": row[10],
        "created_at": row[11],
        "updated_at": row[12],
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
