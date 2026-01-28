import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_NERVE_DB = "rex/nervous_system/db/nerve-db"


def list_signals(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all(
        "select id, key, name, source, description, is_enabled, created_at, updated_at "
        "from signals order by name limit ?",
        (limit,),
    )


def get_signal(signal_id: int) -> dict[str, Any] | None:
    return _fetch_one(
        "select id, key, name, source, description, is_enabled, created_at, updated_at "
        "from signals where id = ?",
        (signal_id,),
    )


def create_signal(payload: dict[str, Any]) -> None:
    now = _timestamp()
    _execute(
        """
        insert into signals (key, name, source, description, is_enabled, created_at, updated_at)
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("key"),
            payload.get("name"),
            payload.get("source"),
            payload.get("description"),
            payload.get("is_enabled", 1),
            now,
            now,
        ),
    )


def update_signal(signal_id: int, payload: dict[str, Any]) -> None:
    now = _timestamp()
    _execute(
        """
        update signals
        set key = ?, name = ?, source = ?, description = ?, is_enabled = ?, updated_at = ?
        where id = ?
        """,
        (
            payload.get("key"),
            payload.get("name"),
            payload.get("source"),
            payload.get("description"),
            payload.get("is_enabled", 1),
            now,
            signal_id,
        ),
    )


def delete_signal(signal_id: int) -> None:
    _execute("delete from signals where id = ?", (signal_id,))


def list_states(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all(
        "select id, key, name, description, is_terminal, is_enabled, created_at, updated_at "
        "from states order by name limit ?",
        (limit,),
    )


def get_state(state_id: int) -> dict[str, Any] | None:
    return _fetch_one(
        "select id, key, name, description, is_terminal, is_enabled, created_at, updated_at "
        "from states where id = ?",
        (state_id,),
    )


def create_state(payload: dict[str, Any]) -> None:
    now = _timestamp()
    _execute(
        """
        insert into states (key, name, description, is_terminal, is_enabled, created_at, updated_at)
        values (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("key"),
            payload.get("name"),
            payload.get("description"),
            payload.get("is_terminal", 0),
            payload.get("is_enabled", 1),
            now,
            now,
        ),
    )


def update_state(state_id: int, payload: dict[str, Any]) -> None:
    now = _timestamp()
    _execute(
        """
        update states
        set key = ?, name = ?, description = ?, is_terminal = ?, is_enabled = ?, updated_at = ?
        where id = ?
        """,
        (
            payload.get("key"),
            payload.get("name"),
            payload.get("description"),
            payload.get("is_terminal", 0),
            payload.get("is_enabled", 1),
            now,
            state_id,
        ),
    )


def delete_state(state_id: int) -> None:
    _execute("delete from states where id = ?", (state_id,))


def list_transitions(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all(
        """
        select id, state_id, signal_id, next_state_id, priority, is_enabled, guard_key,
               action_key, match_any_state, notes, created_at, updated_at
        from transitions
        order by priority, id
        limit ?
        """,
        (limit,),
    )


def get_transition(transition_id: int) -> dict[str, Any] | None:
    return _fetch_one(
        """
        select id, state_id, signal_id, next_state_id, priority, is_enabled, guard_key,
               action_key, match_any_state, notes, created_at, updated_at
        from transitions where id = ?
        """,
        (transition_id,),
    )


def create_transition(payload: dict[str, Any]) -> None:
    now = _timestamp()
    _execute(
        """
        insert into transitions (
            state_id, signal_id, next_state_id, priority, is_enabled, guard_key, action_key,
            match_any_state, notes, created_at, updated_at
        )
        values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("state_id"),
            payload.get("signal_id"),
            payload.get("next_state_id"),
            payload.get("priority", 100),
            payload.get("is_enabled", 1),
            payload.get("guard_key"),
            payload.get("action_key"),
            payload.get("match_any_state", 0),
            payload.get("notes"),
            now,
            now,
        ),
    )


def update_transition(transition_id: int, payload: dict[str, Any]) -> None:
    now = _timestamp()
    _execute(
        """
        update transitions
        set state_id = ?, signal_id = ?, next_state_id = ?, priority = ?, is_enabled = ?,
            guard_key = ?, action_key = ?, match_any_state = ?, notes = ?, updated_at = ?
        where id = ?
        """,
        (
            payload.get("state_id"),
            payload.get("signal_id"),
            payload.get("next_state_id"),
            payload.get("priority", 100),
            payload.get("is_enabled", 1),
            payload.get("guard_key"),
            payload.get("action_key"),
            payload.get("match_any_state", 0),
            payload.get("notes"),
            now,
            transition_id,
        ),
    )


def delete_transition(transition_id: int) -> None:
    _execute("delete from transitions where id = ?", (transition_id,))


def list_signal_queue(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all(
        """
        select id, signal_id, signal_type, payload, source, created_at, processed_at, error, durable
        from signal_queue
        order by created_at desc
        limit ?
        """,
        (limit,),
    )


def _get_connection() -> sqlite3.Connection:
    path = Path(os.getenv("NERVE_DB_PATH", DEFAULT_NERVE_DB))
    return sqlite3.connect(path)


def _fetch_all(query: str, params: tuple) -> list[dict[str, Any]]:
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(query, params)
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def _fetch_one(query: str, params: tuple) -> dict[str, Any] | None:
    conn = _get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.execute(query, params)
    row = cursor.fetchone()
    conn.close()
    return dict(row) if row else None


def _execute(query: str, params: tuple) -> None:
    conn = _get_connection()
    conn.execute(query, params)
    conn.commit()
    conn.close()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()
