from __future__ import annotations

import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path


def list_signals(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM signals ORDER BY id ASC LIMIT ?", (limit,))


def list_states(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM states ORDER BY id ASC LIMIT ?", (limit,))


def list_transitions(limit: int = 200) -> list[dict[str, Any]]:
    query = """
        SELECT t.*, s.key AS state_key, sig.key AS signal_key, ns.key AS next_state_key
        FROM transitions t
        JOIN states s ON s.id = t.state_id
        JOIN signals sig ON sig.id = t.signal_id
        JOIN states ns ON ns.id = t.next_state_id
        ORDER BY t.id ASC
        LIMIT ?
    """
    return _fetch_all(query, (limit,))


def list_signal_queue(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all(
        "SELECT * FROM signal_queue ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )


def list_senses(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM senses ORDER BY id ASC LIMIT ?", (limit,))


def list_trace(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all(
        "SELECT * FROM fsm_trace ORDER BY ts DESC LIMIT ?",
        (limit,),
    )


def get_signal(signal_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM signals WHERE id = ?", (signal_id,))


def get_state(state_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM states WHERE id = ?", (state_id,))


def get_transition(transition_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM transitions WHERE id = ?", (transition_id,))


def get_sense(sense_id: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM senses WHERE id = ?", (sense_id,))


def create_signal(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO signals (key, name, source, description, is_enabled)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            payload.get("key"),
            payload.get("name"),
            payload.get("source", "system"),
            payload.get("description"),
            _bool(payload.get("is_enabled", True)),
        ),
    )
    return payload


def create_state(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO states (key, name, description, is_terminal, is_enabled)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            payload.get("key"),
            payload.get("name"),
            payload.get("description"),
            _bool(payload.get("is_terminal", False)),
            _bool(payload.get("is_enabled", True)),
        ),
    )
    return payload


def create_transition(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO transitions
          (state_id, signal_id, next_state_id, priority, is_enabled, guard_key, action_key, match_any_state, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("state_id"),
            payload.get("signal_id"),
            payload.get("next_state_id"),
            payload.get("priority", 100),
            _bool(payload.get("is_enabled", True)),
            payload.get("guard_key"),
            payload.get("action_key"),
            _bool(payload.get("match_any_state", False)),
            payload.get("notes"),
        ),
    )
    return payload


def create_sense(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO senses (key, name, description, source_type, enabled, owner)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            payload.get("key"),
            payload.get("name"),
            payload.get("description"),
            payload.get("source_type"),
            _bool(payload.get("enabled", True)),
            payload.get("owner"),
        ),
    )
    return payload


def update_signal(signal_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    _execute(
        """
        UPDATE signals
        SET key = ?, name = ?, source = ?, description = ?, is_enabled = ?, updated_at = datetime('now')
        WHERE id = ?
        """,
        (
            payload.get("key"),
            payload.get("name"),
            payload.get("source"),
            payload.get("description"),
            _bool(payload.get("is_enabled", True)),
            signal_id,
        ),
    )
    return get_signal(signal_id)


def update_state(state_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    _execute(
        """
        UPDATE states
        SET key = ?, name = ?, description = ?, is_terminal = ?, is_enabled = ?, updated_at = datetime('now')
        WHERE id = ?
        """,
        (
            payload.get("key"),
            payload.get("name"),
            payload.get("description"),
            _bool(payload.get("is_terminal", False)),
            _bool(payload.get("is_enabled", True)),
            state_id,
        ),
    )
    return get_state(state_id)


def update_transition(transition_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    _execute(
        """
        UPDATE transitions
        SET state_id = ?, signal_id = ?, next_state_id = ?, priority = ?, is_enabled = ?,
            guard_key = ?, action_key = ?, match_any_state = ?, notes = ?, updated_at = datetime('now')
        WHERE id = ?
        """,
        (
            payload.get("state_id"),
            payload.get("signal_id"),
            payload.get("next_state_id"),
            payload.get("priority", 100),
            _bool(payload.get("is_enabled", True)),
            payload.get("guard_key"),
            payload.get("action_key"),
            _bool(payload.get("match_any_state", False)),
            payload.get("notes"),
            transition_id,
        ),
    )
    return get_transition(transition_id)


def update_sense(sense_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    _execute(
        """
        UPDATE senses
        SET key = ?, name = ?, description = ?, source_type = ?, enabled = ?, owner = ?, updated_at = datetime('now')
        WHERE id = ?
        """,
        (
            payload.get("key"),
            payload.get("name"),
            payload.get("description"),
            payload.get("source_type"),
            _bool(payload.get("enabled", True)),
            payload.get("owner"),
            sense_id,
        ),
    )
    return get_sense(sense_id)


def delete_signal(signal_id: str) -> dict[str, Any] | None:
    return _delete("signals", signal_id)


def delete_state(state_id: str) -> dict[str, Any] | None:
    return _delete("states", state_id)


def delete_transition(transition_id: str) -> dict[str, Any] | None:
    return _delete("transitions", transition_id)


def delete_sense(sense_id: str) -> dict[str, Any] | None:
    return _delete("senses", sense_id)


def resolve_transition(state_id: int, signal_id: int) -> dict[str, Any] | None:
    query = """
        SELECT t.*, s.key AS state_key, sig.key AS signal_key, ns.key AS next_state_key
        FROM transitions t
        JOIN states s ON s.id = t.state_id
        JOIN signals sig ON sig.id = t.signal_id
        JOIN states ns ON ns.id = t.next_state_id
        WHERE t.is_enabled = 1
          AND t.signal_id = ?
          AND (
                (t.match_any_state = 0 AND t.state_id = ?)
                OR
                (t.match_any_state = 1)
              )
        ORDER BY t.match_any_state ASC, t.priority ASC, t.id ASC
        LIMIT 1
    """
    return _fetch_one(query, (signal_id, state_id))


def _fetch_all(query: str, params: tuple | None = None) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(query, params or ()).fetchall()
    return [_row_to_dict(row) for row in rows]


def _fetch_one(query: str, params: tuple | None = None) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(query, params or ()).fetchone()
    return _row_to_dict(row) if row else None


def _execute(query: str, params: tuple) -> None:
    with _connect() as conn:
        conn.execute(query, params)
        conn.commit()


def _delete(table: str, record_id: str) -> dict[str, Any] | None:
    record = _fetch_one(f"SELECT * FROM {table} WHERE id = ?", (record_id,))
    if not record:
        return None
    _execute(f"DELETE FROM {table} WHERE id = ?", (record_id,))
    return record


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any]:
    if row is None:
        return {}
    if isinstance(row, sqlite3.Row):
        return dict(row)
    return dict(row)


def _connect() -> sqlite3.Connection:
    path = resolve_nervous_system_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _bool(value: object) -> int:
    if isinstance(value, str):
        return 1 if value.lower() in {"1", "true", "yes", "on"} else 0
    return 1 if bool(value) else 0
