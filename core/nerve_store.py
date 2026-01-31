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


def list_plan_kinds(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all("SELECT * FROM plan_kinds ORDER BY plan_kind ASC LIMIT ?", (limit,))


def list_plan_kind_versions(plan_kind: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
    if plan_kind:
        return _fetch_all(
            "SELECT * FROM plan_kind_versions WHERE plan_kind = ? ORDER BY plan_version ASC LIMIT ?",
            (plan_kind, limit),
        )
    return _fetch_all(
        "SELECT * FROM plan_kind_versions ORDER BY plan_kind ASC, plan_version ASC LIMIT ?",
        (limit,),
    )


def list_plan_executors(limit: int = 200) -> list[dict[str, Any]]:
    return _fetch_all(
        "SELECT * FROM plan_executors ORDER BY plan_kind ASC, plan_version ASC LIMIT ?",
        (limit,),
    )


def list_plan_instances(limit: int = 200, correlation_id: str | None = None) -> list[dict[str, Any]]:
    if correlation_id:
        return _fetch_all(
            """
            SELECT * FROM plan_instances
            WHERE correlation_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (correlation_id, limit),
        )
    return _fetch_all(
        "SELECT * FROM plan_instances ORDER BY created_at DESC LIMIT ?",
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


def get_plan_kind(plan_kind: str) -> dict[str, Any] | None:
    return _fetch_one("SELECT * FROM plan_kinds WHERE plan_kind = ?", (plan_kind,))


def get_plan_kind_version(plan_kind: str, plan_version: int) -> dict[str, Any] | None:
    return _fetch_one(
        "SELECT * FROM plan_kind_versions WHERE plan_kind = ? AND plan_version = ?",
        (plan_kind, plan_version),
    )


def get_plan_executor(plan_kind: str, plan_version: int) -> dict[str, Any] | None:
    return _fetch_one(
        "SELECT * FROM plan_executors WHERE plan_kind = ? AND plan_version = ?",
        (plan_kind, plan_version),
    )


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


def create_plan_kind(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO plan_kinds (plan_kind, description, is_enabled)
        VALUES (?, ?, ?)
        """,
        (
            payload.get("plan_kind"),
            payload.get("description"),
            _bool(payload.get("is_enabled", True)),
        ),
    )
    return payload


def create_plan_kind_version(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO plan_kind_versions (plan_kind, plan_version, json_schema, example, is_deprecated)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            payload.get("plan_kind"),
            payload.get("plan_version"),
            payload.get("json_schema"),
            payload.get("example"),
            _bool(payload.get("is_deprecated", False)),
        ),
    )
    return payload


def create_plan_executor(payload: dict[str, Any]) -> dict[str, Any]:
    _execute(
        """
        INSERT INTO plan_executors (plan_kind, plan_version, executor_key, min_agent_version)
        VALUES (?, ?, ?, ?)
        """,
        (
            payload.get("plan_kind"),
            payload.get("plan_version"),
            payload.get("executor_key"),
            payload.get("min_agent_version"),
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


def update_plan_kind(plan_kind: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    _execute(
        """
        UPDATE plan_kinds
        SET description = ?, is_enabled = ?, created_at = created_at
        WHERE plan_kind = ?
        """,
        (
            payload.get("description"),
            _bool(payload.get("is_enabled", True)),
            plan_kind,
        ),
    )
    return get_plan_kind(plan_kind)


def update_plan_kind_version(plan_kind: str, plan_version: int, payload: dict[str, Any]) -> dict[str, Any] | None:
    _execute(
        """
        UPDATE plan_kind_versions
        SET json_schema = ?, example = ?, is_deprecated = ?, created_at = created_at
        WHERE plan_kind = ? AND plan_version = ?
        """,
        (
            payload.get("json_schema"),
            payload.get("example"),
            _bool(payload.get("is_deprecated", False)),
            plan_kind,
            plan_version,
        ),
    )
    return get_plan_kind_version(plan_kind, plan_version)


def update_plan_executor(plan_kind: str, plan_version: int, payload: dict[str, Any]) -> dict[str, Any] | None:
    _execute(
        """
        UPDATE plan_executors
        SET executor_key = ?, min_agent_version = ?
        WHERE plan_kind = ? AND plan_version = ?
        """,
        (
            payload.get("executor_key"),
            payload.get("min_agent_version"),
            plan_kind,
            plan_version,
        ),
    )
    return get_plan_executor(plan_kind, plan_version)


def delete_signal(signal_id: str) -> dict[str, Any] | None:
    return _delete("signals", signal_id)


def delete_state(state_id: str) -> dict[str, Any] | None:
    return _delete("states", state_id)


def delete_transition(transition_id: str) -> dict[str, Any] | None:
    return _delete("transitions", transition_id)


def delete_sense(sense_id: str) -> dict[str, Any] | None:
    return _delete("senses", sense_id)


def delete_plan_kind(plan_kind: str) -> dict[str, Any] | None:
    record = get_plan_kind(plan_kind)
    if not record:
        return None
    _execute("DELETE FROM plan_kinds WHERE plan_kind = ?", (plan_kind,))
    return record


def delete_plan_kind_version(plan_kind: str, plan_version: int) -> dict[str, Any] | None:
    record = get_plan_kind_version(plan_kind, plan_version)
    if not record:
        return None
    _execute(
        "DELETE FROM plan_kind_versions WHERE plan_kind = ? AND plan_version = ?",
        (plan_kind, plan_version),
    )
    return record


def delete_plan_executor(plan_kind: str, plan_version: int) -> dict[str, Any] | None:
    record = get_plan_executor(plan_kind, plan_version)
    if not record:
        return None
    _execute(
        "DELETE FROM plan_executors WHERE plan_kind = ? AND plan_version = ?",
        (plan_kind, plan_version),
    )
    return record


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
