from __future__ import annotations

import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path
from alphonse.agent.observability.log_manager import get_log_manager

_LOG = get_log_manager()


def write_trace(payload: dict[str, Any]) -> None:
    db_path = resolve_nervous_system_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO fsm_trace
              (correlation_id, state_before, signal_type, transition_id, action_key, state_after, result, error_summary)
            VALUES
              (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload.get("correlation_id"),
                payload.get("state_before"),
                payload.get("signal_type"),
                payload.get("transition_id"),
                payload.get("action_key"),
                payload.get("state_after"),
                payload.get("result"),
                payload.get("error_summary"),
            ),
        )
        conn.commit()
    _LOG.emit(
        event="fsm.trace.written",
        component="trace_store",
        correlation_id=str(payload.get("correlation_id") or "") or None,
        status=str(payload.get("state_before") or "") or None,
        payload={
            "signal_type": payload.get("signal_type"),
            "action_key": payload.get("action_key"),
            "state_before": payload.get("state_before"),
            "state_after": payload.get("state_after"),
            "result": payload.get("result"),
        },
    )
