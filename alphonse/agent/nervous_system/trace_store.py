from __future__ import annotations

import logging
import sqlite3
from typing import Any

from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

logger = logging.getLogger(__name__)


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
    logger.info("Trace written signal=%s state=%s action=%s", payload.get("signal_type"), payload.get("state_before"), payload.get("action_key"))
