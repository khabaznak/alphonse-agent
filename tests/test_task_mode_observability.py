from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

from alphonse.agent.cortex.task_mode.observability import log_task_event
from alphonse.agent.nervous_system.paths import resolve_observability_db_path


def test_log_task_event_emits_required_fields(caplog, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_OBSERVABILITY_DB_PATH", str(tmp_path / "obs.db"))
    monkeypatch.setenv("ALPHONSE_OBSERVABILITY_MAINTENANCE_SECONDS", "999999")
    state = {
        "correlation_id": "cid-123",
        "channel_type": "telegram",
        "actor_person_id": "user-1",
    }
    task_state = {"cycle_index": 3, "status": "running"}

    with caplog.at_level(logging.INFO):
        log_task_event(
            logger=logging.getLogger("alphonse.agent.cortex.task_mode.test"),
            state=state,
            task_state=task_state,
            node="progress_critic_node",
            event="graph.state.updated",
            tool="scratchpad_create",
        )

    record = next((r for r in caplog.records if "task_mode_event " in r.message), None)
    assert record is not None
    raw = record.message.split("task_mode_event ", 1)[1]
    payload = json.loads(raw)
    for key in ("ts", "level", "event", "correlation_id", "channel", "user_id", "node", "cycle", "status"):
        assert key in payload
    assert payload["event"] == "graph.state.updated"
    assert payload["tool"] == "scratchpad_create"
    with sqlite3.connect(resolve_observability_db_path()) as conn:
        row = conn.execute("SELECT COUNT(*) FROM trace_events").fetchone()
        assert row is not None
        assert int(row[0]) >= 1
