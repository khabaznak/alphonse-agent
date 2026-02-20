from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent.nervous_system.paths import resolve_observability_db_path
from alphonse.agent.observability.store import run_maintenance
from alphonse.agent.observability.store import write_task_event


def test_write_task_event_persists_and_rollup(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "observability.db"
    monkeypatch.setenv("ALPHONSE_OBSERVABILITY_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_OBSERVABILITY_MAINTENANCE_SECONDS", "999999")

    write_task_event(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": "info",
            "event": "graph.state.updated",
            "correlation_id": "cid-1",
            "channel": "telegram",
            "user_id": "user-1",
            "node": "update_state_node",
            "cycle": 1,
            "status": "running",
            "tool": "job_create",
        }
    )

    with sqlite3.connect(resolve_observability_db_path()) as conn:
        row = conn.execute("SELECT COUNT(*) FROM trace_events").fetchone()
        assert int(row[0]) == 1
        rollup = conn.execute(
            "SELECT count FROM trace_daily_rollups WHERE event=? AND level=?",
            ("graph.state.updated", "info"),
        ).fetchone()
        assert rollup is not None
        assert int(rollup[0]) == 1


def test_observability_maintenance_prunes_ttl_and_enforces_cap(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "observability.db"
    monkeypatch.setenv("ALPHONSE_OBSERVABILITY_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_OBSERVABILITY_MAINTENANCE_SECONDS", "1")
    monkeypatch.setenv("ALPHONSE_OBSERVABILITY_NON_ERROR_TTL_DAYS", "1")
    monkeypatch.setenv("ALPHONSE_OBSERVABILITY_ERROR_TTL_DAYS", "2")
    monkeypatch.setenv("ALPHONSE_OBSERVABILITY_MAX_ROWS", "2")

    write_task_event(
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": "info",
            "event": "seed",
            "correlation_id": "seed",
            "channel": "telegram",
            "user_id": "user-1",
            "node": "seed",
            "cycle": 1,
            "status": "running",
        }
    )

    now = datetime.now(timezone.utc)
    old_info = (now - timedelta(days=3)).isoformat()
    old_error = (now - timedelta(days=3)).isoformat()
    keep_oldest = (now - timedelta(hours=12)).isoformat()
    keep_mid = (now - timedelta(minutes=10)).isoformat()
    keep_newest = (now - timedelta(minutes=5)).isoformat()

    with sqlite3.connect(resolve_observability_db_path()) as conn:
        conn.execute("DELETE FROM trace_events")
        rows = [
            (old_info, "info", "old_info"),
            (old_error, "warning", "old_warning"),
            (keep_oldest, "info", "keep_oldest"),
            (keep_mid, "info", "keep_mid"),
            (keep_newest, "info", "keep_newest"),
        ]
        for created_at, level, event in rows:
            conn.execute(
                """
                INSERT INTO trace_events (
                  created_at, level, event, correlation_id, channel, user_id, node, cycle, status, detail_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (created_at, level, event, "cid", "telegram", "user-1", "node", 1, "running", "{}"),
            )
        conn.commit()

    run_maintenance(force=True)

    with sqlite3.connect(resolve_observability_db_path()) as conn:
        kept = conn.execute(
            "SELECT event FROM trace_events ORDER BY created_at ASC"
        ).fetchall()
    kept_events = [str(row[0]) for row in kept]
    assert "old_info" not in kept_events
    assert "old_warning" not in kept_events
    assert kept_events == ["keep_mid", "keep_newest"]
