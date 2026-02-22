from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.services.job_store import JobStore
from alphonse.agent.services.scheduled_jobs_reconciler import ScheduledJobsReconciler


def test_reconciler_removes_stale_scheduled_jobs_rows(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    store = JobStore(root=tmp_path / "jobs")
    store.create_job(
        user_id="u1",
        payload={
            "name": "Real job",
            "schedule": {
                "type": "rrule",
                "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                "rrule": "FREQ=DAILY;BYHOUR=7;BYMINUTE=0",
            },
            "payload_type": "internal_event",
            "payload": {"event_name": "ok"},
            "timezone": "UTC",
        },
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO scheduled_jobs (id, name, prompt, owner_id, status, next_run_at, timezone, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (
                "job_stale_x",
                "Stale job",
                "",
                "missing_owner",
                "active",
                datetime.now(timezone.utc).isoformat(),
                "UTC",
            ),
        )
        conn.commit()
    summary = ScheduledJobsReconciler(store=store).reconcile()
    assert int(summary.get("stale_removed") or 0) >= 1
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT id FROM scheduled_jobs WHERE id = 'job_stale_x'").fetchone()
    assert row is None


def test_reconciler_executes_due_job_within_grace(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("JOB_RECONCILIATION_CATCHUP_GRACE_SECONDS", "999999")
    store = JobStore(root=tmp_path / "jobs")
    created = store.create_job(
        user_id="u1",
        payload={
            "name": "Due now",
            "schedule": {
                "type": "rrule",
                "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat(),
                "rrule": "FREQ=MINUTELY;INTERVAL=1",
            },
            "payload_type": "internal_event",
            "payload": {"event_name": "reconcile.run"},
            "timezone": "UTC",
        },
    )
    created.next_run_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    store.save_job(user_id="u1", job=created)
    summary = ScheduledJobsReconciler(store=store).reconcile()
    assert int(summary.get("executed") or 0) >= 1
    executions = store.list_executions(user_id="u1", job_id=created.job_id, limit=5)
    assert executions
