from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.services.job_models import JobSpec
from alphonse.agent.services.job_runner import JobRunner, route_job
from alphonse.agent.services.job_store import JobStore, compute_next_run_at


def test_rrule_next_run_basic() -> None:
    after = datetime(2026, 2, 17, 15, 0, tzinfo=timezone.utc)
    next_run = compute_next_run_at(
        schedule={
            "type": "rrule",
            "dtstart": "2026-02-17T09:00:00-06:00",
            "rrule": "FREQ=WEEKLY;BYDAY=TU;BYHOUR=9;BYMINUTE=0",
        },
        timezone_name="America/Mexico_City",
        after=after,
    )
    assert isinstance(next_run, str)
    assert next_run


def test_job_store_create_and_persist(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    created = store.create_job(
        user_id="u1",
        payload={
            "name": "Weekly chores",
            "description": "Family chores reminder",
            "schedule": {
                "type": "rrule",
                "dtstart": "2026-02-17T09:00:00-06:00",
                "rrule": "FREQ=WEEKLY;BYDAY=TU;BYHOUR=9;BYMINUTE=0",
            },
            "payload_type": "prompt_to_brain",
            "payload": {"prompt_text": "Prepare chores plan"},
            "timezone": "America/Mexico_City",
        },
    )
    again = store.get_job(user_id="u1", job_id=created.job_id)
    assert again.job_id == created.job_id
    assert again.next_run_at
    assert (tmp_path / "data" / "jobs" / "u1" / "jobs.json").exists()


def test_runner_executes_due_job(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    runner = JobRunner(
        job_store=store,
        tick_seconds=5,
    )
    events: list[dict[str, object]] = []
    created = store.create_job(
        user_id="u1",
        payload={
            "name": "Immediate conscious job",
            "description": "Run conscious prompt through brain sink",
            "schedule": {
                "type": "rrule",
                "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(),
                "rrule": "FREQ=DAILY;INTERVAL=1",
            },
            "payload_type": "prompt_to_brain",
            "payload": {"prompt_text": "Send the daily update summary."},
            "timezone": "UTC",
            "safety_level": "low",
            "requires_confirmation": False,
        },
    )
    outcome = runner.run_job_now(
        user_id="u1",
        job_id=created.job_id,
        brain_event_sink=lambda payload: events.append(payload),
    )
    assert outcome["status"] == "ok"
    assert events
    executions = store.list_executions(user_id="u1", job_id=created.job_id, limit=5)
    assert executions


def test_runner_rejects_non_conscious_job_payload_type(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    user_dir = tmp_path / "data" / "jobs" / "u1"
    user_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        "jobs": {
            "job_legacy_1": {
                "job_id": "job_legacy_1",
                "name": "Legacy tool payload",
                "description": "legacy",
                "enabled": True,
                "schedule": {
                    "type": "rrule",
                    "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                    "rrule": "FREQ=DAILY;INTERVAL=1",
                },
                "timezone": "UTC",
                "payload_type": "tool_call",
                "payload": {"tool_call": {"kind": "call_tool", "tool_name": "dummy_tool", "args": {"value": "ok"}}},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        }
    }
    (user_dir / "jobs.json").write_text(json.dumps(raw), encoding="utf-8")
    runner = JobRunner(job_store=store, tick_seconds=5)
    outcome = runner.run_job_now(user_id="u1", job_id="job_legacy_1")
    assert outcome["status"] == "error"
    executions = store.list_executions(user_id="u1", job_id="job_legacy_1", limit=1)
    assert executions
    assert "jobs_conscious_only_policy_violation" in str(executions[0].output_summary or "")


def test_router_sends_to_brain_for_high_safety() -> None:
    job = JobSpec(
        job_id="job_x",
        name="High risk",
        description="Needs confirmation",
        enabled=True,
        schedule={"type": "rrule", "dtstart": "2026-02-17T09:00:00+00:00", "rrule": "FREQ=DAILY"},
        timezone="UTC",
        payload_type="prompt_to_brain",
        payload={"prompt_text": "Send the daily update."},
        safety_level="high",
        requires_confirmation=False,
    )
    decision = route_job(job=job, auto_execute_high_risk=False)
    assert decision.route == "brain"


def test_job_store_normalizes_missing_rrule_fields_and_syncs_timed_signal(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    store = JobStore(root=tmp_path / "data" / "jobs")
    created = store.create_job(
        user_id="u1",
        payload={
            "name": "Legacy-like daily",
            "description": "Created without explicit type and dtstart",
            "schedule": {"rrule": "FREQ=DAILY;BYHOUR=7;BYMINUTE=0", "timezone": "America/Mexico_City"},
            "payload_type": "prompt_to_brain",
            "payload": {"prompt_text": "Ping me"},
            "timezone": "UTC",
        },
    )
    assert created.schedule.get("type") == "rrule"
    assert str(created.schedule.get("dtstart") or "").strip()
    assert created.next_run_at
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        compiled = conn.execute(
            "SELECT id, owner_id, status FROM scheduled_jobs WHERE id = ?",
            (created.job_id,),
        ).fetchone()
        timed = conn.execute(
            "SELECT id, status, target, origin FROM timed_signals WHERE id = ?",
            (f"job_trigger:{created.job_id}",),
        ).fetchone()
    assert compiled is not None
    assert str(compiled[0]) == created.job_id
    assert str(compiled[1]) == "u1"
    assert str(compiled[2]) == "active"
    assert timed is not None
    assert str(timed[0]) == f"job_trigger:{created.job_id}"
    assert str(timed[1]) == "pending"


def test_job_store_normalizes_legacy_prompt_payload_type(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    store = JobStore(root=tmp_path / "data" / "jobs")
    created = store.create_job(
        user_id="u1",
        payload={
            "name": "Legacy prompt payload",
            "description": "Should map prompt -> prompt_to_brain",
            "schedule": {
                "rrule": "FREQ=DAILY;BYHOUR=7;BYMINUTE=0",
                "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            },
            "payload_type": "prompt",
            "payload": {"text": "Send rate"},
            "timezone": "UTC",
        },
    )
    assert created.payload_type == "prompt_to_brain"


def test_job_store_rejects_legacy_tool_call_shape_for_new_writes(tmp_path: Path) -> None:
    store = JobStore(root=tmp_path / "data" / "jobs")
    try:
        store.create_job(
            user_id="u1",
            payload={
                "name": "Legacy tool payload",
                "description": "Should reject tool_name/tool_key legacy shape",
                "schedule": {
                    "type": "rrule",
                    "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                    "rrule": "FREQ=DAILY;INTERVAL=1",
                },
                "payload_type": "tool_call",
                "payload": {"tool_name": "dummy_tool", "args": {"value": "ok"}},
                "timezone": "UTC",
            },
        )
        assert False, "expected strict contract validation to reject legacy tool_call shape"
    except ValueError as exc:
        assert str(exc) == "jobs_conscious_only_payload_type"


def test_job_store_backfill_deletes_legacy_non_conscious_jobs(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    store = JobStore(root=tmp_path / "data" / "jobs")
    user_dir = tmp_path / "data" / "jobs" / "u1"
    user_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        "jobs": {
            "job_legacy_1": {
                "job_id": "job_legacy_1",
                "name": "Legacy tool payload",
                "description": "legacy",
                "enabled": True,
                "schedule": {
                    "type": "rrule",
                    "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                    "rrule": "FREQ=DAILY;INTERVAL=1",
                },
                "timezone": "UTC",
                "payload_type": "tool_call",
                "payload": {"tool_name": "dummy_tool", "args": {"value": "ok"}},
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        }
    }
    (user_dir / "jobs.json").write_text(json.dumps(raw), encoding="utf-8")
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO scheduled_jobs (id, name, prompt, owner_id, status, next_run_at, timezone, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            ("job_legacy_1", "legacy", "", "u1", "active", datetime.now(timezone.utc).isoformat(), "UTC"),
        )
        conn.execute(
            """
            INSERT INTO timed_signals (id, trigger_at, timezone, status, fired_at, signal_type, payload, target, origin, correlation_id, created_at, updated_at)
            VALUES (?, ?, ?, 'pending', NULL, 'timed_signal', ?, ?, ?, ?, datetime('now'), datetime('now'))
            """,
            (
                "job_trigger:job_legacy_1",
                datetime.now(timezone.utc).isoformat(),
                "UTC",
                json.dumps({"kind": "job_trigger", "job_id": "job_legacy_1", "payload_type": "tool_call", "payload": {}}),
                "u1",
                "tests",
                "corr-legacy-1",
            ),
        )
        conn.commit()
    summary = store.backfill_and_sync_jobs(user_id="u1")
    assert int(summary.get("deleted") or 0) == 1
    try:
        store.get_job(user_id="u1", job_id="job_legacy_1")
        assert False, "expected legacy job to be deleted"
    except ValueError:
        pass
    with sqlite3.connect(db_path) as conn:
        scheduled = conn.execute("SELECT id FROM scheduled_jobs WHERE id = 'job_legacy_1'").fetchone()
        timed = conn.execute("SELECT id FROM timed_signals WHERE id = 'job_trigger:job_legacy_1'").fetchone()
    assert scheduled is None
    assert timed is None


def test_job_runner_reschedules_job_trigger_timed_signal_after_run(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    apply_schema(db_path)
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    store = JobStore(root=tmp_path / "data" / "jobs")
    runner = JobRunner(job_store=store, tick_seconds=5)
    created = store.create_job(
        user_id="u1",
        payload={
            "name": "Daily FX update",
            "description": "USD to MXN at 7am",
            "schedule": {
                "type": "rrule",
                "dtstart": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
                "rrule": "FREQ=DAILY;BYHOUR=7;BYMINUTE=0",
            },
            "payload_type": "prompt_to_brain",
            "payload": {"prompt_text": "Share USD to MXN update."},
            "timezone": "UTC",
        },
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE timed_signals SET status = 'fired', fired_at = datetime('now') WHERE id = ?",
            (f"job_trigger:{created.job_id}",),
        )
        conn.commit()
    events: list[dict[str, object]] = []
    outcome = runner.run_job_now(
        user_id="u1",
        job_id=created.job_id,
        brain_event_sink=lambda payload: events.append(payload),
    )
    assert outcome["status"] == "ok"
    assert events
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status, trigger_at, fired_at FROM timed_signals WHERE id = ?",
            (f"job_trigger:{created.job_id}",),
        ).fetchone()
    assert row is not None
    assert str(row[0]) == "pending"
    assert str(row[1]).strip()
    assert row[2] is None
