from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent.actions.handle_timer_fired import HandleTimerFiredAction
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.senses.bus import Signal
from alphonse.agent.services.job_store import JobStore


class _FakeBus:
    def __init__(self) -> None:
        self.events: list[Signal] = []

    def emit(self, signal: Signal) -> None:
        self.events.append(signal)


def test_job_create_compiles_job_trigger_timed_signal(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    store = JobStore(root=tmp_path / "jobs")
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
            "payload": {"prompt_text": "Share USD to MXN update"},
            "timezone": "UTC",
        },
    )
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT id, owner_id, status FROM scheduled_jobs WHERE id = ?",
            (created.job_id,),
        ).fetchone()
    assert row is not None
    assert str(row[0]) == created.job_id
    assert str(row[1]) == "u1"
    assert str(row[2]) == "active"


def test_timer_fired_job_trigger_emits_conscious_message_event(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    store = JobStore(root=tmp_path / "jobs")
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
            "payload": {"prompt_text": "Share USD to MXN update"},
            "timezone": "UTC",
        },
    )

    # Force action runner to use the test store instance.
    import alphonse.agent.actions.handle_timer_fired as handle_timer_fired_module

    monkeypatch.setattr(handle_timer_fired_module, "JobStore", lambda: store)
    bus = _FakeBus()
    action = HandleTimerFiredAction()
    signal = Signal(
        type="timed_signal.fired",
        payload={
            "timed_signal_id": f"job_trigger:{created.job_id}",
            "kind": "job_trigger",
            "mind_layer": "subconscious",
            "dispatch_mode": "deterministic",
            "job_id": created.job_id,
            "target": "u1",
            "payload": {"kind": "job_trigger", "job_id": created.job_id, "user_id": "u1"},
        },
        source="timer",
        correlation_id=created.job_id,
    )
    action.execute({"signal": signal, "ctx": bus, "state": None, "outcome": None})
    assert bus.events
    emitted = bus.events[-1]
    assert emitted.type == "api.message_received"
    text = str((emitted.payload or {}).get("text") or "")
    assert text
