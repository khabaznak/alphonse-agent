from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import list_pdca_events, upsert_pdca_task
from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.services.pdca_queue_runner import PdcaQueueRunner


def test_pdca_watchdog_emits_dispatch_kick_for_runnable_task(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    now = datetime.now(timezone.utc)
    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-1",
            "status": "queued",
            "next_run_at": (now - timedelta(seconds=1)).isoformat(),
            "priority": 50,
        }
    )

    bus = Bus()
    runner = PdcaQueueRunner(
        bus=bus,
        enabled=True,
        poll_seconds=30,
        worker_id="watchdog-test",
    )
    emitted = runner.run_once(now=now.isoformat())
    assert emitted == 1

    signal = bus.get(timeout=0.1)
    assert signal is not None
    assert signal.type == "pdca.dispatch.kick"
    assert signal.payload.get("task_id") == task_id
    assert signal.payload.get("reason") == "watchdog_recovery"

    events = list_pdca_events(task_id=task_id, limit=10)
    assert any(item["event_type"] == "pdca.dispatch.watchdog_rekick" for item in events)


def test_pdca_watchdog_noop_when_disabled(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_PDCA_SLICING_ENABLED", "false")
    apply_schema(db_path)

    _ = upsert_pdca_task(
        {
            "owner_id": "owner-2",
            "conversation_key": "chat-2",
            "status": "queued",
            "priority": 100,
        }
    )
    bus = Bus()
    runner = PdcaQueueRunner(bus=bus, enabled=None)
    assert runner.enabled is False
    assert runner.run_once() == 0
    assert bus.get(timeout=0.05) is None


def test_pdca_watchdog_enabled_by_default(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.delenv("ALPHONSE_PDCA_SLICING_ENABLED", raising=False)
    apply_schema(db_path)
    bus = Bus()
    runner = PdcaQueueRunner(bus=bus, enabled=None)
    assert runner.enabled is True


def test_pdca_watchdog_uses_defaults_for_missing_optional_timing_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    for name in (
        "ALPHONSE_PDCA_DISPATCH_WATCHDOG_SECONDS",
        "ALPHONSE_PDCA_QUEUE_LEASE_SECONDS",
        "ALPHONSE_PDCA_QUEUE_DISPATCH_COOLDOWN_SECONDS",
        "ALPHONSE_PDCA_QUEUE_INTERACTIVE_BOOST",
        "ALPHONSE_PDCA_QUEUE_STARVATION_WARN_SECONDS",
        "ALPHONSE_PDCA_QUEUE_STARVATION_WARN_COOLDOWN_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    apply_schema(db_path)

    runner = PdcaQueueRunner(bus=Bus(), enabled=True)

    assert runner.enabled is True


def test_pdca_watchdog_emits_starvation_warning_with_cooldown(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    now = datetime.now(timezone.utc)
    starved_id = upsert_pdca_task(
        {
            "owner_id": "owner-starved",
            "conversation_key": "chat-starved",
            "status": "queued",
            "priority": 100,
            "next_run_at": (now - timedelta(seconds=120)).isoformat(),
        }
    )

    bus = Bus()
    runner = PdcaQueueRunner(
        bus=bus,
        enabled=True,
        poll_seconds=30,
        starvation_warn_seconds=60,
        starvation_warn_cooldown_seconds=120,
        worker_id="watchdog-health",
    )
    first = runner.run_once(now=now.isoformat())
    second = runner.run_once(now=(now + timedelta(seconds=30)).isoformat())
    assert first >= 1
    assert second >= 1

    events = list_pdca_events(task_id=starved_id, limit=30)
    warning_events = [item for item in events if item["event_type"] == "queue.starvation_warning"]
    assert len(warning_events) == 1
    payload = warning_events[0]["payload"]
    assert int(payload.get("warn_threshold_seconds") or 0) == 60
    assert int(payload.get("queue_depth") or 0) >= 1
