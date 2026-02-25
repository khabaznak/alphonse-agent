from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_task, list_pdca_events, upsert_pdca_task
from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.services.pdca_queue_runner import PdcaQueueRunner


def test_pdca_queue_runner_emits_slice_request_and_cooldown(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
        poll_seconds=0.5,
        lease_seconds=10,
        dispatch_cooldown_seconds=60,
        worker_id="worker-test",
    )
    dispatched = runner.run_once(now=now.isoformat())
    assert dispatched == 1

    signal = bus.get(timeout=0.1)
    assert signal is not None
    assert signal.type == "pdca.slice.requested"
    assert signal.payload.get("task_id") == task_id
    assert signal.payload.get("owner_id") == "owner-1"

    updated = get_pdca_task(task_id)
    assert updated is not None
    assert updated["status"] == "running"
    assert updated["worker_id"] is None
    assert updated["lease_until"] is None
    assert isinstance(updated.get("next_run_at"), str) and updated["next_run_at"]

    events = list_pdca_events(task_id=task_id, limit=10)
    assert len(events) == 1
    assert events[0]["event_type"] == "slice.requested"


def test_pdca_queue_runner_noop_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    monkeypatch.setenv("ALPHONSE_PDCA_SLICING_ENABLED", "false")

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


def test_pdca_queue_runner_avoids_double_dispatch_under_contention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    now = datetime.now(timezone.utc)
    _ = upsert_pdca_task(
        {
            "owner_id": "owner-3",
            "conversation_key": "chat-3",
            "status": "queued",
            "next_run_at": (now - timedelta(seconds=1)).isoformat(),
        }
    )

    bus = Bus()
    runner_a = PdcaQueueRunner(
        bus=bus,
        enabled=True,
        lease_seconds=30,
        dispatch_cooldown_seconds=60,
        worker_id="worker-a",
    )
    runner_b = PdcaQueueRunner(
        bus=bus,
        enabled=True,
        lease_seconds=30,
        dispatch_cooldown_seconds=60,
        worker_id="worker-b",
    )
    first = runner_a.run_once(now=now.isoformat())
    second = runner_b.run_once(now=now.isoformat())
    assert first == 1
    assert second == 0
    signal = bus.get(timeout=0.1)
    assert signal is not None
    assert signal.type == "pdca.slice.requested"
    assert bus.get(timeout=0.05) is None


def test_pdca_queue_runner_skips_leased_tasks_until_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    now = datetime.now(timezone.utc)
    stale_task_id = upsert_pdca_task(
        {
            "owner_id": "owner-stale",
            "conversation_key": "chat-stale",
            "status": "queued",
            "next_run_at": (now - timedelta(seconds=2)).isoformat(),
            "lease_until": (now - timedelta(seconds=1)).isoformat(),
            "worker_id": "other-worker",
        }
    )
    _ = upsert_pdca_task(
        {
            "owner_id": "owner-future",
            "conversation_key": "chat-future",
            "status": "queued",
            "next_run_at": (now - timedelta(seconds=2)).isoformat(),
            "lease_until": (now + timedelta(seconds=120)).isoformat(),
            "worker_id": "other-worker",
        }
    )

    bus = Bus()
    runner = PdcaQueueRunner(
        bus=bus,
        enabled=True,
        poll_seconds=0.5,
        lease_seconds=10,
        dispatch_cooldown_seconds=60,
        worker_id="worker-test",
    )
    dispatched = runner.run_once(now=now.isoformat())
    assert dispatched == 1

    signal = bus.get(timeout=0.1)
    assert signal is not None
    assert signal.type == "pdca.slice.requested"
    assert signal.payload.get("task_id") == stale_task_id
    assert bus.get(timeout=0.05) is None


def test_pdca_queue_runner_rotates_owners_with_equal_priority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    now = datetime.now(timezone.utc)
    task_a1 = upsert_pdca_task(
        {
            "owner_id": "owner-a",
            "conversation_key": "chat-a1",
            "status": "queued",
            "priority": 100,
            "next_run_at": (now - timedelta(seconds=5)).isoformat(),
        }
    )
    _ = upsert_pdca_task(
        {
            "owner_id": "owner-a",
            "conversation_key": "chat-a2",
            "status": "queued",
            "priority": 100,
            "next_run_at": (now - timedelta(seconds=4)).isoformat(),
        }
    )
    task_b1 = upsert_pdca_task(
        {
            "owner_id": "owner-b",
            "conversation_key": "chat-b1",
            "status": "queued",
            "priority": 100,
            "next_run_at": (now - timedelta(seconds=3)).isoformat(),
        }
    )

    bus = Bus()
    runner = PdcaQueueRunner(
        bus=bus,
        enabled=True,
        lease_seconds=10,
        dispatch_cooldown_seconds=1,
        worker_id="worker-fair",
    )
    first = runner.run_once(now=now.isoformat())
    second = runner.run_once(now=(now + timedelta(seconds=2)).isoformat())
    assert first == 1
    assert second == 1

    signal_1 = bus.get(timeout=0.1)
    signal_2 = bus.get(timeout=0.1)
    assert signal_1 is not None and signal_2 is not None
    assert signal_1.payload.get("task_id") == task_a1
    assert signal_2.payload.get("task_id") == task_b1


def test_pdca_queue_runner_prioritizes_interactive_tasks_with_boost(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    now = datetime.now(timezone.utc)
    background_id = upsert_pdca_task(
        {
            "owner_id": "owner-bg",
            "conversation_key": "chat-bg",
            "status": "queued",
            "priority": 100,
            "next_run_at": (now - timedelta(seconds=5)).isoformat(),
            "metadata": {"task_class": "background"},
        }
    )
    interactive_id = upsert_pdca_task(
        {
            "owner_id": "owner-i",
            "conversation_key": "chat-i",
            "status": "queued",
            "priority": 70,
            "next_run_at": (now - timedelta(seconds=2)).isoformat(),
            "metadata": {"task_class": "interactive"},
        }
    )

    bus = Bus()
    runner = PdcaQueueRunner(
        bus=bus,
        enabled=True,
        lease_seconds=10,
        dispatch_cooldown_seconds=60,
        interactive_boost=40,
        worker_id="worker-weighted",
    )
    dispatched = runner.run_once(now=now.isoformat())
    assert dispatched == 1

    signal = bus.get(timeout=0.1)
    assert signal is not None
    assert signal.type == "pdca.slice.requested"
    assert signal.payload.get("task_id") == interactive_id

    background = get_pdca_task(background_id)
    assert background is not None
    assert background["status"] == "queued"


def test_pdca_queue_runner_emits_starvation_warning_event_with_cooldown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    _ = upsert_pdca_task(
        {
            "owner_id": "owner-fresh",
            "conversation_key": "chat-fresh",
            "status": "queued",
            "priority": 100,
            "next_run_at": (now - timedelta(seconds=10)).isoformat(),
        }
    )

    bus = Bus()
    runner = PdcaQueueRunner(
        bus=bus,
        enabled=True,
        lease_seconds=10,
        dispatch_cooldown_seconds=1,
        starvation_warn_seconds=60,
        starvation_warn_cooldown_seconds=120,
        worker_id="worker-health",
    )
    first = runner.run_once(now=now.isoformat())
    second = runner.run_once(now=(now + timedelta(seconds=30)).isoformat())
    assert first == 1
    assert second == 1

    events = list_pdca_events(task_id=starved_id, limit=20)
    warning_events = [item for item in events if item["event_type"] == "queue.starvation_warning"]
    assert len(warning_events) == 1
    payload = warning_events[0]["payload"]
    assert int(payload.get("warn_threshold_seconds") or 0) == 60
    assert int(payload.get("queue_depth") or 0) >= 1
