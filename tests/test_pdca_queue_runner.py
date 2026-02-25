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
