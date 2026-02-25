from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import (
    acquire_pdca_task_lease,
    append_pdca_event,
    get_pdca_task,
    list_pdca_events,
    list_runnable_pdca_tasks,
    load_pdca_checkpoint,
    release_pdca_task_lease,
    save_pdca_checkpoint,
    upsert_pdca_task,
)


def test_pdca_task_upsert_and_runnable_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    now = datetime.now(timezone.utc)
    due = (now - timedelta(seconds=5)).isoformat()
    future = (now + timedelta(hours=1)).isoformat()

    low_id = upsert_pdca_task(
        {
            "owner_id": "u1",
            "conversation_key": "chat-1",
            "priority": 10,
            "status": "queued",
            "next_run_at": due,
        }
    )
    high_id = upsert_pdca_task(
        {
            "owner_id": "u2",
            "conversation_key": "chat-2",
            "priority": 100,
            "status": "queued",
            "next_run_at": due,
        }
    )
    _ = upsert_pdca_task(
        {
            "owner_id": "u3",
            "conversation_key": "chat-3",
            "priority": 999,
            "status": "queued",
            "next_run_at": future,
        }
    )

    runnable = list_runnable_pdca_tasks(now=now.isoformat(), limit=10)
    ids = [row["task_id"] for row in runnable]
    assert ids == [high_id, low_id]

    first = get_pdca_task(high_id)
    assert isinstance(first, dict)
    assert first["priority"] == 100
    assert first["status"] == "queued"


def test_pdca_task_lease_acquire_and_release(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "u1",
            "conversation_key": "chat-lease",
            "status": "queued",
        }
    )

    assert acquire_pdca_task_lease(task_id=task_id, worker_id="w1", lease_seconds=30) is True
    assert acquire_pdca_task_lease(task_id=task_id, worker_id="w2", lease_seconds=30) is False

    leased = get_pdca_task(task_id)
    assert leased is not None
    assert leased["worker_id"] == "w1"
    assert leased["status"] == "running"

    assert release_pdca_task_lease(task_id=task_id, worker_id="w2") is False
    assert release_pdca_task_lease(task_id=task_id, worker_id="w1") is True

    released = get_pdca_task(task_id)
    assert released is not None
    assert released["worker_id"] is None
    assert released["lease_until"] is None


def test_pdca_checkpoint_versioning_and_events(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "u1",
            "conversation_key": "chat-checkpoint",
            "status": "queued",
        }
    )

    v1 = save_pdca_checkpoint(task_id=task_id, state={"a": 1}, task_state={"cycle_index": 1}, expected_version=0)
    assert v1 == 1

    loaded1 = load_pdca_checkpoint(task_id)
    assert loaded1 is not None
    assert loaded1["version"] == 1
    assert loaded1["task_state"]["cycle_index"] == 1

    stale = save_pdca_checkpoint(task_id=task_id, state={"a": 2}, task_state={"cycle_index": 2}, expected_version=0)
    assert stale is None

    v2 = save_pdca_checkpoint(task_id=task_id, state={"a": 2}, task_state={"cycle_index": 2}, expected_version=1)
    assert v2 == 2

    loaded2 = load_pdca_checkpoint(task_id)
    assert loaded2 is not None
    assert loaded2["version"] == 2
    assert loaded2["state"]["a"] == 2

    event_id = append_pdca_event(
        task_id=task_id,
        event_type="slice.persisted",
        payload={"version": v2},
        correlation_id="cid-1",
    )
    assert event_id
    events = list_pdca_events(task_id=task_id, limit=10)
    assert len(events) == 1
    assert events[0]["event_type"] == "slice.persisted"
    assert events[0]["payload"]["version"] == 2
