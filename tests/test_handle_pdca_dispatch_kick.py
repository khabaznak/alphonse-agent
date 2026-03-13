from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from alphonse.agent.actions.handle_pdca_dispatch_kick import HandlePdcaDispatchKickAction
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_task, list_pdca_events, upsert_pdca_task
from alphonse.agent.nervous_system.senses.bus import Bus, Signal


def _kick_signal(*, task_id: str, correlation_id: str) -> Signal:
    return Signal(
        type="pdca.dispatch.kick",
        payload={"task_id": task_id, "correlation_id": correlation_id},
        source="test",
        correlation_id=correlation_id,
    )


def test_handle_pdca_dispatch_kick_emits_slice_request_when_eligible(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_PDCA_QUEUE_DISPATCH_COOLDOWN_SECONDS", "10")
    apply_schema(db_path)

    now = datetime.now(timezone.utc)
    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-1",
            "status": "queued",
            "next_run_at": (now - timedelta(seconds=1)).isoformat(),
        }
    )

    action = HandlePdcaDispatchKickAction()
    bus = Bus()
    result = action.execute({"ctx": bus, "signal": _kick_signal(task_id=task_id, correlation_id="kick-1")})
    assert result.intention_key == "NOOP"

    emitted = bus.get(timeout=0.1)
    assert emitted is not None
    assert emitted.type == "pdca.slice.requested"
    assert emitted.payload.get("task_id") == task_id

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    assert task["status"] == "running"
    assert isinstance(task.get("next_run_at"), str) and task["next_run_at"]

    events = list_pdca_events(task_id=task_id, limit=20)
    assert any(item["event_type"] == "pdca.dispatch.kick.received" for item in events)
    assert any(item["event_type"] == "pdca.dispatch.slice_emitted" for item in events)
    assert any(item["event_type"] == "slice.requested" for item in events)


def test_handle_pdca_dispatch_kick_skips_terminal_task(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-1",
            "status": "done",
        }
    )

    action = HandlePdcaDispatchKickAction()
    bus = Bus()
    _ = action.execute({"ctx": bus, "signal": _kick_signal(task_id=task_id, correlation_id="kick-terminal")})
    assert bus.get(timeout=0.05) is None
    events = list_pdca_events(task_id=task_id, limit=20)
    assert any(item["event_type"] == "pdca.dispatch.kick.skipped" for item in events)
    reasons = [str((item.get("payload") or {}).get("reason") or "") for item in events if item["event_type"] == "pdca.dispatch.kick.skipped"]
    assert "terminal_task" in reasons


def test_handle_pdca_dispatch_kick_skips_when_cooldown_not_elapsed(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    future = (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat()
    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-1",
            "status": "queued",
            "next_run_at": future,
        }
    )

    action = HandlePdcaDispatchKickAction()
    bus = Bus()
    _ = action.execute({"ctx": bus, "signal": _kick_signal(task_id=task_id, correlation_id="kick-cooldown")})
    assert bus.get(timeout=0.05) is None
    events = list_pdca_events(task_id=task_id, limit=20)
    reasons = [str((item.get("payload") or {}).get("reason") or "") for item in events if item["event_type"] == "pdca.dispatch.kick.skipped"]
    assert "cooldown_not_elapsed" in reasons

