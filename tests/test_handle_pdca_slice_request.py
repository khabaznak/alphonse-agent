from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.actions import handle_pdca_slice_request as hpsr
from alphonse.agent.actions.handle_pdca_slice_request import HandlePdcaSliceRequestAction
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import (
    get_pdca_task,
    list_pdca_events,
    load_pdca_checkpoint,
    upsert_pdca_task,
)
from alphonse.agent.nervous_system.senses.bus import Signal


def test_handle_pdca_slice_request_without_text_moves_to_waiting_user(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-1",
            "status": "queued",
        }
    )
    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-1"},
        source="pdca_queue_runner",
    )
    result = action.execute({"signal": signal})
    assert result.intention_key == "NOOP"

    task = get_pdca_task(task_id)
    assert task is not None
    assert task["status"] == "waiting_user"

    events = list_pdca_events(task_id=task_id, limit=10)
    assert events
    assert events[-1]["event_type"] == "slice.blocked.missing_text"
    assert any(item["event_type"] == "slice.request.signal_received" for item in events)


def test_handle_pdca_slice_request_executes_and_persists_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_PDCA_QUEUE_DISPATCH_COOLDOWN_SECONDS", "10")
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-1",
            "status": "queued",
            "metadata": {"pending_user_text": "Please do the task"},
        }
    )

    class _FakeResult:
        reply_text = ""
        plans = []
        cognition_state = {"task_state": {"status": "running", "cycle_index": 1}}
        meta = {}

    class _FakeGraph:
        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = (state, llm_client)
            assert text == "Please do the task"
            return _FakeResult()

    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)

    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-2"},
        source="pdca_queue_runner",
    )
    result = action.execute({"signal": signal, "ctx": None})
    assert result.intention_key == "NOOP"

    task = get_pdca_task(task_id)
    assert task is not None
    assert task["status"] == "queued"
    assert isinstance(task.get("next_run_at"), str) and task["next_run_at"]

    checkpoint = load_pdca_checkpoint(task_id)
    assert checkpoint is not None
    assert checkpoint["version"] >= 1
    assert checkpoint["task_state"]["status"] == "running"

    events = list_pdca_events(task_id=task_id, limit=20)
    assert any(item["event_type"] == "slice.request.signal_received" for item in events)
    assert any(item["event_type"] == "slice.completed.queued" for item in events)


def test_handle_pdca_slice_request_without_task_id_is_noop() -> None:
    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={},
        source="pdca_queue_runner",
    )
    result = action.execute({"signal": signal})
    assert result.intention_key == "NOOP"


def test_handle_pdca_slice_request_ignores_duplicate_signal_correlation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_PDCA_QUEUE_DISPATCH_COOLDOWN_SECONDS", "10")
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-dup",
            "status": "queued",
            "metadata": {"pending_user_text": "Execute this"},
        }
    )

    class _FakeResult:
        reply_text = ""
        plans = []
        cognition_state = {"task_state": {"status": "running", "cycle_index": 1}}
        meta = {}

    class _FakeGraph:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = (state, text, llm_client)
            self.calls += 1
            return _FakeResult()

    fake_graph = _FakeGraph()
    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", fake_graph)
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)

    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "dup-cid"},
        source="pdca_queue_runner",
    )
    _ = action.execute({"signal": signal, "ctx": None})
    _ = action.execute({"signal": signal, "ctx": None})

    assert fake_graph.calls == 1
    events = list_pdca_events(task_id=task_id, limit=50)
    received = [e for e in events if e["event_type"] == "slice.request.signal_received" and e["correlation_id"] == "dup-cid"]
    ignored = [e for e in events if e["event_type"] == "slice.request.duplicate_ignored" and e["correlation_id"] == "dup-cid"]
    assert len(received) == 1
    assert len(ignored) == 1
