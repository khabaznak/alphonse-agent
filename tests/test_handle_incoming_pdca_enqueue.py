from __future__ import annotations

from pathlib import Path

import pytest

from alphonse.agent.actions import handle_incoming_message as him
from alphonse.agent.actions.handle_incoming_message import HandleIncomingMessageAction
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import (
    get_latest_pdca_task_for_conversation,
    get_pdca_task,
    upsert_pdca_task,
)
from alphonse.agent.nervous_system.senses.bus import Bus, Signal


def test_incoming_message_enqueues_new_pdca_task_when_slicing_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_PDCA_SLICING_ENABLED", "true")
    apply_schema(db_path)

    class _FailGraph:
        def invoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("graph invoke should not run when enqueue path is enabled")

    monkeypatch.setattr(him, "_CORTEX_GRAPH", _FailGraph())

    action = HandleIncomingMessageAction()
    bus = Bus()
    signal = Signal(
        type="api.message_received",
        payload={
            "text": "Please help me with a long task",
            "channel": "webui",
            "target": "webui",
        },
        source="api",
    )
    result = action.execute({"signal": signal, "state": None, "outcome": None, "ctx": bus})
    assert result.intention_key == "NOOP"

    emitted = bus.get(timeout=0.2)
    assert emitted is not None
    assert emitted.type == "pdca.slice.requested"
    assert str((emitted.payload or {}).get("text") or "") == "Please help me with a long task"

    task = get_latest_pdca_task_for_conversation(conversation_key="webui:webui")
    assert task is not None
    assert task["status"] == "queued"
    assert str((task.get("metadata") or {}).get("pending_user_text") or "") == "Please help me with a long task"


def test_incoming_message_resumes_waiting_pdca_task_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_PDCA_SLICING_ENABLED", "true")
    apply_schema(db_path)

    existing_task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "webui:webui",
            "status": "waiting_user",
            "metadata": {"pending_user_text": "old text"},
        }
    )

    class _FailGraph:
        def invoke(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise AssertionError("graph invoke should not run when enqueue path is enabled")

    monkeypatch.setattr(him, "_CORTEX_GRAPH", _FailGraph())

    action = HandleIncomingMessageAction()
    bus = Bus()
    signal = Signal(
        type="api.message_received",
        payload={
            "text": "Here is the missing detail",
            "channel": "webui",
            "target": "webui",
        },
        source="api",
    )
    _ = action.execute({"signal": signal, "state": None, "outcome": None, "ctx": bus})

    emitted = bus.get(timeout=0.2)
    assert emitted is not None
    assert emitted.type == "pdca.resume_requested"
    assert str((emitted.payload or {}).get("task_id") or "") == existing_task_id

    task = get_pdca_task(existing_task_id)
    assert task is not None
    assert task["status"] == "queued"
    assert str((task.get("metadata") or {}).get("pending_user_text") or "") == "Here is the missing detail"
