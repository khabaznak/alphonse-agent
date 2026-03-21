from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from alphonse.agent.actions import handle_pdca_slice_request as hpsr
from alphonse.agent.actions.handle_pdca_slice_request import HandlePdcaSliceRequestAction
from alphonse.agent.cortex.state_store import save_state
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pdca_queue_store import (
    get_pdca_task,
    list_pdca_events,
    load_pdca_checkpoint,
    save_pdca_checkpoint,
    update_pdca_task_metadata,
    upsert_pdca_task,
)
from alphonse.agent.nervous_system.senses.bus import Signal
from alphonse.agent.nervous_system.senses.bus import Bus


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
            "metadata": {
                "pending_user_text": "Please do the task",
                "last_user_channel": "telegram",
                "last_user_target": "8553589429",
            },
        }
    )
    projected: list[dict[str, object]] = []

    class _FakeResult:
        reply_text = ""
        plans = []
        cognition_state = {
            "task_state": {
                "status": "running",
                "cycle_index": 1,
                "pending_plan_raw": {
                    "tool_call": {"kind": "call_tool", "tool_name": "job_list", "args": {"limit": 3}},
                    "planner_intent": "Thinking about the best next tool.",
                },
                "plan": {
                    "version": 1,
                    "current_step_id": "step_1",
                    "steps": [
                        {
                            "step_id": "step_1",
                            "status": "proposed",
                            "proposal_raw": {
                                "tool_call": {"kind": "call_tool", "tool_name": "job_list", "args": {"limit": 3}},
                                "planner_intent": "Thinking about the best next tool.",
                            },
                            "raw_source": "complete_with_tools",
                        }
                    ],
                },
                "facts": {
                    "step_1": {
                        "tool": "planner_output",
                        "exception": {
                            "code": "invalid_planner_output",
                            "details": {"raw_output_preview": "{bad planner payload}"},
                        },
                    }
                },
            }
        }
        meta = {}

    class _FakeGraph:
        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = (state, llm_client)
            assert text == "Please do the task"
            sink = state.get("_transition_sink")
            assert callable(sink)
            sink({"type": "agent.transition", "phase": "thinking", "detail": {"presence_event_family": "presence.progress"}})
            return _FakeResult()

    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)
    monkeypatch.setattr(
        hpsr,
        "emit_channel_transition_event",
        lambda _incoming, event: projected.append(event if isinstance(event, dict) else {}),
    )

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
    assert checkpoint["task_state"]["pending_plan_raw"] is None
    checkpoint_plan = checkpoint["task_state"].get("plan")
    assert isinstance(checkpoint_plan, dict)
    checkpoint_steps = checkpoint_plan.get("steps")
    assert isinstance(checkpoint_steps, list) and checkpoint_steps
    first_step = checkpoint_steps[0]
    assert isinstance(first_step, dict)
    assert "proposal_raw" not in first_step
    checkpoint_facts = checkpoint["task_state"].get("facts")
    assert isinstance(checkpoint_facts, dict)
    planner_fact = checkpoint_facts.get("step_1")
    assert isinstance(planner_fact, dict)
    planner_exception = planner_fact.get("exception")
    assert isinstance(planner_exception, dict)
    planner_details = planner_exception.get("details")
    assert isinstance(planner_details, dict)
    assert "raw_output_preview" not in planner_details
    state_events = checkpoint["state"].get("events")
    if isinstance(state_events, list):
        for event in state_events:
            if not isinstance(event, dict):
                continue
            detail = event.get("detail")
            if not isinstance(detail, dict):
                continue
            if str(detail.get("presence_event_family") or "") != "presence.progress":
                continue
            assert "text" not in detail
    assert "_transition_sink" not in checkpoint["state"]

    events = list_pdca_events(task_id=task_id, limit=20)
    assert any(item["event_type"] == "slice.request.signal_received" for item in events)
    assert any(item["event_type"] == "slice.completed.queued" for item in events)
    families = {
        str((event.get("detail") if isinstance(event.get("detail"), dict) else {}).get("presence_event_family") or "")
        for event in projected
    }
    assert "presence.phase_changed" in families
    assert "presence.progress" in families


def test_handle_pdca_slice_request_does_not_dequeue_buffered_inputs_before_graph(
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
            "conversation_key": "chat-buffered",
            "status": "queued",
            "metadata": {
                "pending_user_text": "fallback text",
                "last_user_message": "fallback text",
                "next_unconsumed_index": 0,
                "input_dirty": False,
                "inputs": [
                    {
                        "message_id": "m-1",
                        "correlation_id": "cid-1",
                        "text": "first buffered",
                        "channel": "telegram",
                        "received_at": "2026-03-12T05:00:00+00:00",
                        "consumed_at": None,
                        "sequence": 1,
                    },
                    {
                        "message_id": "m-2",
                        "correlation_id": "cid-2",
                        "text": "second buffered",
                        "channel": "telegram",
                        "received_at": "2026-03-12T05:01:00+00:00",
                        "consumed_at": None,
                        "sequence": 2,
                    },
                ],
            },
        }
    )

    class _FakeResult:
        reply_text = ""
        plans = []
        cognition_state = {"task_state": {"status": "queued", "cycle_index": 1}}
        meta = {}

    class _FakeGraph:
        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = llm_client
            _ = state
            assert text == "fallback text"
            return _FakeResult()

    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)

    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-buffered"},
        source="pdca_queue_runner",
    )
    _ = action.execute({"signal": signal, "ctx": None})

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    assert int(metadata.get("next_unconsumed_index") or 0) == 0
    assert bool(metadata.get("input_dirty")) is False
    inputs = metadata.get("inputs")
    assert isinstance(inputs, list)
    assert all(not str(item.get("consumed_at") or "").strip() for item in inputs if isinstance(item, dict))


def test_handle_pdca_slice_request_accepts_attachment_only_buffered_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-attachment-only",
            "status": "queued",
            "metadata": {
                "pending_user_text": "[attachments: 1 voice]",
                "last_user_message": "[attachments: 1 voice]",
                "next_unconsumed_index": 0,
                "input_dirty": False,
                "inputs": [
                    {
                        "message_id": "m-a1",
                        "correlation_id": "cid-a1",
                        "text": "",
                        "attachments": [{"kind": "voice", "provider": "telegram", "file_id": "voice-1"}],
                        "channel": "telegram",
                        "received_at": "2026-03-12T05:00:00+00:00",
                        "consumed_at": None,
                        "sequence": 1,
                    }
                ],
            },
        }
    )

    class _FakeResult:
        reply_text = ""
        plans = []
        cognition_state = {"task_state": {"status": "queued", "cycle_index": 1}}
        meta = {}

    class _FakeGraph:
        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = llm_client
            assert text == "[attachments: 1 voice]"
            _ = state
            return _FakeResult()

    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)

    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-attachment-only"},
        source="pdca_queue_runner",
    )
    _ = action.execute({"signal": signal, "ctx": None})

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    assert int(metadata.get("next_unconsumed_index") or 0) == 0


def test_handle_pdca_slice_request_preempts_when_new_input_arrives_mid_slice(
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
            "conversation_key": "chat-preempt",
            "status": "queued",
            "metadata": {"pending_user_text": "initial"},
        }
    )

    class _FakeResult:
        reply_text = "stale reply should be suppressed"
        plans = []
        cognition_state = {"task_state": {"status": "running", "cycle_index": 1}}
        meta = {}

    class _FakeGraph:
        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = (state, text, llm_client)
            task = get_pdca_task(task_id)
            assert isinstance(task, dict)
            metadata = dict(task.get("metadata") or {})
            metadata["input_dirty"] = True
            metadata["last_enqueued_at"] = datetime.now(timezone.utc).isoformat()
            metadata["inputs"] = [
                {
                    "message_id": "m-new",
                    "correlation_id": "cid-new",
                    "text": "fresh message",
                    "channel": "telegram",
                    "received_at": datetime.now(timezone.utc).isoformat(),
                    "consumed_at": None,
                    "sequence": 1,
                }
            ]
            metadata["next_unconsumed_index"] = 0
            update_pdca_task_metadata(task_id=task_id, metadata=metadata)
            return _FakeResult()

    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)

    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-preempt"},
        source="pdca_queue_runner",
    )
    _ = action.execute({"signal": signal, "ctx": None})

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    assert task["status"] == "queued"
    events = list_pdca_events(task_id=task_id, limit=50)
    assert any(item["event_type"] == "pdca.slice.preempt_after_step" for item in events)
    completed = [item for item in events if item["event_type"] == "slice.completed.queued"]
    assert completed
    assert str(completed[-1]["payload"].get("reply_text") or "") == ""


def test_handle_pdca_slice_request_injects_session_snapshot_for_plan_and_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-session-1",
            "conversation_key": "telegram:8553589429",
            "status": "queued",
            "metadata": {
                "pending_user_text": "Need context",
                "state": {
                    "channel_type": "telegram",
                    "channel_target": "8553589429",
                    "timezone": "America/Mexico_City",
                },
            },
        }
    )

    class _FakeResult:
        reply_text = ""
        plans = []
        cognition_state = {"task_state": {"status": "queued", "cycle_index": 1}}
        meta = {}

    captured: dict[str, object] = {}

    class _FakeGraph:
        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = (text, llm_client)
            captured["session_state"] = state.get("session_state")
            captured["recent_conversation_block"] = state.get("recent_conversation_block")
            return _FakeResult()

    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)

    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-session-1"},
        source="pdca_queue_runner",
    )
    _ = action.execute({"signal": signal, "ctx": None})

    assert isinstance(captured.get("session_state"), dict)
    rendered = str(captured.get("recent_conversation_block") or "")
    assert "RECENT CONVERSATION" in rendered


def test_handle_pdca_slice_request_does_not_clobber_concurrent_incoming_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-race",
            "status": "queued",
            "metadata": {
                "pending_user_text": "old text",
                "last_user_message_id": "100",
            },
        }
    )

    class _FakeResult:
        reply_text = ""
        plans = []
        cognition_state = {"task_state": {"status": "queued", "cycle_index": 1}}
        meta = {}

    class _FakeGraph:
        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = (state, text, llm_client)
            task = get_pdca_task(task_id)
            assert isinstance(task, dict)
            metadata = dict(task.get("metadata") or {})
            metadata["pending_user_text"] = "new text"
            metadata["last_user_message_id"] = "999"
            update_pdca_task_metadata(task_id=task_id, metadata=metadata)
            return _FakeResult()

    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)

    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-race-1"},
        source="pdca_queue_runner",
    )
    _ = action.execute({"signal": signal, "ctx": None})

    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    metadata = task.get("metadata") if isinstance(task.get("metadata"), dict) else {}
    assert metadata.get("pending_user_text") == "new text"
    assert metadata.get("last_user_message_id") == "999"


def test_handle_pdca_slice_request_without_task_id_is_noop() -> None:
    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={},
        source="pdca_queue_runner",
    )
    result = action.execute({"signal": signal})
    assert result.intention_key == "NOOP"


def test_handle_pdca_slice_request_ignores_terminal_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-terminal",
            "conversation_key": "chat-terminal",
            "status": "done",
            "metadata": {"pending_user_text": "ignored"},
        }
    )

    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-terminal"},
        source="pdca_dispatcher",
    )
    result = action.execute({"signal": signal, "ctx": None})
    assert result.intention_key == "NOOP"

    events = list_pdca_events(task_id=task_id, limit=20)
    assert any(item["event_type"] == "slice.request.terminal_ignored" for item in events)


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


def test_handle_pdca_slice_request_blocks_when_max_cycles_reached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-cycle-limit",
            "status": "queued",
            "max_cycles": 2,
            "metadata": {"pending_user_text": "Continue"},
        }
    )
    _ = save_pdca_checkpoint(
        task_id=task_id,
        state={"conversation_key": "chat-cycle-limit"},
        task_state={"cycle_index": 2, "status": "running"},
        expected_version=0,
    )

    class _FakeGraph:
        calls = 0

        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = (state, text, llm_client)
            self.calls += 1
            raise AssertionError("graph should not be called when max_cycles is reached")

    fake_graph = _FakeGraph()
    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", fake_graph)
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)

    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-cycle"},
        source="pdca_queue_runner",
    )
    action = HandlePdcaSliceRequestAction()
    _ = action.execute({"signal": signal, "ctx": None})

    assert fake_graph.calls == 0
    task = get_pdca_task(task_id)
    assert task is not None
    assert task["status"] == "failed"
    events = list_pdca_events(task_id=task_id, limit=20)
    assert any(item["event_type"] == "slice.blocked.budget_exhausted" for item in events)


def test_handle_pdca_slice_request_blocks_when_token_budget_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-token-limit",
            "status": "queued",
            "token_budget_remaining": 0,
            "metadata": {"pending_user_text": "Continue"},
        }
    )

    class _FakeGraph:
        calls = 0

        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = (state, text, llm_client)
            self.calls += 1
            raise AssertionError("graph should not be called when token budget is exhausted")

    fake_graph = _FakeGraph()
    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", fake_graph)
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)

    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-token"},
        source="pdca_queue_runner",
    )
    action = HandlePdcaSliceRequestAction()
    _ = action.execute({"signal": signal, "ctx": None})

    assert fake_graph.calls == 0
    task = get_pdca_task(task_id)
    assert task is not None
    assert task["status"] == "failed"
    events = list_pdca_events(task_id=task_id, limit=20)
    blocked = [item for item in events if item["event_type"] == "slice.blocked.budget_exhausted"]
    assert blocked
    assert blocked[-1]["payload"].get("reason") == "token_budget_exhausted"


def test_handle_pdca_slice_request_blocks_when_runtime_budget_exhausted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "chat-runtime-limit",
            "status": "queued",
            "max_runtime_seconds": 1,
            "metadata": {
                "pending_user_text": "Continue",
                "started_at": "2000-01-01T00:00:00+00:00",
            },
        }
    )

    class _FakeGraph:
        calls = 0

        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = (state, text, llm_client)
            self.calls += 1
            raise AssertionError("graph should not be called when runtime budget is exhausted")

    fake_graph = _FakeGraph()
    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", fake_graph)
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)

    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-runtime"},
        source="pdca_queue_runner",
    )
    action = HandlePdcaSliceRequestAction()
    _ = action.execute({"signal": signal, "ctx": None})

    assert fake_graph.calls == 0
    task = get_pdca_task(task_id)
    assert task is not None
    assert task["status"] == "failed"
    events = list_pdca_events(task_id=task_id, limit=20)
    blocked = [item for item in events if item["event_type"] == "slice.blocked.budget_exhausted"]
    assert blocked
    assert blocked[-1]["payload"].get("reason") == "max_runtime_reached"


def test_handle_pdca_slice_request_prefers_checkpoint_state_for_rehydration(
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
            "conversation_key": "chat-rehydrate",
            "status": "queued",
            "metadata": {"state": {"from_metadata": True}},
        }
    )
    save_state("chat-rehydrate", {"from_session_store": True})
    _ = save_pdca_checkpoint(
        task_id=task_id,
        state={
            "from_checkpoint": True,
            "last_user_message": "resume from checkpoint",
            "task_state": {"cycle_index": 5},
        },
        task_state={"cycle_index": 5, "status": "running"},
        expected_version=0,
    )

    class _FakeResult:
        reply_text = ""
        plans = []
        cognition_state = {"task_state": {"status": "running", "cycle_index": 6}}
        meta = {}

    class _FakeGraph:
        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = llm_client
            assert state.get("from_checkpoint") is True
            assert state.get("from_session_store") is None
            task_state = state.get("task_state")
            assert isinstance(task_state, dict)
            assert task_state.get("cycle_index") == 5
            assert text == "resume from checkpoint"
            return _FakeResult()

    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)

    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-rehydrate"},
        source="pdca_queue_runner",
    )
    action = HandlePdcaSliceRequestAction()
    _ = action.execute({"signal": signal, "ctx": None})

    checkpoint = load_pdca_checkpoint(task_id)
    assert checkpoint is not None
    assert checkpoint["task_state"]["cycle_index"] == 6


def test_handle_pdca_slice_request_sanitizes_stale_session_pdca_state_for_new_task(
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
            "conversation_key": "telegram:8553589429",
            "status": "queued",
            "metadata": {
                "pending_user_text": "Yo Alphonse!",
                "last_user_channel": "telegram",
                "last_user_target": "8553589429",
            },
        }
    )
    save_state(
        "telegram:8553589429",
        {
            "conversation_key": "telegram:8553589429",
            "channel_type": "telegram",
            "channel_target": "8553589429",
            "locale": "en-US",
            "task_state": {
                "status": "failed",
                "planner_error_streak": 3,
                "facts": {"step_99": {"tool": "planner_output"}},
            },
            "pending_interaction": {"type": "SLOT_FILL"},
            "response_text": "stale-response",
            "events": [{"type": "agent.transition", "phase": "failed"}],
        },
    )
    observed: list[tuple[str, dict[str, object]]] = []

    class _FakeResult:
        reply_text = ""
        plans = []
        cognition_state = {"task_state": {"status": "running", "cycle_index": 1}}
        meta = {}

    class _FakeGraph:
        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = llm_client
            assert text == "Yo Alphonse!"
            assert state.get("conversation_key") == "telegram:8553589429"
            assert state.get("locale") == "en-US"
            task_state = state.get("task_state")
            assert isinstance(task_state, dict)
            assert task_state.get("check_provenance") is None
            assert task_state.get("planner_error_streak") is None
            assert task_state.get("facts") is None
            assert state.get("pending_interaction") is None
            assert state.get("response_text") is None
            assert state.get("events") is None
            return _FakeResult()

    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", _FakeGraph())
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)
    monkeypatch.setattr(
        hpsr,
        "_LOG",
        type(
            "_CaptureLog",
            (),
            {"emit": staticmethod(lambda **kwargs: observed.append((str(kwargs.get("event") or ""), kwargs.get("payload") or {})))},
        )(),
    )

    action = HandlePdcaSliceRequestAction()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-sanitize"},
        source="pdca_queue_runner",
    )
    _ = action.execute({"signal": signal, "ctx": None})

    checkpoint = load_pdca_checkpoint(task_id)
    assert checkpoint is not None
    saved_state = checkpoint.get("state") if isinstance(checkpoint.get("state"), dict) else {}
    assert "task_state" in saved_state  # only new cycle state from this run
    assert "pending_interaction" not in saved_state
    assert "response_text" in saved_state
    assert "events" not in saved_state
    sanitize_events = [payload for event, payload in observed if event == "pdca.slice.base_state.sanitized"]
    assert sanitize_events
    removed_keys = sanitize_events[-1].get("removed_keys")
    assert isinstance(removed_keys, list)
    assert "task_state" in removed_keys


def test_handle_pdca_slice_request_classifies_engine_unavailable_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "telegram:8553589429",
            "status": "queued",
            "metadata": {"pending_user_text": "hello"},
        }
    )

    class _UnavailableGraph:
        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = (state, text, llm_client)
            raise RuntimeError(
                "HTTPConnectionPool(host='127.0.0.1', port=4096): "
                "Max retries exceeded with url: /session "
                "(Caused by NewConnectionError('[Errno 61] Connection refused'))"
            )

    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", _UnavailableGraph())
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)
    bus = Bus()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-engine-down"},
        source="pdca_queue_runner",
    )
    action = HandlePdcaSliceRequestAction()
    _ = action.execute({"signal": signal, "ctx": bus})

    emitted = bus.get(timeout=0.05)
    assert emitted is not None
    assert emitted.type == "pdca.failed"
    assert emitted.payload.get("failure_code") == "engine_unavailable"
    assert emitted.payload.get("user_notice_required") is True


def test_handle_pdca_slice_request_classifies_generic_failure_without_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "telegram:8553589429",
            "status": "queued",
            "metadata": {"pending_user_text": "hello"},
        }
    )

    class _GenericFailureGraph:
        def invoke(self, state, text, *, llm_client=None):  # noqa: ANN001
            _ = (state, text, llm_client)
            raise ValueError("bad_plan_shape")

    monkeypatch.setattr(hpsr, "_CORTEX_GRAPH", _GenericFailureGraph())
    monkeypatch.setattr(hpsr, "build_llm_client", lambda: None)
    bus = Bus()
    signal = Signal(
        type="pdca.slice.requested",
        payload={"task_id": task_id, "correlation_id": "cid-generic-fail"},
        source="pdca_queue_runner",
    )
    action = HandlePdcaSliceRequestAction()
    _ = action.execute({"signal": signal, "ctx": bus})

    emitted = bus.get(timeout=0.05)
    assert emitted is not None
    assert emitted.type == "pdca.failed"
    assert emitted.payload.get("failure_code") == "execution_failed"
    assert emitted.payload.get("user_notice_required") is False


def test_emit_context_continuity_gap_when_ingress_context_dropped() -> None:
    emitted: list[dict[str, object]] = []

    class _FakeLog:
        def emit(self, **kwargs: object) -> None:
            emitted.append(dict(kwargs))

    original_log = hpsr._LOG
    hpsr._LOG = _FakeLog()
    task = {
        "task_id": "task-gap-1",
        "metadata": {
            "state": {
                "channel_type": "telegram",
                "channel_target": "8553589429",
                "actor_person_id": "user-1",
                "incoming_user_id": "ext-1",
                "incoming_user_name": "Alex",
            }
        },
    }
    invoke_state = {"channel_type": "telegram"}
    try:
        hpsr._emit_context_continuity_gap_if_any(
            task=task,
            invoke_state=invoke_state,
            correlation_id="corr-gap-1",
        )
    finally:
        hpsr._LOG = original_log

    event = next((item for item in emitted if str(item.get("event") or "") == "pdca.context.continuity_gap"), None)
    assert isinstance(event, dict)
    payload = event.get("payload")
    assert isinstance(payload, dict)
    assert "channel_target" in list(payload.get("missing_in_invoke_state") or [])


def test_resolve_incoming_context_includes_message_id() -> None:
    task = {
        "task_id": "task-msgid-1",
        "conversation_key": "telegram:8553589429",
        "metadata": {
            "last_user_channel": "telegram",
            "last_user_target": "8553589429",
            "last_user_message_id": "321",
            "state": {
                "actor_person_id": "user-1",
            },
        },
    }
    incoming = hpsr._resolve_incoming_context(task=task, correlation_id="cid-msgid-1")
    assert incoming is not None
    assert incoming.message_id == "321"


def test_base_state_hydrates_correlation_id_for_loaded_session_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)

    task_id = upsert_pdca_task(
        {
            "owner_id": "owner-1",
            "conversation_key": "telegram:8553589429",
            "status": "queued",
            "metadata": {
                "last_enqueue_correlation_id": "cid-last-enqueue",
                "state": {
                    "conversation_key": "telegram:8553589429",
                    "channel_type": "telegram",
                    "channel_target": "8553589429",
                },
            },
        }
    )
    task = get_pdca_task(task_id)
    assert isinstance(task, dict)
    save_state("telegram:8553589429", {"conversation_key": "telegram:8553589429"})

    base = hpsr._base_state(task=task, checkpoint=None, correlation_id="cid-signal-123")
    assert str(base.get("correlation_id") or "") == "cid-signal-123"
