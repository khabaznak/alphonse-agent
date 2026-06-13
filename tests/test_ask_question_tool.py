from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import alphonse.agent.services.question_answer_service as answer_service
import alphonse.agent.tools.ask_question_tool as ask_module
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.pending_questions import create_pending_question
from alphonse.agent.nervous_system.pending_questions import bind_outbound_message
from alphonse.agent.nervous_system.pending_questions import get_pending_question
from alphonse.agent.nervous_system.pending_questions import list_pending_for_respondent
from alphonse.agent.nervous_system.pdca_queue_store import flush_pdca_runtime_state
from alphonse.agent.nervous_system.pdca_queue_store import get_pdca_task
from alphonse.agent.nervous_system.pdca_queue_store import load_pdca_checkpoint
from alphonse.agent.nervous_system.pdca_queue_store import list_runnable_pdca_tasks
from alphonse.agent.nervous_system.pdca_queue_store import save_pdca_checkpoint
from alphonse.agent.nervous_system.pdca_queue_store import upsert_pdca_task
from alphonse.agent.services.question_answer_service import process_expired_questions
from alphonse.agent.services.question_answer_service import route_inbound_question_answer
from alphonse.agent.tools.ask_question_tool import AskQuestionTool
from alphonse.agent.tools.registry import build_default_tool_registry


class _SuccessfulSender:
    def execute(self, **_: Any) -> dict[str, Any]:
        return {"output": {"delivered": True}, "exception": None, "metadata": {"tool": "communication.send_message"}}


class _CapturingSender:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        return {"output": {"delivered": True}, "exception": None, "metadata": {"tool": "communication.send_message"}}


class _Bus:
    def __init__(self) -> None:
        self.signals: list[Any] = []

    def emit(self, signal: Any) -> None:
        self.signals.append(signal)


@pytest.fixture(autouse=True)
def _question_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    monkeypatch.setenv("ALPHONSE_SESSION_ROOT", str(tmp_path / "sessions"))
    apply_schema(db_path)
    flush_pdca_runtime_state()
    users = {
        "user-a": {"user_id": "user-a", "display_name": "Alex"},
        "user-b": {"user_id": "user-b", "display_name": "Bea"},
    }
    monkeypatch.setattr(ask_module.identity, "get_user", lambda user_id: users.get(str(user_id)))
    monkeypatch.setattr(ask_module.identity, "get_preferred_service_id", lambda _user_id: 1)
    monkeypatch.setattr(ask_module.identity, "resolve_service_key", lambda _service_id: "telegram")
    monkeypatch.setattr(ask_module.identity, "resolve_delivery_target", lambda user_id, service_id: f"chat-{user_id}")
    monkeypatch.setattr(answer_service.identity, "get_user", lambda user_id: users.get(str(user_id)))
    monkeypatch.setattr(answer_service.SendMessageTool, "execute", lambda self, **kwargs: {"output": kwargs, "exception": None})
    yield
    flush_pdca_runtime_state()


def _create_task(task_id: str = "task-a") -> TaskRecord:
    upsert_pdca_task(
        {
            "task_id": task_id,
            "owner_id": "user-a",
            "conversation_key": "telegram:chat-user-a",
            "status": "running",
            "metadata": {"inputs": [], "next_unconsumed_index": 0},
        }
    )
    record = TaskRecord(
        task_id=task_id,
        user_id="user-a",
        correlation_id="corr-a",
        goal="complete the form",
    )
    save_pdca_checkpoint(task_id=task_id, state={"task_record": record.to_dict()}, expected_version=0)
    return record


def test_ask_question_is_registered() -> None:
    registry = build_default_tool_registry()
    definition = registry.get("communication.ask_question")
    assert definition is not None
    assert registry.get("askQuestion") is definition
    assert definition.spec.input_schema["properties"]["respondent_user_id"]


def test_ask_question_parks_task_record_and_creates_pending_question() -> None:
    record = _create_task()
    tool = AskQuestionTool(_send_message_tool=_SuccessfulSender())
    result = tool.execute(
        question="Where did you live when you were eight?",
        state={
            "task_record": record,
            "conversation_key": "telegram:chat-user-a",
            "channel_type": "telegram",
            "channel_target": "chat-user-a",
        },
    )
    assert result["exception"] is None
    assert result["output"]["waiting_for_answer"] is True
    assert record.status == "waiting_user"
    pending = list_pending_for_respondent("user-a")
    assert len(pending) == 1
    assert pending[0]["task_id"] == "task-a"


def test_owner_question_falls_back_to_active_task_channel_without_preference(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _create_task()
    sender = _CapturingSender()
    monkeypatch.setattr(ask_module.identity, "get_preferred_service_id", lambda _user_id: None)
    monkeypatch.setattr(ask_module.identity, "resolve_service_id", lambda channel: 1 if channel == "telegram" else None)
    monkeypatch.setattr(
        ask_module.identity,
        "resolve_delivery_target",
        lambda user_id, service_id: None,
    )

    result = AskQuestionTool(_send_message_tool=sender).execute(
        question="Where did you live when you were eight?",
        state={
            "task_record": record,
            "conversation_key": "telegram:8553589429",
            "channel_type": "telegram",
            "channel_target": "8553589429",
        },
    )

    assert result["exception"] is None
    assert result["output"]["respondent_conversation_key"] == "telegram:8553589429"
    assert record.status == "waiting_user"
    assert sender.calls[0]["UserId"] == "user-a"
    assert sender.calls[0]["Channel"] == "telegram"


def test_delegated_question_does_not_fall_back_to_originators_active_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    record = _create_task()
    monkeypatch.setattr(ask_module.identity, "get_preferred_service_id", lambda _user_id: None)
    monkeypatch.setattr(ask_module.identity, "resolve_delivery_target", lambda user_id, service_id: None)

    result = AskQuestionTool(_send_message_tool=_SuccessfulSender()).execute(
        question="What do you want for dinner?",
        respondent_user_id="user-b",
        state={
            "task_record": record,
            "channel_type": "telegram",
            "channel_target": "chat-user-a",
        },
    )

    assert result["exception"]["code"] == "unresolved_delivery_target"
    assert record.status == "running"
    assert list_pending_for_respondent("user-b") == []


def test_delegated_answer_queues_originating_task_and_updates_checkpoint() -> None:
    record = _create_task()
    question = create_pending_question(
        task_id="task-a",
        originator_user_id="user-a",
        originator_conversation_key="telegram:chat-user-a",
        respondent_user_id="user-b",
        respondent_conversation_key="telegram:chat-user-b",
        question_text="What do you want for dinner?",
    )
    bus = _Bus()
    routed = route_inbound_question_answer(
        payload={
            "service_key": "telegram",
            "channel_target": "chat-user-b",
            "provider_message_id": "answer-1",
            "text": "Sushi",
        },
        respondent_user_id="user-b",
        correlation_id="corr-answer-1",
        bus=bus,
    )
    assert routed == {"handled": True, "question_id": question["question_id"], "task_id": "task-a"}
    assert get_pdca_task("task-a")["status"] == "queued"
    assert get_pending_question(question["question_id"])["status"] == "answered"
    checkpoint = load_pdca_checkpoint("task-a")
    assert "Bea answered: Sushi" in checkpoint["state"]["task_record"]["recent_conversation_md"]
    assert bus.signals and bus.signals[0].type == "pdca.dispatch.kick"


def test_waiting_task_does_not_block_other_runnable_tasks() -> None:
    upsert_pdca_task(
        {"task_id": "task-waiting", "owner_id": "user-a", "conversation_key": "chat-a", "status": "waiting_user"}
    )
    upsert_pdca_task(
        {"task_id": "task-job", "owner_id": "user-a", "conversation_key": "job:a", "status": "queued"}
    )
    runnable_ids = {str(item.get("task_id")) for item in list_runnable_pdca_tasks()}
    assert "task-job" in runnable_ids
    assert "task-waiting" not in runnable_ids


def test_direct_reply_selects_bound_question() -> None:
    _create_task("task-a")
    _create_task("task-b")
    first = create_pending_question(
        task_id="task-a",
        originator_user_id="user-a",
        originator_conversation_key="telegram:chat-user-a",
        respondent_user_id="user-b",
        respondent_conversation_key="telegram:chat-user-b",
        question_text="First question",
    )
    second = create_pending_question(
        task_id="task-b",
        originator_user_id="user-a",
        originator_conversation_key="telegram:chat-user-a",
        respondent_user_id="user-b",
        respondent_conversation_key="telegram:chat-user-b",
        question_text="Second question",
    )
    bind_outbound_message(question_id=first["question_id"], provider_message_id="out-1")
    bind_outbound_message(question_id=second["question_id"], provider_message_id="out-2")
    routed = route_inbound_question_answer(
        payload={
            "service_key": "telegram",
            "provider_message_id": "answer-direct",
            "reply_to_provider_message_id": "out-2",
            "text": "Second answer",
        },
        respondent_user_id="user-b",
        correlation_id="corr-direct",
        bus=_Bus(),
    )
    assert routed["question_id"] == second["question_id"]
    assert get_pending_question(first["question_id"])["status"] == "pending"


def test_multiple_pending_questions_are_ambiguous_without_direct_reply() -> None:
    _create_task("task-a")
    _create_task("task-b")
    for task_id in ("task-a", "task-b"):
        create_pending_question(
            task_id=task_id,
            originator_user_id="user-a",
            originator_conversation_key="telegram:chat-user-a",
            respondent_user_id="user-b",
            respondent_conversation_key="telegram:chat-user-b",
            question_text=f"Question for {task_id}",
        )
    routed = route_inbound_question_answer(
        payload={"service_key": "telegram", "provider_message_id": "answer-x", "text": "An answer"},
        respondent_user_id="user-b",
        correlation_id="corr-x",
        bus=_Bus(),
    )
    assert routed["handled"] is True
    assert routed["ambiguous"] is True
    assert len(list_pending_for_respondent("user-b")) == 2


def test_duplicate_answer_does_not_resume_twice() -> None:
    _create_task()
    question = create_pending_question(
        task_id="task-a",
        originator_user_id="user-a",
        originator_conversation_key="telegram:chat-user-a",
        respondent_user_id="user-b",
        respondent_conversation_key="telegram:chat-user-b",
        question_text="Question",
    )
    payload = {"service_key": "telegram", "provider_message_id": "answer-1", "text": "First"}
    first = route_inbound_question_answer(
        payload=payload, respondent_user_id="user-b", correlation_id="corr-1", bus=_Bus()
    )
    second = route_inbound_question_answer(
        payload=payload, respondent_user_id="user-b", correlation_id="corr-1", bus=_Bus()
    )
    assert first["question_id"] == question["question_id"]
    assert second == {"handled": False}


def test_expired_question_keeps_task_parked(monkeypatch: pytest.MonkeyPatch) -> None:
    _create_task()
    question = create_pending_question(
        task_id="task-a",
        originator_user_id="user-a",
        originator_conversation_key="telegram:chat-user-a",
        respondent_user_id="user-b",
        respondent_conversation_key="telegram:chat-user-b",
        question_text="Question",
        expires_in_seconds=60,
    )
    import alphonse.agent.nervous_system.pending_questions as store

    monkeypatch.setattr(store, "_now_iso", lambda: "2999-01-01T00:00:00+00:00")
    assert process_expired_questions() == 1
    task = get_pdca_task("task-a")
    assert task["status"] == "waiting_user"
    assert task["last_error"] == "question_expired"
    assert get_pending_question(question["question_id"])["status"] == "expired"
