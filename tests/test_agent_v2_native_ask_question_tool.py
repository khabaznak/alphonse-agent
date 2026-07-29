from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from importlib import import_module

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.intelligence.pdca.nodes import do_node
from alphonse.agent_v2.core.intelligence.pdca.processor import PDCAIntelligenceProcessor
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry.native import ASK_QUESTION_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import ASK_QUESTION_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry


def test_native_registry_registers_ask_question_tool() -> None:
    registry = build_native_tool_registry()

    descriptor = registry.get(ASK_QUESTION_TOOL_NAME)

    assert descriptor is not None
    assert descriptor.tool_id == ASK_QUESTION_TOOL_ID
    assert descriptor.argument_schema["required"] == ["question", "question_kind"]
    assert "interrupt" in descriptor.capabilities


def test_question_project_migration_backfills_checkpoint_idempotently(tmp_path) -> None:
    db_path = tmp_path / "questions.sqlite3"
    store = SQLiteQuestionStore(db_path)
    question = store.create_question(
        task=TaskState(task_id="task-migrate", goal="Migrate", user="alex", project_id="alpha"),
        question="Continue?",
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_v2_questions_respondent_project_status")
        conn.execute("ALTER TABLE v2_questions DROP COLUMN project_id")

    migrated = SQLiteQuestionStore(db_path)
    restarted = SQLiteQuestionStore(db_path)

    assert migrated.get_question(question.question_id).project_id == "alpha"
    assert restarted.get_question(question.question_id).project_id == "alpha"


def test_do_node_executes_ask_question_and_parks_task() -> None:
    task = TaskState(goal="Need a choice", user="alex", acceptance_criteria_md="1.- [ ] User answered")
    task.append_plan_call(
        {
            "id": "plan-call-ask",
            "tool_id": ASK_QUESTION_TOOL_ID,
            "tool_name": ASK_QUESTION_TOOL_NAME,
            "arguments": {"question": "Continue?", "question_kind": "yes_no"},
            "internal_state": "Asking for confirmation.",
        }
    )
    store = SQLiteQuestionStore()

    do_node(
        task,
        context=CoreLoopContext(
            messages=InMemoryMessageQueue(),
            tools=build_native_tool_registry(),
            question_store=store,
        ),
    )

    assert task.status == "waiting_user"
    assert task.metadata["task_parked"] is True
    assert task.metadata["question_interrupt"]["kind"] == "yes_no"
    execution = json.loads(task.plan_json)[0]["execution"]
    assert execution["status"] == "waiting"
    assert execution["result"]["waiting_for_answer"] is True
    assert len(store.list_pending_for_respondent("alex")) == 1


def test_question_answer_resumes_serialized_task_state() -> None:
    store = SQLiteQuestionStore()
    task = TaskState(task_id="task-1", goal="Need a choice", user="alex")
    question = store.create_question(task=task, question="Continue?", kind="yes_no")

    result = store.route_answer(respondent_user_id="alex", question_id=question.question_id, text="yes")

    assert result.handled is True
    assert result.resumed_task is not None
    assert result.resumed_task.status == "running"
    assert result.answer == {"answer": True}
    assert "question_answer" not in result.resumed_task.metadata
    execution = json.loads(result.resumed_task.plan_json)[0]["execution"]
    assert execution["status"] == "success"
    assert execution["result"] == {
        "question_id": question.question_id,
        "question_kind": "yes_no",
        "answer": {"answer": True},
        "answered_by": "alex",
    }


def test_multiple_pending_questions_are_ambiguous_without_direct_reference() -> None:
    store = SQLiteQuestionStore()
    store.create_question(task=TaskState(task_id="task-1", goal="One", user="alex"), question="One?", kind="open_text")
    store.create_question(task=TaskState(task_id="task-2", goal="Two", user="alex"), question="Two?", kind="open_text")

    result = store.route_answer(respondent_user_id="alex", text="answer")

    assert result.handled is True
    assert result.ambiguous is True
    assert result.resumed_task is None


def test_pdca_processor_returns_parked_status_for_waiting_question(monkeypatch) -> None:
    act_module = import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.act_node")
    plan_module = import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node")

    monkeypatch.setattr(act_module, "_call_acceptance_criteria_llm", lambda prompt: "1.- [ ] Confirmation received")
    monkeypatch.setattr(
        plan_module,
        "_call_tool_planning_llm",
        lambda prompt: {
            "id": "ask-1",
            "tool_id": ASK_QUESTION_TOOL_ID,
            "tool_name": ASK_QUESTION_TOOL_NAME,
            "arguments": {"question": "Continue?", "question_kind": "yes_no"},
            "internal_state": "Asking for confirmation.",
        },
    )
    queue = InMemoryMessageQueue()
    queued = queue.enqueue(CoreMessage(timestamp=datetime.now().astimezone(), prompt="Ask first", user="alex"))
    events = []
    result = PDCAIntelligenceProcessor().process(
        TaskState.from_queued_message(queued),
        CoreLoopContext(
            messages=queue,
            tools=build_native_tool_registry(),
            question_store=SQLiteQuestionStore(),
            ui_event_sink=events.append,
        ),
    )

    assert result.status.value == "parked"
    assert result.snapshot.metadata["question_interrupt"]["message"] == "Continue?"
    assert any(event.event_type == "state_snapshot" for event in events)


def test_core_parked_result_leaves_loop_available(monkeypatch) -> None:
    from alphonse.agent_v2.core.core import AlphonseCore
    from alphonse.agent_v2.core.state import AVAILABLE
    from alphonse.agent_v2.core.state import reset_state
    from alphonse.agent_v2.interfaces.tui import InMemoryInternalState, NullMemory, NullPromptLoader

    act_module = import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.act_node")
    plan_module = import_module("alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node")
    monkeypatch.setattr(act_module, "_call_acceptance_criteria_llm", lambda prompt: "1.- [ ] Confirmation received")
    monkeypatch.setattr(
        plan_module,
        "_call_tool_planning_llm",
        lambda prompt: {
            "id": "ask-1",
            "tool_id": ASK_QUESTION_TOOL_ID,
            "tool_name": ASK_QUESTION_TOOL_NAME,
            "arguments": {"question": "Continue?", "question_kind": "yes_no"},
            "internal_state": "Asking for confirmation.",
        },
    )
    reset_state()
    queue = InMemoryMessageQueue()
    queue.enqueue(CoreMessage(timestamp=datetime.now().astimezone(), prompt="Ask first", user="alex"))
    core = AlphonseCore(
        intelligence=PDCAIntelligenceProcessor(),
        messages=queue,
        tools=build_native_tool_registry(),
        prompts=NullPromptLoader(),
        state=InMemoryInternalState(),
        memory=NullMemory(),
        question_store=SQLiteQuestionStore(),
    )

    result = core.step()

    assert result.status == LoopStepStatus.PARKED
    assert result.state_after.key == AVAILABLE
