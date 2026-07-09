from __future__ import annotations

import json

from alphonse.agent_v2.core.core import CoreUiEvent
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.interfaces.ag_ui import AgUiAdapter
from alphonse.agent_v2.interfaces.tui import build_tui_runtime
from alphonse.agent_v2.interfaces.tui import queue_tui_input
from alphonse.agent_v2.interfaces.tui import _response_from_snapshot
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.core import LoopStepStatus


def test_tui_renders_yes_no_question_interrupt() -> None:
    snapshot = StateSnapshot(
        metadata={
            "question_interrupt": {
                "message": "Continue?",
                "kind": "yes_no",
                "choices": [],
            }
        }
    )

    response = _response_from_snapshot(snapshot, type("Step", (), {"status": LoopStepStatus.PARKED})())

    assert response == "Continue?\n[yes/no]"


def test_tui_answer_resumes_pending_question() -> None:
    store = SQLiteQuestionStore()
    task = TaskState(task_id="task-1", goal="Need answer", user="local")
    question = store.create_question(task=task, question="Continue?", kind="yes_no")
    runtime = build_tui_runtime(user="local", question_store=store)

    result = queue_tui_input(runtime, "yes")

    assert result.queued is True
    queued = runtime.queue.peek()
    assert queued is not None
    assert queued.message.metadata["answered_question_id"] == question.question_id
    task_state = queued.message.metadata["task_state"]
    assert "question_answer" not in task_state["metadata"]
    execution = json.loads(task_state["plan_json"])[0]["execution"]
    assert execution["result"]["answer"] == {"answer": True}


def test_tui_invalid_choice_answer_does_not_resume() -> None:
    store = SQLiteQuestionStore()
    task = TaskState(task_id="task-1", goal="Need answer", user="local")
    store.create_question(
        task=task,
        question="Pick one",
        kind="single_choice",
        choices=[{"id": "a", "label": "Alpha"}],
    )
    runtime = build_tui_runtime(user="local", question_store=store)

    result = queue_tui_input(runtime, "Beta")

    assert result.queued is False
    assert "Please choose one of" in result.response
    assert runtime.queue.size() == 0


def test_ag_ui_adapter_emits_interrupt_run_finished_with_snapshots() -> None:
    store = SQLiteQuestionStore()
    task = TaskState(task_id="task-1", goal="Need answer", user="alex", recent_conversation_md='- alex: "Need answer"')
    question = store.create_question(task=task, question="Continue?", kind="yes_no")
    adapter = AgUiAdapter(question_store=store)

    events = adapter.map_event(
        CoreUiEvent(
            event_type="run_finished",
            payload={"task": task.to_dict(), "question": question.to_dict()},
        )
    )

    assert [event["type"] for event in events] == ["STATE_SNAPSHOT", "MESSAGES_SNAPSHOT", "RUN_FINISHED"]
    finished = events[-1]
    assert finished["outcome"]["type"] == "interrupt"
    interrupt = finished["outcome"]["interrupts"][0]
    assert interrupt["id"] == question.question_id
    assert interrupt["reason"] == "confirmation"
    assert interrupt["responseSchema"]["properties"]["answer"]["type"] == "boolean"


def test_ag_ui_resume_resolves_question_through_store() -> None:
    store = SQLiteQuestionStore()
    task = TaskState(task_id="task-1", goal="Need answer", user="alex")
    question = store.create_question(task=task, question="Continue?", kind="yes_no")
    adapter = AgUiAdapter(question_store=store)

    events = adapter.resume(
        {
            "threadId": question.thread_id,
            "runId": "run-2",
            "resume": [
                {
                    "interruptId": question.question_id,
                    "status": "resolved",
                    "payload": {"answer": True},
                }
            ],
        }
    )

    assert events[0].event_type == "question_interrupt_resolved"
    assert events[0].payload["answer"] == {"answer": True}
    resumed_task = events[0].payload["resumed_task"]
    assert "question_answer" not in resumed_task["metadata"]
    execution = json.loads(resumed_task["plan_json"])[0]["execution"]
    assert execution["result"]["question_id"] == question.question_id
