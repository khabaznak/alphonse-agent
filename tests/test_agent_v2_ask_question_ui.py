from __future__ import annotations

import json

from alphonse.agent_v2.core.core import CoreUiEvent
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.interfaces.ag_ui import AgUiAdapter








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
