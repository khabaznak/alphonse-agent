from __future__ import annotations

from datetime import datetime, timezone

from alphonse.agent_v2.core.core import CoreLoopContext, CoreMessage
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.planner import _ingest_relevant_messages
from alphonse.agent_v2.core.messages.queue import InMemoryMessageQueue


class _Jev:
    def __init__(self, selected: tuple[str, ...]) -> None:
        self.selected = selected
        self.candidates = []

    def triage_plan_messages(self, *, task, candidates):
        _ = task
        self.candidates = candidates
        return self.selected


def _message(prompt: str, *, user: str = "alex", disposition: str = "steering") -> CoreMessage:
    return CoreMessage(
        timestamp=datetime.now(timezone.utc), prompt=prompt, user=user, project_id="home",
        metadata={"routing_disposition": disposition},
    )


def test_plan_intake_batches_eligible_messages_and_consumes_only_jev_selected_ids() -> None:
    queue = InMemoryMessageQueue()
    relevant = queue.enqueue(_message("Consult B before X"))
    unrelated = queue.enqueue(_message("Do unrelated work", disposition="pdca_task"))
    jev = _Jev((relevant.message_id,))
    task = TaskState(user="alex", project_id="home", task_id="task-1", goal="Do X")
    context = CoreLoopContext(messages=queue, system_one=jev)

    _ingest_relevant_messages(task, context)

    assert [item["message_id"] for item in jev.candidates] == [relevant.message_id]
    assert "Consult B before X" in task.recent_conversation_md
    assert queue.size() == 1
    assert queue.peek().message_id == unrelated.message_id
    assert task.metadata["v3_intake_ignored_message_ids"] == []


def test_plan_intake_keeps_jev_rejected_steering_queued_without_reinterrupting_task() -> None:
    queue = InMemoryMessageQueue()
    queued = queue.enqueue(_message("Something unrelated"))
    task = TaskState(user="alex", project_id="home", task_id="task-1", goal="Do X")
    context = CoreLoopContext(messages=queue, system_one=_Jev(()))

    _ingest_relevant_messages(task, context)

    assert queue.peek().message_id == queued.message_id
    assert task.metadata["v3_intake_ignored_message_ids"] == [queued.message_id]
