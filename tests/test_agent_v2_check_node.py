from __future__ import annotations

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.pdca.nodes.check_node import check_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import CommunicationChannel, InMemoryMessageQueue
from alphonse.agent_v2.system_one import SystemOneReviewResult, SystemOneUnavailableError


class _Jev:
    def __init__(self, updates=(), ambiguous=()):
        self.updates = updates
        self.ambiguous = ambiguous
        self.calls = 0

    def evaluate(self, **_values):
        self.calls += 1
        return SystemOneReviewResult(tuple(self.updates), tuple(self.ambiguous))


def _task_with_evidence() -> TaskState:
    task = TaskState(goal="Create a file", user="alex", project_id="alpha")
    task.set_acceptance_contract_from_markdown("- [ ] The file exists")
    task.append_plan_call({"id": "call-1", "tool_id": "native.project_search", "tool_name": "project_search", "arguments": {}, "internal_state": "Find file"})
    task.record_plan_call_success("call-1", {"path": "a.txt", "contents": "verified"})
    return task


def test_check_marks_new_task_without_llm_or_jev_calls() -> None:
    task = TaskState(goal="Write a file")
    check_node(task)
    assert task.check_verdict == "new"


def test_check_does_not_consume_steering_when_disabled() -> None:
    queue = InMemoryMessageQueue()
    CommunicationChannel(queue).queue_message(prompt="Add tests", user="alex", project_id="alpha", metadata={"routing_disposition": "steering"})
    task = _task_with_evidence()
    jev = _Jev()

    check_node(task, CoreLoopContext(messages=queue, system_one=jev), consume_steering=False)

    assert queue.size() == 1
    assert jev.calls == 1


def test_check_uses_jev_evidence_review_without_llm_and_marks_success() -> None:
    task = _task_with_evidence()
    jev = _Jev(({
        "criterion_id": "ac-1", "status": "satisfied", "evidence_refs": ["tool-call:call-1"],
    },))
    context = CoreLoopContext(messages=InMemoryMessageQueue(), system_one=jev)

    check_node(task, context)

    assert task.check_verdict == "mission_success"
    assert task.acceptance_criteria_all_complete()
    assert task.metadata["system_one_check_review"]["updates"]


def test_check_preserves_ambiguous_criteria_for_act_recommendation() -> None:
    task = _task_with_evidence()
    jev = _Jev(ambiguous=("ac-1",))

    check_node(task, CoreLoopContext(messages=InMemoryMessageQueue(), system_one=jev))

    assert task.check_verdict == "wip"
    assert not task.acceptance_criteria_all_complete()


def test_check_fails_closed_when_jev_is_unavailable_for_evidence_review() -> None:
    task = _task_with_evidence()
    with pytest.raises(SystemOneUnavailableError, match="check_criteria_review_unavailable"):
        check_node(task, CoreLoopContext(messages=InMemoryMessageQueue()))
