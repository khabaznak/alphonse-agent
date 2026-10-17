from __future__ import annotations

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.pdca.nodes.act_node import act_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.system_one import SystemOneActRecommendation, SystemOneUnavailableError


class _Jev:
    def __init__(self, action: str, answers: dict[str, float] | None = None) -> None:
        self.action = action
        self.answers = answers or {
            "continuation_is_worthwhile": 0.95,
            "user_input_can_unblock": 0.95,
            "closure_explanation_is_warranted": 0.95,
        }
        self.state = None

    def recommend_act(self, *, state):
        self.state = state
        return SystemOneActRecommendation(
            self.action, 0.95, True, f"Jev recommends {self.action}.", answers=self.answers,
        )


def _context(jev: _Jev) -> CoreLoopContext:
    return CoreLoopContext(messages=InMemoryMessageQueue(), system_one=jev)


def test_act_routes_new_mission_to_plan_without_inference() -> None:
    task = TaskState(goal="Record exercise", check_verdict="new")

    act_node(task)

    assert task.metadata["act_route"] == "plan"
    assert "acceptance_contract" not in task.metadata


def test_act_uses_jev_recommendation_and_provides_mission_evidence() -> None:
    jev = _Jev("replan")
    task = TaskState(goal="Connect to server", check_verdict="wip", acceptance_criteria_md="1.- [ ] Connected")

    act_node(task, _context(jev))

    assert task.metadata["act_route"] == "plan"
    assert task.metadata["act_directive"]["action"] == "replan"
    assert jev.state["check_verdict"] == "wip"
    assert jev.state["acceptance_contract"]["criteria"]


def test_act_asks_plan_for_user_question_when_jev_says_user_can_unblock() -> None:
    jev = _Jev("ask_user")
    task = TaskState(goal="Connect to server", check_verdict="mission_failed", acceptance_criteria_md="1.- [ ] Connected")

    act_node(task, _context(jev))

    assert task.metadata["act_route"] == "plan"
    assert task.metadata["act_directive"]["action"] == "ask_user"
    assert task.status == "running"


def test_act_schedules_one_closure_response_before_final_failure() -> None:
    jev = _Jev("fail_explain")
    task = TaskState(goal="Connect to server", check_verdict="mission_failed", acceptance_criteria_md="1.- [ ] Connected")

    act_node(task, _context(jev))

    assert task.metadata["act_route"] == "plan"
    assert task.metadata["act_directive"]["closure_only"] is True
    assert task.metadata["act_directive"]["terminal_outcome"] == "failed"
    assert task.status == "running"


def test_act_fails_immediately_when_jev_says_no_closure_is_warranted() -> None:
    jev = _Jev("fail", {"closure_explanation_is_warranted": 0.1})
    task = TaskState(goal="Connect to server", check_verdict="mission_failed", acceptance_criteria_md="1.- [ ] Connected")

    act_node(task, _context(jev))

    assert task.metadata["act_route"] == "end"
    assert task.status == "failed"


def test_act_requires_jev_and_fails_closed_when_service_is_unavailable() -> None:
    task = TaskState(goal="Continue", check_verdict="wip", acceptance_criteria_md="1.- [ ] Done")

    with pytest.raises(SystemOneUnavailableError, match="act_recommendation_unavailable"):
        act_node(task, CoreLoopContext(messages=InMemoryMessageQueue()))
