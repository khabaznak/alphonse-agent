from __future__ import annotations

from alphonse.agent_v2.core.intelligence.pdca.nodes.act_node import act_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState


def test_act_routes_new_mission_to_plan_without_inference():
    task = TaskState(goal="Record exercise", check_verdict="new")
    act_node(task)
    assert task.metadata["act_route"] == "plan"
    assert "acceptance_contract" not in task.metadata


def test_act_hands_unfinished_work_and_the_check_reason_to_plan():
    task = TaskState(goal="Connect to server", check_verdict="wip", check_reason="Connection timed out")
    act_node(task)
    assert task.metadata["act_route"] == "plan"
    assert task.metadata["act_directive"] == {"action": "replan", "rationale": "Connection timed out"}


def test_act_schedules_one_closure_response_before_final_failure():
    task = TaskState(goal="Connect to server", check_verdict="mission_failed", check_reason="Server unreachable")
    act_node(task)
    assert task.metadata["act_route"] == "plan"
    assert task.metadata["act_directive"]["closure_only"] is True
    assert task.metadata["act_directive"]["terminal_outcome"] == "failed"
    assert task.metadata["act_directive"]["reason"] == "Server unreachable"
    assert task.status == "running"

    task.metadata["prepared_user_response"] = {"message": "The server was unreachable."}
    act_node(task)
    assert task.metadata["act_route"] == "end"
    assert task.status == "failed"
