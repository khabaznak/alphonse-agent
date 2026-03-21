from __future__ import annotations

from typing import Any, TypedDict

from alphonse.agent.cortex.task_mode.types import AcceptanceCriterion
from alphonse.agent.cortex.task_mode.types import JudgeVerdict


class TaskState(TypedDict, total=False):
    mode: str
    goal: str
    facts: dict[str, Any]
    plan: dict[str, Any]
    trace: dict[str, Any]
    last_validation_error: dict[str, Any] | None
    repair_attempts: int
    status: str
    outcome: dict[str, Any] | None
    next_user_question: str | None
    pdca_phase: str
    cycle_index: int
    initialized: bool
    acceptance_criteria: list[AcceptanceCriterion]
    pending_plan_raw: Any | None
    current_plan_step: dict[str, Any] | None
    pending_control_tool_call: dict[str, Any] | None
    success_evaluation_last: dict[str, Any] | None
    completion_decision: dict[str, Any] | None
    zero_progress_last_signature: str | None
    zero_progress_streak: int
    planner_error_streak: int
    planner_error_last: dict[str, Any] | None
    planner_error_last_fact_key: str | None
    check_decision_last: dict[str, Any] | None
    check_provenance: str | None
    judge_verdict: JudgeVerdict | None
    judge_invalid_streak: int
    steering_consumed_in_check: bool


def build_default_task_state() -> TaskState:
    return {
        "mode": "task",
        "goal": "",
        "facts": {},
        "plan": {
            "version": 1,
            "steps": [],
            "current_step_id": None,
        },
        "trace": {
            "summary": "",
            "recent": [],
        },
        "last_validation_error": None,
        "repair_attempts": 0,
        "status": "running",
        "outcome": None,
        "next_user_question": None,
        "pdca_phase": "plan",
        "cycle_index": 0,
        "initialized": True,
        "acceptance_criteria": [],
        "pending_plan_raw": None,
        "current_plan_step": None,
        "pending_control_tool_call": None,
        "success_evaluation_last": None,
        "completion_decision": None,
        "zero_progress_last_signature": None,
        "zero_progress_streak": 0,
        "planner_error_streak": 0,
        "planner_error_last": None,
        "planner_error_last_fact_key": None,
        "check_decision_last": None,
        "check_provenance": "entry",
        "judge_verdict": None,
        "judge_invalid_streak": 0,
        "steering_consumed_in_check": False,
    }
