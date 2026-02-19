from __future__ import annotations

from typing import Any, TypedDict


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
    acceptance_criteria: list[str]


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
    }
