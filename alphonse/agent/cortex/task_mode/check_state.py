from __future__ import annotations

from typing import Any, Callable


def goal_satisfied(
    task_state: dict[str, Any],
    *,
    has_acceptance_criteria: Callable[[dict[str, Any]], bool],
) -> bool:
    outcome = task_state.get("outcome")
    if not isinstance(outcome, dict) or not outcome:
        return False
    kind = str(outcome.get("kind") or "").strip().lower()
    if kind == "task_completed":
        summary = str(
            outcome.get("final_text")
            or outcome.get("summary")
            or ""
        ).strip()
        if not summary:
            return False
        return has_acceptance_criteria(task_state)
    return True


def derive_outcome_from_state(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any] | None:
    _ = state
    _ = current_step
    return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None
