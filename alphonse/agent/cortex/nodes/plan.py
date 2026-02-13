from __future__ import annotations

from typing import Any, Callable


def plan_node(
    state: dict[str, Any],
    *,
    run_planning_cycle: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any]:
    """Run planning/replanning once and return state deltas."""
    if state.get("response_text") or state.get("response_key"):
        return {}
    result = run_planning_cycle(state)
    return result if isinstance(result, dict) else {}


def next_step_index(
    steps: list[dict[str, Any]],
    allowed_statuses: set[str],
) -> int | None:
    for idx, step in enumerate(steps):
        status = str(step.get("status") or "").strip().lower()
        if status in allowed_statuses:
            return idx
    return None


def plan_node_stateful(
    state: dict[str, Any],
    *,
    llm_client_from_state: Callable[[dict[str, Any]], Any],
    run_planning_cycle_with_llm: Callable[[dict[str, Any], Any], dict[str, Any] | None],
) -> dict[str, Any]:
    return plan_node(
        state,
        run_planning_cycle=lambda s: run_planning_cycle_with_llm(
            s,
            llm_client_from_state(s),
        ),
    )
