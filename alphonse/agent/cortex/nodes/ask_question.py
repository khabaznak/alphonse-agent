from __future__ import annotations

from typing import Any, Callable


def ask_question_node(
    state: dict[str, Any],
    *,
    run_ask_question_step: Callable[[dict[str, Any], dict[str, Any], dict[str, Any] | None, int | None], dict[str, Any]],
    next_step_index: Callable[[list[dict[str, Any]], set[str]], int | None],
) -> dict[str, Any]:
    """Emit question for askQuestion step and park on pending interaction."""
    loop_state = state.get("ability_state")
    if not isinstance(loop_state, dict):
        return {}
    steps = loop_state.get("steps")
    if not isinstance(steps, list) or not steps:
        return {}
    idx_raw = state.get("selected_step_index")
    idx = idx_raw if isinstance(idx_raw, int) else next_step_index(steps, {"ready"})
    if idx is None or idx < 0 or idx >= len(steps):
        return {}
    step = steps[idx]
    if not isinstance(step, dict):
        return {}
    if str(step.get("tool") or "").strip() != "askQuestion":
        return {}
    return run_ask_question_step(state, step, loop_state, idx)
