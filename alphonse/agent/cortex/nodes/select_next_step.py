from __future__ import annotations

from typing import Any, Callable

from alphonse.agent.cortex.nodes.plan import next_step_index as _next_step_index


def select_next_step_node(
    state: dict[str, Any],
    *,
    next_step_index: Callable[[list[dict[str, Any]], set[str]], int | None],
    is_planning_loop_state: Callable[[dict[str, Any]], bool],
) -> dict[str, Any]:
    """Pick next step and emit routing decision for conditional edges."""
    ability_state = state.get("ability_state")
    if not isinstance(ability_state, dict) or not is_planning_loop_state(ability_state):
        return {"route_decision": "respond", "selected_step_index": None}
    steps = ability_state.get("steps")
    if not isinstance(steps, list) or not steps:
        return {"route_decision": "respond", "selected_step_index": None}
    idx = next_step_index(steps, {"ready"})
    if idx is None or idx < 0 or idx >= len(steps):
        return {"route_decision": "respond", "selected_step_index": None}
    step = steps[idx]
    tool_name = str(step.get("tool") or "").strip()
    if tool_name == "askQuestion":
        return {"route_decision": "ask_question", "selected_step_index": idx}
    return {"route_decision": "execute_tool", "selected_step_index": idx}


def select_next_step_node_stateful(state: dict[str, Any]) -> dict[str, Any]:
    return select_next_step_node(
        state,
        next_step_index=_next_step_index,
        is_planning_loop_state=_is_planning_loop_state,
    )


def _is_planning_loop_state(ability_state: dict[str, Any]) -> bool:
    return str(ability_state.get("kind") or "") == "discovery_loop"
