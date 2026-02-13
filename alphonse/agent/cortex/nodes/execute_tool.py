from __future__ import annotations

from typing import Any, Callable


def execute_tool_node(
    state: dict[str, Any],
    *,
    run_planning_loop_step: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Execute one step from discovery-loop state."""
    loop_state = state.get("ability_state")
    if not isinstance(loop_state, dict):
        return {}
    return run_planning_loop_step(state, loop_state)


def execute_tool_node_stateful(
    state: dict[str, Any],
    *,
    llm_client_from_state: Callable[[dict[str, Any]], Any],
    execute_tool_node_impl: Callable[..., dict[str, Any]],
    run_planning_loop_step_with_llm: Callable[[dict[str, Any], dict[str, Any], Any], dict[str, Any]],
) -> dict[str, Any]:
    llm_client = llm_client_from_state(state)
    return execute_tool_node_impl(
        state,
        run_planning_loop_step=lambda s, loop_state: run_planning_loop_step_with_llm(
            s,
            loop_state,
            llm_client,
        ),
    )
