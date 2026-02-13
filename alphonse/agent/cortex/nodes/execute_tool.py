from __future__ import annotations

from typing import Any, Callable


def execute_tool_node(
    state: dict[str, Any],
    *,
    run_discovery_loop_step: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    """Execute one step from discovery-loop state."""
    loop_state = state.get("ability_state")
    if not isinstance(loop_state, dict):
        return {}
    return run_discovery_loop_step(state, loop_state)
