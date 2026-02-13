from __future__ import annotations

from typing import Any, Callable


def plan_node(
    state: dict[str, Any],
    *,
    run_intent_discovery: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any]:
    """Run planning/replanning once and return state deltas."""
    if state.get("response_text") or state.get("response_key"):
        return {}
    result = run_intent_discovery(state)
    return result if isinstance(result, dict) else {}
