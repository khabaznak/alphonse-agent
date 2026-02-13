from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PlanningDiscoveryDeps:
    require_last_user_message: Callable[[dict[str, Any]], str]
    handle_pending_interaction_for_discovery: Callable[..., dict[str, Any] | None]
    handle_ability_state_for_discovery: Callable[..., dict[str, Any] | None]
    run_fresh_discovery_for_message: Callable[..., dict[str, Any] | None]


def run_intent_discovery(
    state: dict[str, Any],
    llm_client: Any,
    deps: PlanningDiscoveryDeps,
) -> dict[str, Any] | None:
    text = deps.require_last_user_message(state)

    pending_result = deps.handle_pending_interaction_for_discovery(
        state=state,
        llm_client=llm_client,
        text=text,
    )
    if pending_result is not None:
        return pending_result

    ability_state_result = deps.handle_ability_state_for_discovery(
        state=state,
        llm_client=llm_client,
        text=text,
    )
    if ability_state_result is not None:
        return ability_state_result

    return deps.run_fresh_discovery_for_message(
        state=state,
        llm_client=llm_client,
        text=text,
    )
