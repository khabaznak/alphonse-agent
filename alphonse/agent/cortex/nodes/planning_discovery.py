from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PlanningDiscoveryDeps:
    require_last_user_message: Callable[[dict[str, Any]], str]
    handle_pending_interaction_for_discovery: Callable[..., dict[str, Any] | None]
    handle_ability_state_for_discovery: Callable[..., dict[str, Any] | None]
    maybe_run_discovery_loop_result: Callable[..., dict[str, Any] | None]
    run_fresh_discovery_for_message: Callable[..., dict[str, Any] | None]


def run_intent_discovery(
    state: dict[str, Any],
    llm_client: Any,
    *,
    execute_until_blocked: bool,
    deps: PlanningDiscoveryDeps,
) -> dict[str, Any] | None:
    text = deps.require_last_user_message(state)

    pending_result = deps.handle_pending_interaction_for_discovery(
        state=state,
        llm_client=llm_client,
        text=text,
    )
    if pending_result is not None:
        return deps.maybe_run_discovery_loop_result(
            state=state,
            llm_client=llm_client,
            result=pending_result,
            execute_until_blocked=execute_until_blocked,
        )

    ability_state_result = deps.handle_ability_state_for_discovery(
        state=state,
        llm_client=llm_client,
        text=text,
    )
    if ability_state_result is not None:
        return deps.maybe_run_discovery_loop_result(
            state=state,
            llm_client=llm_client,
            result=ability_state_result,
            execute_until_blocked=execute_until_blocked,
        )

    return deps.maybe_run_discovery_loop_result(
        state=state,
        llm_client=llm_client,
        result=deps.run_fresh_discovery_for_message(
            state=state,
            llm_client=llm_client,
            text=text,
        ),
        execute_until_blocked=execute_until_blocked,
    )
