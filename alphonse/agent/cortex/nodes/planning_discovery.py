from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class PlanningDiscoveryDeps:
    require_last_user_message: Callable[[dict[str, Any]], str]
    handle_pending_interaction_for_discovery: Callable[..., dict[str, Any] | None]
    handle_ability_state_for_discovery: Callable[..., dict[str, Any] | None]
    run_fresh_discovery_for_message: Callable[..., dict[str, Any] | None]


@dataclass(frozen=True)
class AbilityStateDiscoveryDeps:
    is_discovery_loop_state: Callable[[dict[str, Any]], bool]
    ability_registry_getter: Callable[[], Any]
    tool_registry: Any
    logger_info: Callable[[str, Any, Any, str, str], None]


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


def handle_ability_state_for_discovery(
    *,
    state: dict[str, Any],
    llm_client: Any,
    text: str,
    deps: AbilityStateDiscoveryDeps,
) -> dict[str, Any] | None:
    _ = llm_client
    ability_state = state.get("ability_state")
    if not isinstance(ability_state, dict):
        return None
    if deps.is_discovery_loop_state(ability_state):
        source_message = str(ability_state.get("source_message") or "").strip()
        if source_message and source_message != text:
            deps.logger_info(
                "cortex clearing stale discovery_loop chat_id=%s correlation_id=%s prev_message=%s new_message=%s",
                state.get("chat_id"),
                state.get("correlation_id"),
                source_message,
                text,
            )
            state["ability_state"] = {}
            ability_state = {}
        else:
            return {"ability_state": ability_state}
    intent_name = str(ability_state.get("intent") or "")
    if intent_name:
        ability = deps.ability_registry_getter().get(intent_name)
        if ability is not None:
            state["intent"] = intent_name
            return ability.execute(state, deps.tool_registry)
    return None
