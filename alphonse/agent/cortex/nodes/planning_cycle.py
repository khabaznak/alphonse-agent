from __future__ import annotations

import logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Any, Callable

from alphonse.agent.cognition.planning_catalog import discover_plan, format_available_abilities
from alphonse.agent.cortex.nodes.apology import run_capability_gap_tool
from alphonse.agent.cortex.nodes.ask_question import bind_answer_to_steps
from alphonse.agent.cortex.nodes.plan import next_step_index
from alphonse.agent.cortex.nodes.planning_fresh import (
    DispatchCycleDeps,
    FreshPlanningDeps,
    build_planning_loop_state,
    dispatch_cycle_result,
    run_fresh_planning_pass,
)
from alphonse.agent.cortex.nodes.planning_pending import (
    EmptyCycleResultDeps,
    PendingCycleDeps,
    handle_pending_interaction_for_cycle,
    is_effectively_empty_cycle_result,
)
from alphonse.agent.cortex.providers import get_ability_registry

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlanningCycleDeps:
    require_last_user_message: Callable[[dict[str, Any]], str]
    handle_pending_interaction_for_cycle: Callable[..., dict[str, Any] | None]
    handle_ability_state_for_cycle: Callable[..., dict[str, Any] | None]
    run_fresh_planning_pass: Callable[..., dict[str, Any] | None]


@dataclass(frozen=True)
class AbilityStateCycleDeps:
    is_planning_loop_state: Callable[[dict[str, Any]], bool]
    ability_registry_getter: Callable[[], Any]
    tool_registry: Any
    logger_info: Callable[[str, Any, Any, str, str], None]


def build_planning_cycle_runner(
    *,
    tool_registry: Any,
) -> Callable[[dict[str, Any], Any], dict[str, Any] | None]:
    def _runner(state: dict[str, Any], llm_client: Any) -> dict[str, Any] | None:
        return run_planning_cycle(
            state,
            llm_client,
            deps=PlanningCycleDeps(
                require_last_user_message=_require_last_user_message,
                handle_pending_interaction_for_cycle=lambda **kwargs: handle_pending_interaction_for_cycle(
                    **kwargs,
                    deps=PendingCycleDeps(
                        now_iso_utc=lambda: datetime.now(timezone.utc).isoformat(),
                        is_effectively_empty_cycle_result=lambda result: is_effectively_empty_cycle_result(
                            result,
                            deps=EmptyCycleResultDeps(
                                is_planning_loop_state=_is_planning_loop_state,
                                next_step_index=next_step_index,
                            ),
                        ),
                        is_planning_loop_state=_is_planning_loop_state,
                        next_step_index=next_step_index,
                        bind_answer_to_steps=bind_answer_to_steps,
                        ability_registry_getter=get_ability_registry,
                        tool_registry=tool_registry,
                        logger_info=lambda msg, chat_id, correlation_id: logger.info(
                            msg,
                            chat_id,
                            correlation_id,
                        ),
                    ),
                ),
                handle_ability_state_for_cycle=lambda **kwargs: handle_ability_state_for_cycle(
                    **kwargs,
                    deps=AbilityStateCycleDeps(
                        is_planning_loop_state=_is_planning_loop_state,
                        ability_registry_getter=get_ability_registry,
                        tool_registry=tool_registry,
                        logger_info=lambda msg, chat_id, correlation_id, prev_message, new_message: logger.info(
                            msg,
                            chat_id,
                            correlation_id,
                            prev_message,
                            new_message,
                        ),
                    ),
                ),
                run_fresh_planning_pass=lambda **kwargs: run_fresh_planning_pass(
                    **kwargs,
                    deps=FreshPlanningDeps(
                        run_capability_gap_tool=run_capability_gap_tool,
                        format_available_abilities=format_available_abilities,
                        planning_context_for_cycle=lambda s, _text: (
                            s.get("planning_context")
                            if isinstance(s.get("planning_context"), dict)
                            else None
                        ),
                        discover_plan=discover_plan,
                        locale_for_state=lambda s: (
                            s.get("locale")
                            if isinstance(s.get("locale"), str) and s.get("locale")
                            else "en-US"
                        ),
                        dispatch_cycle_result=lambda **dispatch_kwargs: dispatch_cycle_result(
                            **dispatch_kwargs,
                            deps=DispatchCycleDeps(
                                run_capability_gap_tool=run_capability_gap_tool,
                                build_planning_loop_state=build_planning_loop_state,
                            ),
                        ),
                    ),
                ),
            ),
        )

    return _runner


def run_planning_cycle(
    state: dict[str, Any],
    llm_client: Any,
    deps: PlanningCycleDeps,
) -> dict[str, Any] | None:
    text = deps.require_last_user_message(state)

    pending_result = deps.handle_pending_interaction_for_cycle(
        state=state,
        llm_client=llm_client,
        text=text,
    )
    if pending_result is not None:
        return pending_result

    ability_state_result = deps.handle_ability_state_for_cycle(
        state=state,
        llm_client=llm_client,
        text=text,
    )
    if ability_state_result is not None:
        return ability_state_result

    return deps.run_fresh_planning_pass(
        state=state,
        llm_client=llm_client,
        text=text,
    )


def handle_ability_state_for_cycle(
    *,
    state: dict[str, Any],
    llm_client: Any,
    text: str,
    deps: AbilityStateCycleDeps,
) -> dict[str, Any] | None:
    _ = llm_client
    ability_state = state.get("ability_state")
    if not isinstance(ability_state, dict):
        return None
    if deps.is_planning_loop_state(ability_state):
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


def _require_last_user_message(state: dict[str, Any]) -> str:
    raw = state.get("last_user_message")
    if not isinstance(raw, str):
        raise TypeError("last_user_message must be a string in CortexState")
    text = raw.strip()
    if not text:
        raise ValueError("last_user_message must be non-empty")
    return text


def _is_planning_loop_state(ability_state: dict[str, Any]) -> bool:
    return str(ability_state.get("kind") or "") == "discovery_loop"
