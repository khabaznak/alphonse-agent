from __future__ import annotations

from functools import partial
from typing import Any, Callable

from alphonse.agent.cognition.intent_types import IntentCategory
from alphonse.agent.cognition.preferences.store import get_or_create_principal_for_channel
from alphonse.agent.cortex.nodes.capability_gap import build_gap_plan
from alphonse.agent.cortex.nodes.critic import build_ask_question_executor
from alphonse.agent.cortex.nodes.discovery_loop import PlanningLoopDeps, run_planning_loop_step
from alphonse.agent.cortex.nodes.plan import next_step_index
from alphonse.agent.cortex.providers import get_ability_registry
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.cortex.utils import safe_json


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


def build_execute_tool_node(
    *,
    llm_client_from_state: Callable[[dict[str, Any]], Any],
    run_planning_loop_step_with_llm: Callable[[dict[str, Any], dict[str, Any], Any], dict[str, Any]],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    return partial(
        execute_tool_node_stateful,
        llm_client_from_state=llm_client_from_state,
        execute_tool_node_impl=execute_tool_node,
        run_planning_loop_step_with_llm=run_planning_loop_step_with_llm,
    )


def build_execute_stage_node(
    *,
    tool_registry: Any,
    llm_client_from_state: Callable[[dict[str, Any]], Any],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    ask_question_executor = build_ask_question_executor()
    return build_execute_tool_node(
        llm_client_from_state=llm_client_from_state,
        run_planning_loop_step_with_llm=lambda s, loop_state, llm_client: run_planning_loop_step(
            s,
            loop_state,
            llm_client,
            deps=PlanningLoopDeps(
                next_step_index=next_step_index,
                safe_json=lambda value, limit: safe_json(value, limit=limit),
                build_gap_plan=lambda *, state, reason, missing_slots: build_gap_plan(
                    state=state,
                    reason=reason,
                    missing_slots=missing_slots,
                    get_or_create_principal_for_channel=get_or_create_principal_for_channel,
                ),
                execute_tool_node=ask_question_executor,
                ability_registry_getter=get_ability_registry,
                tool_registry=tool_registry,
                emit_transition_event=emit_transition_event,
                task_plane_category=IntentCategory.TASK_PLANE.value,
            ),
        ),
    )
