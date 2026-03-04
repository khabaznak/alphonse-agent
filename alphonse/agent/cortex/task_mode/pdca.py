from __future__ import annotations

from typing import Any, Callable

from alphonse.agent.cortex.task_mode.act_node import evaluate_tool_execution
from alphonse.agent.cortex.task_mode.check import check_node_impl
from alphonse.agent.cortex.task_mode.check import route_after_check_impl
from alphonse.agent.cortex.task_mode.check_state import derive_outcome_from_state
from alphonse.agent.cortex.task_mode.check_state import goal_satisfied
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.cortex.task_mode.plan import build_next_step_node_impl
from alphonse.agent.cortex.task_mode.plan import route_after_next_step_impl
from alphonse.agent.cortex.task_mode.progress_critic_node import DEFAULT_PROGRESS_CHECK_CYCLE_THRESHOLD
from alphonse.agent.cortex.task_mode.progress_critic_node import DEFAULT_WIP_EMIT_EVERY_CYCLES
from alphonse.agent.cortex.task_mode.progress_critic_node import progress_critic_node_stateful
from alphonse.agent.cortex.task_mode.progress_critic_node import route_after_progress_critic_stateful
from alphonse.agent.cortex.task_mode.observability import log_task_event
from alphonse.agent.cortex.task_mode.execute_step import execute_step_node_impl
from alphonse.agent.cortex.task_mode.task_state_helpers import append_trace_event
from alphonse.agent.cortex.task_mode.task_state_helpers import correlation_id
from alphonse.agent.cortex.task_mode.task_state_helpers import current_step
from alphonse.agent.cortex.task_mode.task_state_helpers import has_acceptance_criteria
from alphonse.agent.cortex.task_mode.task_state_helpers import next_step_id
from alphonse.agent.cortex.task_mode.task_state_helpers import normalize_acceptance_criteria_values
from alphonse.agent.cortex.task_mode.task_state_helpers import task_plan
from alphonse.agent.cortex.task_mode.task_state_helpers import task_state_with_defaults
from alphonse.agent.cortex.task_mode.task_state_helpers import task_trace
from alphonse.agent.observability.log_manager import get_component_logger
logger = get_component_logger("task_mode.pdca")

_PROGRESS_CHECK_CYCLE_THRESHOLD = DEFAULT_PROGRESS_CHECK_CYCLE_THRESHOLD
_WIP_EMIT_EVERY_CYCLES = DEFAULT_WIP_EMIT_EVERY_CYCLES


def build_next_step_node(*, tool_registry: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _node(state: dict[str, Any]) -> dict[str, Any]:
        emit_transition_event(state, "thinking")
        return build_next_step_node_impl(
            state=state,
            tool_registry=tool_registry,
            task_state_with_defaults=task_state_with_defaults,
            correlation_id=correlation_id,
            next_step_id=next_step_id,
            task_plan=task_plan,
            has_acceptance_criteria=has_acceptance_criteria,
            normalize_acceptance_criteria_values=normalize_acceptance_criteria_values,
            append_trace_event=append_trace_event,
            logger=logger,
            log_task_event=log_task_event,
        )

    return _node


def route_after_next_step(state: dict[str, Any]) -> str:
    return route_after_next_step_impl(
        state,
        correlation_id=correlation_id,
        logger=logger,
    )


def validate_step_node(state: dict[str, Any], *, tool_registry: Any) -> dict[str, Any]:
    # Compatibility wrapper: validation stage was removed from runtime wiring.
    _ = tool_registry
    return {"task_state": task_state_with_defaults(state)}


def route_after_validate_step(state: dict[str, Any]) -> str:
    # Compatibility wrapper: runtime path now routes through check -> act.
    task_state = state.get("task_state")
    if isinstance(task_state, dict) and str(task_state.get("status") or "").strip().lower() == "waiting_user":
        return "respond_node"
    return "execute_step_node"


def execute_step_node(state: dict[str, Any], *, tool_registry: Any) -> dict[str, Any]:
    emit_transition_event(state, "executing")
    return execute_step_node_impl(
        state,
        tool_registry=tool_registry,
        task_state_with_defaults=task_state_with_defaults,
        correlation_id=correlation_id,
        current_step=current_step,
        append_trace_event=append_trace_event,
        logger=logger,
        log_task_event=log_task_event,
    )


def check_node(state: dict[str, Any], *, tool_registry: Any) -> dict[str, Any]:
    return check_node_impl(
        state=state,
        tool_registry=tool_registry,
        logger=logger,
        log_task_event=log_task_event,
        wip_emit_every_cycles=_WIP_EMIT_EVERY_CYCLES,
    )


def route_after_check(state: dict[str, Any]) -> str:
    return route_after_check_impl(state)


def update_state_node(state: dict[str, Any]) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "check"
    corr = correlation_id(state)
    task_state["cycle_index"] = int(task_state.get("cycle_index") or 0) + 1
    outcome = derive_outcome_from_state(state=state, task_state=task_state, current_step=current_step)
    task_state["outcome"] = outcome
    trace = task_trace(task_state)
    trace["summary"] = f"PDCA cycle {task_state['cycle_index']} complete."
    append_trace_event(
        task_state,
        {
            "type": "state_updated",
            "summary": f"State updated at cycle {task_state['cycle_index']}.",
            "correlation_id": corr,
        },
    )
    logger.info(
        "task_mode update_state correlation_id=%s cycle=%s status=%s has_outcome=%s",
        corr,
        int(task_state.get("cycle_index") or 0),
        str(task_state.get("status") or ""),
        bool(outcome),
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="update_state_node",
        event="graph.state.updated",
        has_outcome=bool(outcome),
    )
    return {"task_state": task_state}


def progress_critic_node(state: dict[str, Any]) -> dict[str, Any]:
    return progress_critic_node_stateful(
        state,
        task_state_with_defaults=task_state_with_defaults,
        correlation_id=correlation_id,
        current_step=current_step,
        goal_satisfied=lambda task_state: goal_satisfied(
            task_state,
            has_acceptance_criteria=has_acceptance_criteria,
        ),
        evaluate_tool_execution=lambda *, task_state, current_step: evaluate_tool_execution(
            task_state=task_state,
            current_step=current_step,
            task_plan=task_plan,
        ),
        append_trace_event=append_trace_event,
        logger=logger,
        progress_check_cycle_threshold=_PROGRESS_CHECK_CYCLE_THRESHOLD,
        wip_emit_every_cycles=_WIP_EMIT_EVERY_CYCLES,
    )


def route_after_progress_critic(state: dict[str, Any]) -> str:
    return route_after_progress_critic_stateful(state, correlation_id=correlation_id)


def act_node(state: dict[str, Any]) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "act"
    return {"task_state": task_state}


def route_after_act(state: dict[str, Any]) -> str:
    task_state = state.get("task_state") if isinstance(state.get("task_state"), dict) else {}
    route = str(task_state.get("check_route") or "").strip().lower()
    if route in {"respond_node", "next_step_node"}:
        logger.info(
            "task_mode route_after_act correlation_id=%s route=%s",
            correlation_id(state),
            route,
        )
        return route
    logger.info(
        "task_mode route_after_act correlation_id=%s route=next_step_node",
        correlation_id(state),
    )
    return "next_step_node"
