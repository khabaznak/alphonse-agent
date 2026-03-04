from __future__ import annotations

import os
from typing import Any, Callable

from alphonse.agent.cognition.first_decision_engine import decide_first_action
from alphonse.agent.cognition.tool_schemas import canonical_tool_names
from alphonse.agent.cortex.task_mode.act_node import evaluate_tool_execution
from alphonse.agent.cortex.task_mode.check_state import derive_outcome_from_state
from alphonse.agent.cortex.task_mode.check_state import goal_satisfied
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.cortex.task_mode.plan import build_next_step_node_impl
from alphonse.agent.cortex.task_mode.plan import route_after_next_step_impl
from alphonse.agent.cortex.task_mode.progress_critic_node import DEFAULT_PROGRESS_CHECK_CYCLE_THRESHOLD
from alphonse.agent.cortex.task_mode.progress_critic_node import DEFAULT_WIP_EMIT_EVERY_CYCLES
from alphonse.agent.cortex.task_mode.progress_critic_node import build_wip_update_detail
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
from alphonse.agent.session.day_state import render_recent_conversation_block
logger = get_component_logger("task_mode.pdca")

_PROGRESS_CHECK_CYCLE_THRESHOLD = DEFAULT_PROGRESS_CHECK_CYCLE_THRESHOLD
_WIP_EMIT_EVERY_CYCLES = DEFAULT_WIP_EMIT_EVERY_CYCLES
_PLANNER_RETRY_BUDGET_DEFAULT = 2


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
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "check"
    corr = correlation_id(state)
    cycle = int(task_state.get("cycle_index") or 0)

    if not has_acceptance_criteria(task_state):
        goal = str(task_state.get("goal") or "").strip()
        if goal:
            task_state["acceptance_criteria"] = [
                f"The request is satisfied for goal: {goal[:160]}"
            ]
            append_trace_event(
                task_state,
                {
                    "type": "acceptance_criteria_derived",
                    "summary": "Check node derived default acceptance criteria.",
                    "correlation_id": corr,
                },
            )

    status = str(task_state.get("status") or "").strip().lower()
    if cycle == 0 and status not in {"done", "failed", "waiting_user"}:
        llm_client = state.get("_llm_client")
        text = str(state.get("last_user_message") or "").strip()
        if llm_client and text:
            recent = str(state.get("recent_conversation_block") or "").strip()
            if not recent:
                session_state = state.get("session_state") if isinstance(state.get("session_state"), dict) else None
                if session_state:
                    recent = render_recent_conversation_block(session_state)
            decision = decide_first_action(
                text=text,
                llm_client=llm_client,
                locale=state.get("locale"),
                tone=state.get("tone"),
                address_style=state.get("address_style"),
                channel_type=state.get("channel_type"),
                available_tool_names=canonical_tool_names(tool_registry),
                recent_conversation_block=recent,
            )
            route = str(decision.get("route") or "").strip().lower()
            if route == "direct_reply":
                reply_text = str(decision.get("reply_text") or "").strip()
                if reply_text:
                    task_state["status"] = ""
                    task_state["check_route"] = "respond_node"
                    return {"task_state": task_state, "response_text": reply_text}
            elif route == "clarify":
                question = str(decision.get("clarify_question") or "").strip()
                if question:
                    task_state["status"] = "waiting_user"
                    task_state["next_user_question"] = question

    planner_error = task_state.get("planner_error_last") if isinstance(task_state.get("planner_error_last"), dict) else None
    if planner_error:
        streak = int(task_state.get("planner_error_streak") or 0) + 1
        task_state["planner_error_streak"] = streak
        budget = _planner_retry_budget()
        if streak <= budget:
            task_state["status"] = "running"
            task_state["next_user_question"] = None
            task_state["last_validation_error"] = None
            append_trace_event(
                task_state,
                {
                    "type": "planner_retry",
                    "summary": f"Planner output invalid; retrying planning ({streak}/{budget}).",
                    "correlation_id": corr,
                },
            )
            log_task_event(
                logger=logger,
                state=state,
                task_state=task_state,
                node="check_node",
                event="graph.check.planner_retry",
                planner_error_streak=streak,
                planner_retry_budget=budget,
                error_code=str(planner_error.get("code") or ""),
                level="warning",
            )
            task_state["check_route"] = "next_step_node"
            return {"task_state": task_state}

        task_state["status"] = "failed"
        task_state["next_user_question"] = None
        task_state["last_validation_error"] = dict(planner_error)
        append_trace_event(
            task_state,
            {
                "type": "status_changed",
                "summary": "Check marked task failed after planner retry budget exhausted.",
                "correlation_id": corr,
            },
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="graph.check.planner_failed",
            planner_error_streak=streak,
            planner_retry_budget=budget,
            error_code=str(planner_error.get("code") or ""),
            level="warning",
        )
    else:
        if int(task_state.get("planner_error_streak") or 0) != 0:
            task_state["planner_error_streak"] = 0
        if task_state.get("planner_error_last") is not None:
            task_state["planner_error_last"] = None

    current = current_step(task_state)
    current_status = str((current or {}).get("status") or "").strip().lower()
    if current_status == "failed":
        evaluation = evaluate_tool_execution(task_state=task_state, current_step=current, task_plan=task_plan)
        task_state["execution_eval"] = evaluation if isinstance(evaluation, dict) else {}
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="graph.check.failure_evaluated",
            step_id=str((current or {}).get("step_id") or ""),
            reason=str((evaluation or {}).get("reason") or ""),
            total_failures=int((evaluation or {}).get("total_failures") or 0),
            same_signature_failures=int((evaluation or {}).get("same_signature_failures") or 0),
            unique_signatures=int((evaluation or {}).get("unique_signatures") or 0),
            level="warning",
        )

    if int(task_state.get("cycle_index") or 0) > 0:
        emit_every = max(1, int(_WIP_EMIT_EVERY_CYCLES))
        step_status = str((current or {}).get("status") or "").strip().lower()
        if step_status == "proposed" and int(task_state.get("cycle_index") or 0) % emit_every == 0:
            detail = build_wip_update_detail(
                task_state=task_state,
                cycle=int(task_state.get("cycle_index") or 0),
                current_step=current,
            )
            emit_transition_event(state, "wip_update", detail)
            log_task_event(
                logger=logger,
                state=state,
                task_state=task_state,
                node="check_node",
                event="graph.check.wip_update",
                cycle=int(detail.get("cycle") or 0),
                tool=str(detail.get("tool") or ""),
                intention=str(detail.get("intention") or ""),
            )

    if goal_satisfied(task_state, has_acceptance_criteria=has_acceptance_criteria):
        if str(task_state.get("status") or "").strip().lower() != "done":
            append_trace_event(
                task_state,
                {
                    "type": "status_changed",
                    "summary": "Check marked task as done from outcome evidence.",
                    "correlation_id": corr,
                },
            )
        task_state["status"] = "done"
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="graph.task.completed",
        )

    status = str(task_state.get("status") or "").strip().lower()
    if status in {"done", "waiting_user", "failed"}:
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="graph.check.skipped",
            status=status,
            cycle=cycle,
            level="debug",
        )
    else:
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="graph.check.continue",
            cycle=cycle,
            level="debug",
        )
    check_route = "respond_node" if status in {"waiting_user", "failed", "done"} else "next_step_node"
    task_state["check_route"] = check_route
    logger.info(
        "task_mode check correlation_id=%s cycle=%s status=%s route=%s",
        corr,
        cycle,
        status or "running",
        check_route,
    )
    return {"task_state": task_state}


def _planner_retry_budget() -> int:
    raw = str(os.getenv("ALPHONSE_TASK_MODE_PLANNER_RETRY_BUDGET") or "").strip()
    if not raw:
        return _PLANNER_RETRY_BUDGET_DEFAULT
    try:
        parsed = int(raw)
    except ValueError:
        return _PLANNER_RETRY_BUDGET_DEFAULT
    return parsed if parsed >= 0 else 0


def route_after_check(state: dict[str, Any]) -> str:
    task_state = state.get("task_state") if isinstance(state.get("task_state"), dict) else {}
    route = str(task_state.get("check_route") or "").strip().lower()
    if route in {"respond_node", "next_step_node"}:
        return route
    status = str(task_state.get("status") or "").strip().lower()
    return "respond_node" if status in {"waiting_user", "failed", "done"} else "next_step_node"


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
