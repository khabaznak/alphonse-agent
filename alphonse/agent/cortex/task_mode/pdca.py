from __future__ import annotations

from typing import Any, Callable

from alphonse.agent.cognition.providers.factory import build_text_completion_provider
from alphonse.agent.cognition.providers.factory import build_tool_calling_provider
from alphonse.agent.cortex.task_mode.check import check_node_impl
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.cortex.task_mode.plan import build_next_step_node_impl
from alphonse.agent.cortex.task_mode.plan import route_after_next_step_impl
from alphonse.agent.cortex.task_mode.observability import log_task_event
from alphonse.agent.cortex.task_mode.execute_step import execute_step_node_impl
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.cortex.task_mode.task_state_helpers import append_trace_event
from alphonse.agent.cortex.task_mode.task_state_helpers import correlation_id
from alphonse.agent.cortex.task_mode.task_state_helpers import current_step
from alphonse.agent.cortex.task_mode.task_state_helpers import has_acceptance_criteria
from alphonse.agent.cortex.task_mode.task_state_helpers import next_step_id
from alphonse.agent.cortex.task_mode.task_state_helpers import normalize_acceptance_criteria_values
from alphonse.agent.cortex.task_mode.task_state_helpers import task_plan
from alphonse.agent.cortex.task_mode.task_state_helpers import task_metrics
from alphonse.agent.cortex.task_mode.task_state_helpers import task_state_with_defaults
from alphonse.agent.cortex.task_mode.task_state_helpers import task_trace
from alphonse.agent.observability.log_manager import get_component_logger
logger = get_component_logger("task_mode.pdca")

_TRANSITION_EVENT_THINKING = "thinking"
_TRANSITION_EVENT_EXECUTING = "executing"

def build_next_step_node(*, tool_registry: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _node(state: dict[str, Any]) -> dict[str, Any]:
        emit_transition_event(state, _TRANSITION_EVENT_THINKING)
        llm_client, llm_error = _resolve_plan_provider()
        return build_next_step_node_impl(
            state=state,
            llm_client=llm_client,
            llm_error=llm_error,
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
    emit_transition_event(state, _TRANSITION_EVENT_EXECUTING)
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


def check_node(task_record: TaskRecord, *, provenance: str) -> dict[str, Any]:
    result = check_node_impl(
        task_record,
        provenance=provenance,
        llm_client=_resolve_check_provider(),
        logger=logger,
        log_task_event=log_task_event,
    )
    return result


def route_after_check(state: dict[str, Any]) -> str:
    check_result = state.get("check_result") if isinstance(state.get("check_result"), dict) else {}
    verdict = str(check_result.get("verdict") or "").strip().lower()
    if verdict == "plan":
        return "next_step_node"
    if verdict in {"mission_success", "mission_failed"}:
        return "respond_node"
    task_record = state.get("task_record")
    status = str(task_record.status or "").strip().lower() if isinstance(task_record, TaskRecord) else ""
    return "respond_node" if status in {"waiting_user", "failed", "done"} else "next_step_node"


def _resolve_plan_provider() -> tuple[Any, str | None]:
    try:
        return build_tool_calling_provider(), None
    except Exception as exc:
        return None, str(exc)


def _resolve_check_provider():
    try:
        return build_text_completion_provider()
    except Exception:
        return None


def update_state_node(state: dict[str, Any]) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_record = state.get("task_record") if isinstance(state.get("task_record"), TaskRecord) else None
    task_state["pdca_phase"] = "check"
    corr = correlation_id(state)
    task_state["cycle_index"] = int(task_state.get("cycle_index") or 0) + 1
    outcome = task_record.outcome if isinstance(task_record, TaskRecord) and isinstance(task_record.outcome, dict) else None
    task_state["outcome"] = outcome
    if isinstance(task_record, TaskRecord):
        task_state["status"] = str(task_record.status or "").strip() or task_state.get("status") or "running"
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


def act_node(state: dict[str, Any]) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "act"
    return {"task_state": task_state}


def route_after_act(state: dict[str, Any]) -> str:
    check_result = state.get("check_result") if isinstance(state.get("check_result"), dict) else {}
    verdict_kind = str(check_result.get("verdict") or "").strip().lower()
    if verdict_kind == "plan":
        logger.info(
            "task_mode route_after_act correlation_id=%s route=next_step_node verdict=%s",
            correlation_id(state),
            verdict_kind,
        )
        return "next_step_node"
    if verdict_kind in {"mission_success", "mission_failed"}:
        logger.info(
            "task_mode route_after_act correlation_id=%s route=respond_node verdict=%s",
            correlation_id(state),
            verdict_kind,
        )
        return "respond_node"
    task_record = state.get("task_record") if isinstance(state.get("task_record"), TaskRecord) else None
    status = str(task_record.status or "").strip().lower() if isinstance(task_record, TaskRecord) else ""
    if status in {"done", "failed", "waiting_user"}:
        logger.info(
            "task_mode route_after_act correlation_id=%s route=respond_node status=%s",
            correlation_id(state),
            status,
        )
        return "respond_node"
    logger.info(
        "task_mode route_after_act correlation_id=%s route=next_step_node missing_check_route=true",
        correlation_id(state),
    )
    return "next_step_node"
