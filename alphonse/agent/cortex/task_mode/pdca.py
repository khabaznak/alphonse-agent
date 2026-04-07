from __future__ import annotations

from typing import Any, Callable

from alphonse.agent.cognition.providers.factory import build_text_completion_provider
from alphonse.agent.cognition.providers.factory import build_tool_calling_provider
from alphonse.agent.cortex.task_mode.act_node import ActResult
from alphonse.agent.cortex.task_mode.act_node import act_node_impl
from alphonse.agent.cortex.task_mode.check import CheckResult
from alphonse.agent.cortex.task_mode.check import check_node_impl
from alphonse.agent.cortex.task_mode.execute_step import DoResult
from alphonse.agent.cortex.task_mode.execute_step import execute_step_node_impl
from alphonse.agent.cortex.task_mode.observability import log_task_event
from alphonse.agent.cortex.task_mode.plan import PlannerOutput
from alphonse.agent.cortex.task_mode.plan import plan_node_impl
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("task_mode.pdca")

_TRANSITION_EVENT_THINKING = "thinking"
_TRANSITION_EVENT_EXECUTING = "executing"


def build_next_step_node(*, tool_registry: Any) -> Callable[[TaskRecord], PlannerOutput]:
    _ = tool_registry

    def _node(task_record: TaskRecord) -> PlannerOutput:
        _emit_transition_event_for_task_record(task_record, _TRANSITION_EVENT_THINKING)
        llm_client, llm_error = _resolve_plan_provider()
        if llm_error:
            logger.warning(
                "task_mode planner provider unavailable task_id=%s error=%s",
                str(task_record.task_id or ""),
                llm_error,
            )
        return plan_node_impl(
            task_record,
            llm_client=llm_client,
            logger=logger,
            log_task_event=log_task_event,
        )

    return _node


def route_after_next_step(planner_output: Any) -> str:
    _ = planner_output
    return "execute_step_node"


def execute_step_node(task_record: TaskRecord, planner_output: PlannerOutput) -> DoResult:
    _emit_transition_event_for_task_record(task_record, _TRANSITION_EVENT_EXECUTING)
    return execute_step_node_impl(
        task_record,
        planner_output,
        logger=logger,
        log_task_event=log_task_event,
    )


def check_node(task_record: TaskRecord, *, provenance: str) -> CheckResult:
    result = check_node_impl(
        task_record,
        provenance=provenance,
        llm_client=_resolve_check_provider(),
        logger=logger,
        log_task_event=log_task_event,
    )
    return result


def route_after_check(check_result: Any) -> str:
    verdict = ""
    if isinstance(check_result, dict):
        verdict = str(check_result.get("verdict") or "").strip().lower()
    if verdict == "plan":
        return "next_step_node"
    if verdict in {"mission_success", "mission_failed"}:
        return "respond_node"
    raise ValueError("route_after_check.invalid_result: missing semantic check verdict")


def act_node(verdict: str, task_record: TaskRecord) -> ActResult:
    return act_node_impl(
        verdict,
        task_record,
        logger=logger,
        log_task_event=log_task_event,
    )


def route_after_act(act_result: Any) -> str:
    route = ""
    verdict = ""
    if isinstance(act_result, dict):
        route = str(act_result.get("route") or "").strip()
        check_result = act_result.get("check_result")
        if isinstance(check_result, dict):
            verdict = str(check_result.get("verdict") or "").strip().lower()
    if route in {"next_step_node", "respond_node"}:
        return route
    if verdict == "plan":
        return "next_step_node"
    if verdict in {"mission_success", "mission_failed"}:
        return "respond_node"
    raise ValueError("route_after_act.invalid_result: missing semantic act route")


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


def _emit_transition_event_for_task_record(task_record: TaskRecord, phase: str) -> None:
    emit_transition_event(
        {
            "events": [],
            "correlation_id": task_record.correlation_id or None,
        },
        phase,
    )
