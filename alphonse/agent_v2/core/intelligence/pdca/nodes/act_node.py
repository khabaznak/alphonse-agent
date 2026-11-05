"""Act node for the v2 PDCA graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.intelligence.task_state import TaskState

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext

_PLAN_ROUTE = "plan"
_END_ROUTE = "end"
_TEMPORARY_MAX_COMPLETED_CYCLES = 10
_PLAN_CALL_EXCEPTION_FAILURE_THRESHOLD = 3


def act_node(task: TaskState, context: CoreLoopContext | None = None) -> TaskState:
    """Act on the check verdict without re-checking or executing work."""
    if context is not None:
        context.emit_activity(
            phase=ImprovementPhase.ACT,
            label="deciding",
            message="Deciding the next CAPD route.",
        )
    _observe_completed_do_cycle(task)
    verdict = str(task.check_verdict or "").strip().lower()
    if verdict == "wip":
        return _act_on_wip(task)

    if verdict == "new":
        task.metadata["act_route"] = _PLAN_ROUTE
        task.append_update("Act routed the new mission to Plan to establish acceptance criteria.")
        return task

    if verdict == "steer":
        task.metadata["act_route"] = _PLAN_ROUTE
        task.append_update("Act routed steering to Plan for acceptance-contract and strategy revision.")
        return task

    return _route_checked_task(task)


def _act_on_wip(task: TaskState) -> TaskState:
    if task.metadata.get("pending_silent_bash_confirmation"):
        task.metadata["act_route"] = _PLAN_ROUTE
        task.append_update("Act requires a native.respond confirmation before completing the task.")
        return task

    return _route_checked_task(task)


def _route_checked_task(task: TaskState) -> TaskState:
    if task.metadata.get("cancel_requested") is True or task.metadata.get("kill_switch_cancelled") is True:
        task.status = "cancelled"
        task.metadata["act_route"] = _END_ROUTE
        return task
    directive = task.metadata.get("act_directive")
    if isinstance(directive, dict) and directive.get("closure_only"):
        return _mark_mission_failed(task, str(directive.get("reason") or "The mission could not be verified."))
    verdict = str(task.check_verdict or "").strip().lower()
    verified = verdict in {"wip", "success", "mission_success"} and task.acceptance_criteria_all_complete()
    if verified and task.has_prepared_user_response():
        return _mark_mission_success(task, task.check_reason or "Check verified the acceptance criteria.")
    if _temporary_cycle_limit_reached(task):
        return _mark_mission_failed(task, _mission_failure_reason(task) or "The mission reached the execution cycle limit.")
    reason = _mission_failure_reason(task)
    if verdict in {"failure", "mission_failed"} or reason:
        reason = reason or task.check_reason or "The mission could not be verified."
        task.metadata["act_directive"] = {
            "action": "fail_explain", "closure_only": True, "response_required": True,
            "terminal_outcome": "failed", "reason": reason,
        }
        task.metadata["pending_user_response"] = True
    elif verified:
        task.metadata["pending_user_response"] = True
        task.metadata["act_directive"] = {"action": "complete", "response_required": True}
    else:
        task.metadata["act_directive"] = {"action": "replan", "rationale": task.check_reason}
    task.metadata["act_route"] = _PLAN_ROUTE
    return task


def _mark_mission_success(task: TaskState, reason: str) -> TaskState:
    task.set_check_result(
        verdict="mission_success",
        reason=reason,
        confidence=1.0,
        evidence_refs=list(task.check_evidence_refs or []),
        new_message_count=task.check_new_message_count,
    )
    task.status = "completed"
    task.outcome = {"status": "success", "reason": reason}
    task.metadata["act_route"] = _END_ROUTE
    task.metadata["act_terminal_decision"] = "mission_success"
    task.append_update("Act marked the task as mission success.")
    return task


def _mark_mission_failed(task: TaskState, reason: str) -> TaskState:
    task.set_check_result(
        verdict="mission_failed",
        reason=reason,
        confidence=1.0,
        evidence_refs=list(task.check_evidence_refs or []),
        new_message_count=task.check_new_message_count,
    )
    task.status = "failed"
    task.outcome = {"status": "failed", "reason": reason}
    task.metadata["act_route"] = _END_ROUTE
    task.metadata["act_terminal_decision"] = "mission_failed"
    task.append_update("Act marked the task as mission failed.")
    return task


def _mission_failure_reason(task: TaskState) -> str:
    if task.metadata.get("cancel_requested") is True:
        return str(task.metadata.get("failure_reason") or "Task was cancelled.")
    if task.metadata.get("mission_failed") is True:
        return str(task.metadata.get("failure_reason") or "Mission failure was explicitly signaled.")
    failure_reason = str(task.metadata.get("failure_reason") or "").strip()
    if failure_reason:
        return failure_reason
    if task.count_plan_call_exceptions() >= _PLAN_CALL_EXCEPTION_FAILURE_THRESHOLD:
        return f"Planned tool calls reached {_PLAN_CALL_EXCEPTION_FAILURE_THRESHOLD} exceptions."
    return ""


def _observe_completed_do_cycle(task: TaskState) -> None:
    if task.metadata.get("do_executed_since_last_act") is not True:
        return
    task.pdca_cycle_count = max(0, int(task.pdca_cycle_count or 0)) + 1
    task.metadata["do_executed_since_last_act"] = False
    task.metadata["completed_capd_cycle_count"] = task.pdca_cycle_count
    task.append_update(f"Act observed completed CAPD execution cycle {task.pdca_cycle_count}.")


def _temporary_cycle_limit_reached(task: TaskState) -> bool:
    completed_cycle_limit = max(0, int(task.pdca_cycle_count or 0)) >= _TEMPORARY_MAX_COMPLETED_CYCLES
    stubbed_planning_pass = task.metadata.get("tool_call_planning_llm_stubbed") is True
    return completed_cycle_limit or stubbed_planning_pass
