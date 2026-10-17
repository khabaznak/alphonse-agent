"""Act node for the v2 PDCA graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.acceptance_planning import _render_acceptance_criteria_prompt, _render_acceptance_criteria_amendment_prompt
from alphonse.agent_v2.system_one import SystemOneUnavailableError

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
        return _act_on_wip(task, context)

    if verdict == "new":
        task.metadata["act_route"] = _PLAN_ROUTE
        task.append_update("Act routed the new mission to Plan to establish acceptance criteria.")
        return task

    if verdict == "steer":
        task.metadata["act_route"] = _PLAN_ROUTE
        task.append_update("Act routed steering to Plan for acceptance-contract and strategy revision.")
        return task

    return _recommend_act(task, context)


def _act_on_wip(task: TaskState, context: CoreLoopContext | None) -> TaskState:
    if task.metadata.get("pending_silent_bash_confirmation"):
        task.metadata["act_route"] = _PLAN_ROUTE
        task.append_update("Act requires a native.respond confirmation before completing the task.")
        return task

    return _recommend_act(task, context)


def _recommend_act(task: TaskState, context: CoreLoopContext | None) -> TaskState:
    if task.metadata.get("cancel_requested") is True or task.metadata.get("kill_switch_cancelled") is True:
        task.status = "cancelled"
        task.metadata["act_route"] = _END_ROUTE
        return task
    directive = task.metadata.get("act_directive")
    if isinstance(directive, dict) and directive.get("closure_only") and task.has_prepared_user_response():
        return _mark_mission_failed(task, str(directive.get("reason") or "Mission failed; final explanation delivered."))
    if context is None or context.system_one is None or not callable(getattr(context.system_one, "recommend_act", None)):
        raise SystemOneUnavailableError("act_recommendation_unavailable")
    state = {
        "goal": task.goal,
        "check_verdict": task.check_verdict,
        "check_reason": task.check_reason,
        "acceptance_contract": task.ensure_acceptance_contract(),
        "task_state": task.to_dict(),
        "complete_tool_execution_log": task.plan_json,
        "user_constraints": task.metadata.get("user_constraints", {}),
        "completed_cycles": task.pdca_cycle_count,
        "cycle_limit": _TEMPORARY_MAX_COMPLETED_CYCLES,
        "plan_call_exceptions": task.count_plan_call_exceptions(),
        "plan_call_exception_limit": _PLAN_CALL_EXCEPTION_FAILURE_THRESHOLD,
        "failure_reason": _mission_failure_reason(task),
    }
    try:
        recommendation = context.system_one.recommend_act(state=state)
    except Exception as exc:
        raise SystemOneUnavailableError(f"act_recommendation:{type(exc).__name__}") from exc
    task.metadata["act_recommendation"] = recommendation.to_metadata()
    action = recommendation.action
    yes_threshold = float(getattr(recommendation, "yes_threshold", 0.8))
    if _temporary_cycle_limit_reached(task) and action in {"continue", "replan"}:
        action = "fail_explain" if recommendation.answers.get("closure_explanation_is_warranted", 0.0) >= yes_threshold else "fail"
    if action == "complete" and not task.acceptance_criteria_all_complete():
        action = "continue" if recommendation.answers.get("continuation_is_worthwhile", 0.0) >= yes_threshold else "replan"
    if action == "complete" and not task.has_prepared_user_response():
        task.metadata["pending_user_response"] = True
        task.metadata["act_directive"] = {"action": "complete", "response_required": True}
        action = "continue"
    if action == "ask_user" and recommendation.answers.get("user_input_can_unblock", 0.0) < yes_threshold:
        action = "replan"
    if action == "fail_explain" and recommendation.answers.get("closure_explanation_is_warranted", 0.0) < yes_threshold:
        action = "fail"
    if not recommendation.confident and action not in {"fail", "fail_explain"}:
        action = "replan"
    if action == "fail_explain":
        task.metadata["act_directive"] = {
            "action": "fail_explain", "closure_only": True, "response_required": True,
            "terminal_outcome": "failed", "reason": recommendation.rationale,
        }
        task.metadata["pending_user_response"] = True
        action = "continue"
    if action in {"continue", "replan", "ask_user"}:
        task.metadata.setdefault("act_directive", {"action": action, "rationale": recommendation.rationale})
        task.metadata["act_route"] = _PLAN_ROUTE
        task.append_update(f"Act routed to Plan on Jev recommendation: {action}.")
        return task
    if action == "complete":
        return _mark_mission_success(task, recommendation.rationale)
    if action == "fail":
        return _mark_mission_failed(task, recommendation.rationale)
    raise SystemOneUnavailableError("act_recommendation_action_invalid")


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


def _markdown_has_acceptance_criteria(value: str) -> bool:
    rendered = str(value or "").strip()
    return bool(rendered and rendered != "- (none)")
