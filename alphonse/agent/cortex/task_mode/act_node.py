from __future__ import annotations

import json
import logging
from typing import Any, Callable


def act_node_stateful(
    state: dict[str, Any],
    *,
    task_state_with_defaults: Callable[[dict[str, Any]], dict[str, Any]],
    correlation_id: Callable[[dict[str, Any]], str | None],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    task_plan: Callable[[dict[str, Any]], dict[str, Any]],
    logger: logging.Logger,
) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "act"
    corr = correlation_id(state)
    status = str(task_state.get("status") or "running")
    current = current_step(task_state)
    current_status = str((current or {}).get("status") or "").strip().lower()

    if status == "waiting_user":
        logger.info(
            "task_mode act waiting_user correlation_id=%s step_id=%s",
            corr,
            str((current or {}).get("step_id") or ""),
        )
        return {"task_state": task_state}

    if status == "failed":
        logger.info(
            "task_mode act failed correlation_id=%s step_id=%s",
            corr,
            str((current or {}).get("step_id") or ""),
        )
        return {"task_state": task_state}

    if status == "done":
        logger.info(
            "task_mode act already_done correlation_id=%s step_id=%s",
            corr,
            str((current or {}).get("step_id") or ""),
        )
        return {"task_state": task_state}

    if current_status == "executed":
        logger.info(
            "task_mode act continue_unsatisfied correlation_id=%s step_id=%s",
            corr,
            str((current or {}).get("step_id") or ""),
        )
        task_state["status"] = "running"
        return {"task_state": task_state}

    if current_status == "failed":
        evaluation = evaluate_tool_execution(task_state=task_state, current_step=current, task_plan=task_plan)
        task_state["execution_eval"] = evaluation
        if evaluation.get("should_pause"):
            task_state["status"] = "waiting_user"
            task_state["next_user_question"] = build_execution_pause_prompt(evaluation)
            append_trace_event(
                task_state,
                {
                    "type": "status_changed",
                    "summary": str(evaluation.get("summary") or "Status changed to waiting_user after repeated failures."),
                    "correlation_id": corr,
                },
            )
            logger.info(
                "task_mode act waiting_user_execution_eval correlation_id=%s step_id=%s tool=%s reason=%s total_failures=%s same_signature=%s",
                corr,
                str((current or {}).get("step_id") or ""),
                str(evaluation.get("tool_name") or ""),
                str(evaluation.get("reason") or ""),
                int(evaluation.get("total_failures") or 0),
                int(evaluation.get("same_signature_failures") or 0),
            )
            return {"task_state": task_state}
        task_state["status"] = "running"
        logger.info(
            "task_mode act continue_after_failure correlation_id=%s step_id=%s tool=%s reason=%s total_failures=%s same_signature=%s",
            corr,
            str((current or {}).get("step_id") or ""),
            str(evaluation.get("tool_name") or ""),
            str(evaluation.get("reason") or ""),
            int(evaluation.get("total_failures") or 0),
            int(evaluation.get("same_signature_failures") or 0),
        )
        return {"task_state": task_state}

    task_state["status"] = "running"
    logger.info(
        "task_mode act continue correlation_id=%s step_id=%s",
        corr,
        str((current or {}).get("step_id") or ""),
    )
    return {"task_state": task_state}


def route_after_act_stateful(
    state: dict[str, Any],
    *,
    correlation_id: Callable[[dict[str, Any]], str | None],
    logger: logging.Logger,
) -> str:
    task_state = state.get("task_state")
    status = str((task_state or {}).get("status") or "").strip().lower() if isinstance(task_state, dict) else ""
    if status == "waiting_user":
        logger.info(
            "task_mode route_after_act correlation_id=%s route=respond_node status=%s",
            correlation_id(state),
            status,
        )
        return "respond_node"
    logger.info(
        "task_mode route_after_act correlation_id=%s route=next_step_node status=%s",
        correlation_id(state),
        status or "running",
    )
    return "next_step_node"


def evaluate_tool_execution(
    *,
    task_state: dict[str, Any],
    current_step: dict[str, Any] | None,
    task_plan: Callable[[dict[str, Any]], dict[str, Any]],
) -> dict[str, Any]:
    max_evolving_failures = 10
    immediate_repeat_limit = 2
    current_signature = _tool_signature_for_step(current_step)
    current_tool = _tool_name_for_step(current_step)
    retryable = _failure_retryable(current_step)
    plan = task_plan(task_state)
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []

    failed_steps: list[dict[str, Any]] = [
        step
        for step in steps
        if isinstance(step, dict) and str(step.get("status") or "").strip().lower() == "failed"
    ]
    total_failures = len(failed_steps)
    same_signature_failures = 0
    signatures: list[str] = []
    for step in failed_steps:
        signature = _tool_signature_for_step(step)
        if signature:
            signatures.append(signature)
        if current_signature and signature == current_signature:
            same_signature_failures += 1
    unique_signatures = len({item for item in signatures if item})
    evolving = unique_signatures > 1

    if retryable is False:
        return {
            "should_pause": True,
            "reason": "non_retryable_failure",
            "summary": f"Tool {current_tool or 'tool'} failed with a non-retryable error.",
            "tool_name": current_tool,
            "total_failures": total_failures,
            "same_signature_failures": same_signature_failures,
            "unique_signatures": unique_signatures,
            "evolving": evolving,
            "max_evolving_failures": max_evolving_failures,
        }
    if same_signature_failures >= immediate_repeat_limit:
        return {
            "should_pause": True,
            "reason": "repeated_identical_failure",
            "summary": f"Repeated identical tool attempt failed for {current_tool or 'tool'}.",
            "tool_name": current_tool,
            "total_failures": total_failures,
            "same_signature_failures": same_signature_failures,
            "unique_signatures": unique_signatures,
            "evolving": evolving,
            "max_evolving_failures": max_evolving_failures,
        }
    if total_failures >= max_evolving_failures:
        return {
            "should_pause": True,
            "reason": "failure_budget_exhausted",
            "summary": "Failure budget exhausted while trying multiple strategies.",
            "tool_name": current_tool,
            "total_failures": total_failures,
            "same_signature_failures": same_signature_failures,
            "unique_signatures": unique_signatures,
            "evolving": evolving,
            "max_evolving_failures": max_evolving_failures,
        }
    return {
        "should_pause": False,
        "reason": "continue_learning" if evolving else "single_failure",
        "summary": "Continue with next planning attempt.",
        "tool_name": current_tool,
        "total_failures": total_failures,
        "same_signature_failures": same_signature_failures,
        "unique_signatures": unique_signatures,
        "evolving": evolving,
        "max_evolving_failures": max_evolving_failures,
    }


def build_execution_pause_prompt(evaluation: dict[str, Any]) -> str:
    reason = str(evaluation.get("reason") or "")
    tool_name = str(evaluation.get("tool_name") or "tool")
    total = int(evaluation.get("total_failures") or 0)
    unique = int(evaluation.get("unique_signatures") or 0)
    if reason == "non_retryable_failure":
        return (
            f"`{tool_name}` failed with a non-retryable condition. "
            "I paused the plan to avoid repeating a request that cannot succeed right now. "
            "Please confirm what to do next."
        )
    if reason == "repeated_identical_failure":
        return (
            f"I got stuck repeating the same failed action with `{tool_name}`. "
            "I paused the plan to avoid waste. Do you want me to keep trying with your approval, "
            "or provide steering on a different approach?"
        )
    return (
        f"I tried {total} times across {unique} strategy variants and I'm still blocked. "
        "I paused the plan. Should I keep pursuing this goal, or do you want to steer me?"
    )


def _tool_name_for_step(step: dict[str, Any] | None) -> str:
    if not isinstance(step, dict):
        return ""
    proposal = step.get("proposal") if isinstance(step.get("proposal"), dict) else {}
    return str(proposal.get("tool_name") or "").strip()


def _tool_signature_for_step(step: dict[str, Any] | None) -> str:
    if not isinstance(step, dict):
        return ""
    proposal = step.get("proposal") if isinstance(step.get("proposal"), dict) else {}
    tool_name = str(proposal.get("tool_name") or "").strip()
    args = proposal.get("args") if isinstance(proposal.get("args"), dict) else {}
    try:
        args_text = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        args_text = str(args)
    return f"{tool_name}|{args_text}"


def _failure_retryable(step: dict[str, Any] | None) -> bool | None:
    if not isinstance(step, dict):
        return None
    value = step.get("failure_retryable")
    if value is None:
        return None
    return bool(value)
