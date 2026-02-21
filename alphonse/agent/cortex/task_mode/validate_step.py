from __future__ import annotations

import logging
from typing import Any, Callable


def validate_step_node_impl(
    state: dict[str, Any],
    *,
    tool_registry: Any,
    task_state_with_defaults: Callable[[dict[str, Any]], dict[str, Any]],
    correlation_id: Callable[[dict[str, Any]], str | None],
    task_plan: Callable[[dict[str, Any]], dict[str, Any]],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
    validate_proposal: Callable[..., dict[str, Any]],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    logger: logging.Logger,
    log_task_event: Callable[..., None],
) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "check"
    corr = correlation_id(state)
    _ = task_plan(task_state)
    current = current_step(task_state)
    proposal = (current or {}).get("proposal")
    validation = validate_proposal(proposal=proposal, tool_registry=tool_registry)

    if validation["ok"]:
        task_state["last_validation_error"] = None
        if current is not None:
            current["status"] = "validated"
        logger.info(
            "task_mode validate passed correlation_id=%s step_id=%s kind=%s",
            corr,
            str((current or {}).get("step_id") or ""),
            str((proposal or {}).get("kind") if isinstance(proposal, dict) else ""),
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="validate_step_node",
            event="graph.step.validated",
            step_id=str((current or {}).get("step_id") or ""),
            kind=str((proposal or {}).get("kind") if isinstance(proposal, dict) else ""),
        )
        return {"task_state": task_state}

    attempts = int(task_state.get("repair_attempts") or 0) + 1
    task_state["repair_attempts"] = attempts
    error = {
        "reason": validation.get("reason") or "validation_failed",
        "proposal": proposal,
    }
    task_state["last_validation_error"] = error
    if current is not None:
        current["status"] = "validation_failed"
    append_trace_event(
        task_state,
        {
            "type": "validation_failed",
            "summary": f"Validation failed: {error['reason']}.",
            "correlation_id": corr,
        },
    )
    logger.info(
        "task_mode validate failed correlation_id=%s step_id=%s reason=%s repair_attempts=%s",
        corr,
        str((current or {}).get("step_id") or ""),
        str(error.get("reason") or ""),
        attempts,
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="validate_step_node",
        event="graph.step.validation_failed",
        level="warning",
        step_id=str((current or {}).get("step_id") or ""),
        reason=str(error.get("reason") or ""),
        repair_attempts=attempts,
    )
    return {"task_state": task_state}
