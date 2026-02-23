from __future__ import annotations

from typing import Any, Callable

from alphonse.agent.cortex.task_mode.observability import log_task_event
from alphonse.agent.observability.log_manager import get_component_logger

logger = get_component_logger("task_mode.progress_critic")


def progress_critic_node(
    state: dict[str, Any],
    *,
    task_state_with_defaults: Callable[[dict[str, Any]], dict[str, Any]],
    correlation_id: Callable[[dict[str, Any]], str | None],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
    goal_satisfied: Callable[[dict[str, Any]], bool],
    evaluate_tool_execution: Callable[..., dict[str, Any]],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    build_progress_checkin_question: Callable[..., str],
    maybe_emit_periodic_wip_update: Callable[..., None],
    progress_check_cycle_threshold: int,
) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "critic"
    corr = correlation_id(state)
    status = str(task_state.get("status") or "").strip().lower()
    current = current_step(task_state)
    current_status = str((current or {}).get("status") or "").strip().lower()
    cycle = int(task_state.get("cycle_index") or 0)

    if goal_satisfied(task_state):
        task_state["status"] = "done"
        append_trace_event(
            task_state,
            {
                "type": "status_changed",
                "summary": "Progress critic marked task as done from outcome evidence.",
                "correlation_id": corr,
            },
        )
        logger.info(
            "task_mode progress_critic done correlation_id=%s cycle=%s",
            corr,
            cycle,
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="progress_critic_node",
            event="graph.task.completed",
        )
        return {"task_state": task_state}

    if status in {"done", "waiting_user", "failed"}:
        logger.info(
            "task_mode progress_critic skip correlation_id=%s status=%s cycle=%s",
            corr,
            status,
            cycle,
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="progress_critic_node",
            event="graph.critic.skipped",
            status=status,
            cycle=cycle,
            level="debug",
        )
        return {"task_state": task_state}

    maybe_emit_periodic_wip_update(state=state, task_state=task_state, cycle=cycle, current_step=current)

    evaluation = (
        evaluate_tool_execution(task_state=task_state, current_step=current)
        if current_status == "failed"
        else {}
    )
    task_state["execution_eval"] = evaluation if isinstance(evaluation, dict) else {}
    if current_status == "failed" and int((evaluation or {}).get("same_signature_failures") or 0) >= 3:
        task_state["status"] = "failed"
        task_state["outcome"] = {
            "kind": "task_failed",
            "summary": str(
                (evaluation or {}).get("summary")
                or "I did not make enough progress to continue safely."
            ),
        }
        append_trace_event(
            task_state,
            {
                "type": "status_changed",
                "summary": "Progress critic marked task as failed due to repeated same-signature failures.",
                "correlation_id": corr,
            },
        )
        logger.info(
            "task_mode progress_critic fail correlation_id=%s step_id=%s same_signature=%s",
            corr,
            str((current or {}).get("step_id") or ""),
            int((evaluation or {}).get("same_signature_failures") or 0),
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="progress_critic_node",
            event="graph.task.failed",
            level="warning",
            step_id=str((current or {}).get("step_id") or ""),
            same_signature_failures=int((evaluation or {}).get("same_signature_failures") or 0),
        )
        return {"task_state": task_state}

    if cycle < progress_check_cycle_threshold:
        logger.info(
            "task_mode progress_critic continue correlation_id=%s cycle=%s threshold=%s",
            corr,
            cycle,
            progress_check_cycle_threshold,
        )
        return {"task_state": task_state}

    question = build_progress_checkin_question(
        state=state,
        task_state=task_state,
        evaluation=evaluation if isinstance(evaluation, dict) else {},
    )
    task_state["status"] = "waiting_user"
    task_state["next_user_question"] = question
    append_trace_event(
        task_state,
        {
            "type": "status_changed",
            "summary": "Progress critic requested user guidance after extended work-in-progress.",
            "correlation_id": corr,
        },
    )
    logger.info(
        "task_mode progress_critic ask_user correlation_id=%s cycle=%s step_id=%s",
        corr,
        cycle,
        str((current or {}).get("step_id") or ""),
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="progress_critic_node",
        event="graph.critic.ask_user",
        step_id=str((current or {}).get("step_id") or ""),
    )
    return {"task_state": task_state}


def route_after_progress_critic(
    state: dict[str, Any],
    *,
    correlation_id: Callable[[dict[str, Any]], str | None],
) -> str:
    task_state = state.get("task_state")
    status = str((task_state or {}).get("status") or "").strip().lower() if isinstance(task_state, dict) else ""
    if status in {"waiting_user", "failed", "done"}:
        logger.info(
            "task_mode route_after_progress_critic correlation_id=%s route=respond_node status=%s",
            correlation_id(state),
            status,
        )
        return "respond_node"
    logger.info(
        "task_mode route_after_progress_critic correlation_id=%s route=act_node status=%s",
        correlation_id(state),
        status or "running",
    )
    return "act_node"
