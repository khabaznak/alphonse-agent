from __future__ import annotations

from typing import Any, Literal, TypedDict

from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.cortex.transitions import emit_presence_transition_event

_VERDICT_KINDS = {"plan", "mission_success", "mission_failed"}


class ActResult(TypedDict):
    route: Literal["next_step_node", "respond_node"]
    task_record: TaskRecord


def act_node_impl(
    verdict: str,
    task_record: TaskRecord,
    *,
    logger: Any,
    log_task_event: Any,
) -> ActResult:
    normalized_verdict = _validate_verdict(verdict)
    if normalized_verdict is None:
        raise ValueError(
            "act_node.invalid_verdict: verdict must be one of plan|mission_success|mission_failed"
        )

    route = "next_step_node" if normalized_verdict == "plan" else "respond_node"
    _emit_terminal_transition_if_needed(task_record=task_record, verdict=normalized_verdict)
    _log_act_result(
        verdict=normalized_verdict,
        route=route,
        task_record=task_record,
        logger=logger,
        log_task_event=log_task_event,
    )
    return {
        "route": route,
        "task_record": task_record,
    }


def _validate_verdict(verdict: str | None) -> str | None:
    rendered = str(verdict or "").strip().lower()
    return rendered if rendered in _VERDICT_KINDS else None


def _emit_terminal_transition_if_needed(*, task_record: TaskRecord, verdict: str) -> None:
    if verdict == "plan":
        return
    phase = "done" if verdict == "mission_success" else "failed"
    emit_presence_transition_event(
        {
            "events": [],
            "correlation_id": task_record.correlation_id or None,
        },
        event_family="presence.task_terminal",
        phase=phase,
        detail={
            "verdict": verdict,
            "task_id": str(task_record.task_id or ""),
        },
    )


def _log_act_result(
    *,
    verdict: str,
    route: str,
    task_record: TaskRecord,
    logger: Any,
    log_task_event: Any,
) -> None:
    logger.info(
        "task_mode act verdict=%s route=%s task_id=%s",
        verdict,
        route,
        str(task_record.task_id or ""),
    )
    log_task_event(
        logger=logger,
        state={
            "correlation_id": task_record.correlation_id or None,
            "channel_type": None,
            "actor_person_id": task_record.user_id,
        },
        node="act_node",
        event="graph.act.routed",
        task_record=task_record,
        cycle_index=0,
        verdict=verdict,
        route=route,
    )
