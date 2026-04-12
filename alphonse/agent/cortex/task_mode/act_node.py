from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from alphonse.agent.cognition.providers.factory import build_text_completion_provider
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.cortex.transitions import emit_presence_transition_event

_VERDICT_PLAN = "plan"
_VERDICT_MISSION_SUCCESS = "mission_success"
_VERDICT_MISSION_FAILED = "mission_failed"
_ACT_ROUTE_NEXT_STEP = "next_step_node"
_ACT_ROUTE_END = "end"
_FAILURE_SUMMARY_MAX_CHARS = 256


class ActResult(TypedDict):
    route: Literal["next_step_node", "end"]
    task_record: TaskRecord
    response_text: str | None


def act_node_impl(
    task_record: TaskRecord,
    *,
    logger: Any,
    log_task_event: Any,
) -> ActResult:
    normalized_verdict = _verdict_from_task_record(task_record)
    if normalized_verdict is None:
        raise ValueError(
            "act_node.invalid_task_record_status: status must imply one of plan|mission_success|mission_failed"
        )

    route = _ACT_ROUTE_NEXT_STEP if normalized_verdict == _VERDICT_PLAN else _ACT_ROUTE_END
    response_text = None
    if normalized_verdict == _VERDICT_MISSION_SUCCESS:
        response_text = _success_response_for_user(task_record)
    elif normalized_verdict == _VERDICT_MISSION_FAILED:
        response_text = _summarize_failure_for_user(task_record=task_record, logger=logger)
        if response_text:
            outcome = dict(task_record.outcome or {})
            outcome["final_text"] = response_text
            task_record.outcome = outcome
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
        "response_text": response_text,
    }


def _verdict_from_task_record(task_record: TaskRecord) -> str | None:
    status = str(task_record.status or "").strip().lower()
    if status == "running":
        return _VERDICT_PLAN
    if status == "done":
        return _VERDICT_MISSION_SUCCESS
    if status == "failed":
        return _VERDICT_MISSION_FAILED
    return None


def _emit_terminal_transition_if_needed(*, task_record: TaskRecord, verdict: str) -> None:
    if verdict == _VERDICT_PLAN:
        return
    phase = "done" if verdict == _VERDICT_MISSION_SUCCESS else "failed"
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
            "user_id": task_record.user_id,
        },
        node="act_node",
        event="graph.act.routed",
        task_record=task_record,
        cycle_index=0,
        verdict=verdict,
        route=route,
    )


def _summarize_failure_for_user(*, task_record: TaskRecord, logger: Any) -> str:
    fallback = _deterministic_failure_summary(task_record)
    try:
        llm_client = build_text_completion_provider()
        raw = llm_client.complete(
            "You write concise, user-facing task failure summaries.",
            _failure_summary_prompt(task_record),
        )
    except Exception as exc:
        logger.warning(
            "task_mode act failure summary llm unavailable task_id=%s error=%s",
            str(task_record.task_id or ""),
            str(exc)[:160],
        )
        return fallback
    rendered = str(raw or "").strip()
    if rendered.startswith("```"):
        rendered = rendered.strip("`").strip()
        if rendered.lower().startswith("text"):
            rendered = rendered[4:].strip()
    return _clip_summary(rendered or fallback)


def _success_response_for_user(task_record: TaskRecord) -> str | None:
    if _has_public_message_send_success(task_record):
        return None
    outcome = task_record.outcome if isinstance(task_record.outcome, dict) else {}
    for key in ("final_text", "summary"):
        rendered = str(outcome.get(key) or "").strip()
        if rendered:
            return rendered
    return None


def _failure_summary_prompt(task_record: TaskRecord) -> str:
    payload = {
        "task_id": task_record.task_id,
        "goal": task_record.goal,
        "outcome": task_record.outcome,
        "facts": task_record.get_facts_md(),
        "tool_call_history": task_record.get_tool_call_history_md(),
    }
    return (
        "Summarize why this task failed in 256 characters or fewer. "
        "Do not include stack traces. Speak directly to the user.\n\n"
        "## TaskRecord\n```json\n"
        f"{json.dumps(payload, ensure_ascii=False, default=str, indent=2)}\n"
        "```"
    )


def _deterministic_failure_summary(task_record: TaskRecord) -> str:
    outcome = task_record.outcome if isinstance(task_record.outcome, dict) else {}
    for key in ("final_text", "summary", "failure_class", "kind"):
        value = str(outcome.get(key) or "").strip()
        if value:
            return _clip_summary(value)
    history = str(task_record.get_tool_call_history_md() or "").strip()
    if history and history != "- (none)":
        return _clip_summary(history.splitlines()[-1].strip().removeprefix("- ").strip())
    return "I could not complete the task. Please try again or provide more detail."


def _has_public_message_send_success(task_record: TaskRecord) -> bool:
    history = [
        line.strip().removeprefix("- ").strip()
        for line in str(task_record.get_tool_call_history_md() or "").splitlines()
        if line.strip()
    ]
    for entry in history:
        if "communication.send_message" in entry and "exception=null" in entry:
            return True
    return False


def _clip_summary(value: str) -> str:
    rendered = " ".join(str(value or "").split())
    if len(rendered) <= _FAILURE_SUMMARY_MAX_CHARS:
        return rendered
    return rendered[: _FAILURE_SUMMARY_MAX_CHARS - 3].rstrip() + "..."
