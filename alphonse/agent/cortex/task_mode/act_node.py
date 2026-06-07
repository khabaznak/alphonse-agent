from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from alphonse.agent.cognition.providers.factory import build_text_completion_provider
from alphonse.agent.cognition.prompt_templates_runtime import ACT_CRITERIA_SYSTEM_PROMPT_TEMPLATE
from alphonse.agent.cognition.prompt_templates_runtime import ACT_CRITERIA_USER_PROMPT_TEMPLATE
from alphonse.agent.cognition.prompt_templates_runtime import render_prompt_template
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.cortex.transitions import emit_presence_transition_event

_VERDICT_NEW = "new"
_VERDICT_STEER = "steer"
_VERDICT_WIP = "wip"
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
    verdict = str(task_record.check_verdict or "").strip().lower()
    if verdict not in {
        _VERDICT_NEW,
        _VERDICT_STEER,
        _VERDICT_WIP,
        _VERDICT_MISSION_SUCCESS,
        _VERDICT_MISSION_FAILED,
    }:
        raise ValueError(f"act_node.invalid_check_verdict: {verdict or '(missing)'}")

    response_text = None
    if verdict == _VERDICT_NEW:
        _replace_acceptance_criteria(task_record, _generate_acceptance_criteria(task_record=task_record, mode="new"))
        task_record.status = "running"
        task_record.outcome = None
        route = _ACT_ROUTE_NEXT_STEP
    elif verdict == _VERDICT_STEER:
        _replace_acceptance_criteria(task_record, _generate_acceptance_criteria(task_record=task_record, mode="steer"))
        task_record.status = "running"
        task_record.outcome = None
        route = _ACT_ROUTE_NEXT_STEP
    elif verdict == _VERDICT_WIP:
        # TODO: enforce future WIP policies such as max attempts, max runtime, and token budgets here.
        task_record.pdca_cycle_count += 1
        task_record.status = "running"
        task_record.outcome = None
        route = _ACT_ROUTE_NEXT_STEP
    elif verdict == _VERDICT_MISSION_SUCCESS:
        _apply_mission_success(task_record)
        response_text = _success_response_for_user(task_record)
        route = _ACT_ROUTE_END
    else:
        _apply_mission_failed(task_record)
        response_text = _summarize_failure_for_user(task_record=task_record, logger=logger)
        if response_text:
            outcome = dict(task_record.outcome or {})
            outcome["final_text"] = response_text
            task_record.outcome = outcome
        route = _ACT_ROUTE_END

    _emit_terminal_transition_if_needed(task_record=task_record, verdict=verdict)
    _log_act_result(
        verdict=verdict,
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


def _generate_acceptance_criteria(*, task_record: TaskRecord, mode: Literal["new", "steer"]) -> list[str]:
    llm_client = build_text_completion_provider()
    prompt = render_prompt_template(
        ACT_CRITERIA_USER_PROMPT_TEMPLATE,
        {
            "CHECK_VERDICT": mode,
            "CHECK_REASON": task_record.check_reason,
            "GOAL": task_record.goal,
            "RECENT_CONVERSATION": task_record.recent_conversation_md,
            "ACCEPTANCE_CRITERIA": task_record.get_acceptance_criteria_md(),
            "FACTS_SECTION": task_record.get_facts_md(),
            "TOOL_CALL_HISTORY_SECTION": task_record.get_tool_call_history_md(),
        },
    )
    raw = llm_client.complete(
        system_prompt=ACT_CRITERIA_SYSTEM_PROMPT_TEMPLATE,
        user_prompt=prompt,
    )
    criteria = _parse_criteria_response(raw)
    if not criteria:
        raise ValueError("act_criteria_empty")
    return criteria


def _parse_criteria_response(raw: Any) -> list[str]:
    parsed = _parse_json_object(raw)
    if not isinstance(parsed, dict):
        raise ValueError("act_criteria_invalid_json")
    raw_criteria = parsed.get("criteria")
    if not isinstance(raw_criteria, list):
        raise ValueError("act_criteria_missing_criteria")
    return [str(item).strip() for item in raw_criteria if str(item).strip()][:24]


def _replace_acceptance_criteria(task_record: TaskRecord, criteria: list[str]) -> None:
    task_record.acceptance_criteria_md = "- (none)"
    for criterion in criteria:
        task_record.append_acceptance_criterion(criterion)


def _apply_mission_success(task_record: TaskRecord) -> None:
    reason = str(task_record.check_reason or "").strip() or "Mission completed successfully."
    task_record.status = "done"
    task_record.outcome = {
        "kind": "task_completed",
        "summary": reason,
        "final_text": reason,
    }


def _apply_mission_failed(task_record: TaskRecord) -> None:
    reason = str(task_record.check_reason or "").strip() or "Mission failed."
    failure_class = "mission_failed"
    task_record.status = "failed"
    task_record.outcome = {
        "kind": "task_failed",
        "summary": reason,
        "final_text": reason,
        "failure_class": failure_class,
    }


def route_after_act(act_result: Any) -> str:
    route = ""
    if isinstance(act_result, dict):
        route = str(act_result.get("route") or "").strip()
    if route in {_ACT_ROUTE_NEXT_STEP, _ACT_ROUTE_END}:
        return route
    raise ValueError("route_after_act.invalid_result: missing semantic act route")


def _emit_terminal_transition_if_needed(*, task_record: TaskRecord, verdict: str) -> None:
    if verdict not in {_VERDICT_MISSION_SUCCESS, _VERDICT_MISSION_FAILED}:
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
        cycle_index=task_record.pdca_cycle_count,
        verdict=verdict,
        route=route,
    )


def _summarize_failure_for_user(*, task_record: TaskRecord, logger: Any) -> str:
    existing = _existing_failure_response_text(task_record)
    if existing:
        return existing
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


def _existing_failure_response_text(task_record: TaskRecord) -> str | None:
    outcome = task_record.outcome if isinstance(task_record.outcome, dict) else {}
    for key in ("final_text",):
        rendered = str(outcome.get(key) or "").strip()
        if rendered:
            return _clip_summary(rendered)
    return None


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


def _parse_json_object(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    value = str(raw or "").strip()
    if not value:
        return None
    if value.startswith("```"):
        value = value.strip("`").strip()
        if value.lower().startswith("json"):
            value = value[4:].strip()
    try:
        parsed = json.loads(value)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None
