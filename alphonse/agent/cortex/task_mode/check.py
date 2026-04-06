from __future__ import annotations

import json
from typing import Any

from alphonse.agent.cognition.prompt_templates_runtime import CHECK_JUDGE_SYSTEM_PROMPT_TEMPLATE
from alphonse.agent.cognition.prompt_templates_runtime import CHECK_JUDGE_USER_PROMPT_TEMPLATE
from alphonse.agent.cognition.prompt_templates_runtime import render_prompt_template
from alphonse.agent.cognition.utterance_policy import render_utterance_policy_block
from alphonse.agent.cortex.llm_output.json_parse import parse_json_object
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.cortex.task_mode.task_state_helpers import append_trace_event
from alphonse.agent.cortex.task_mode.task_state_helpers import correlation_id
from alphonse.agent.cortex.task_mode.task_state_helpers import normalize_acceptance_criteria_records
from alphonse.agent.cortex.task_mode.task_state_helpers import task_metrics
from alphonse.agent.cortex.task_mode.task_state_helpers import task_state_with_defaults
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.nervous_system.pdca_queue_store import append_pdca_event
from alphonse.agent.services.pdca_task_inputs import consume_task_inputs_for_check
from alphonse.agent.session.day_state import render_recent_conversation_block

_JUDGE_INVALID_BUDGET_DEFAULT = 2

_CASE_TYPES = {"new_request", "execution_review", "task_resumption"}
_VERDICT_KINDS = {"conversation", "plan", "mission_success", "mission_failed"}
_PDCA_PHASE_CHECK = "check"

class JudgePromptTemplateError(RuntimeError):
    def __init__(self, *, template_id: str, message: str) -> None:
        super().__init__(message)
        self.template_id = template_id


def _prepare_check_context(state: dict[str, Any]) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    metrics = task_metrics(task_state)
    task_state["pdca_phase"] = _PDCA_PHASE_CHECK
    task_state["acceptance_criteria"] = normalize_acceptance_criteria_records(task_state.get("acceptance_criteria"))
    metrics["steering_consumed_in_check"] = False
    return {
        "task_state": task_state,
        "correlation_id": correlation_id(state),
        "cycle": int(task_state.get("cycle_index") or 0),
    }


def check_node_impl(
    task_record: TaskRecord,
    *,
    llm_client: Any,
    logger: Any,
    log_task_event: Any,
) -> dict[str, Any]:
    case_type = _validate_case_type(_determine_case_type(task_record))
    if case_type is None:
        return _build_invalid_case_type_result(task_record=task_record)

    consumed_inputs = _consume_task_inputs_for_task_record(task_record)
    if consumed_inputs:
        _append_consumed_inputs_to_task_record(task_record, consumed_inputs=consumed_inputs)
        task_record.replan()

    try:
        judge_prompt = _get_judge_prompt_from_task_record(
            task_record=task_record,
            case_type=case_type,
        )
    except JudgePromptTemplateError as exc:
        return _build_judge_prompt_failure_result(
            task_record=task_record,
            logger=logger,
            log_task_event=log_task_event,
            error=str(exc),
        )
    judge_result = _conduct_trial(
        llm_client=llm_client,
        judge_prompt=judge_prompt,
        case_type=case_type,
        task_record=task_record,
    )
    verdict = _extract_verdict_kind(judge_result)
    updated_task_record = _apply_check_updates(
        task_record=task_record,
        criteria_updates=judge_result.get("criteria_updates"),
        case_type=case_type,
    )
    updated_task_record = _apply_terminal_result(task_record=updated_task_record, verdict=verdict, judge_result=judge_result)
    result = _build_check_result(
        task_record=updated_task_record,
        verdict=verdict,
        judge_result=judge_result,
        case_type=case_type,
        consumed_inputs=consumed_inputs,
    )
    _log_check_result(
        task_record=updated_task_record,
        verdict=verdict,
        judge_result=judge_result,
        logger=logger,
        log_task_event=log_task_event,
    )
    return result


def _determine_case_type(task_record: TaskRecord) -> str:
    if task_record.get_acceptance_criteria_md() == "- (none)" and task_record.get_tool_call_history_md() == "- (none)":
        return "new_request"
    if task_record.get_tool_call_history_md() != "- (none)":
        return "execution_review"
    return "task_resumption"


def _build_invalid_case_type_result(*, task_record: TaskRecord) -> dict[str, Any]:
    judge_result = _mission_failed_judge_result(
        case_type="execution_review",
        reason="case_type is required and must be one of: new_request|execution_review|task_resumption.",
        failure_class="invalid_case_type",
    )
    verdict = _extract_verdict_kind(judge_result)
    failed_record = _apply_terminal_result(task_record=task_record, verdict=verdict, judge_result=judge_result)
    return _build_check_result(task_record=failed_record, verdict=verdict, judge_result=judge_result, case_type="execution_review")


def _get_judge_prompt_from_task_record(
    *,
    task_record: TaskRecord,
    case_type: str,
) -> str:
    policy = render_utterance_policy_block(
        locale=None,
        tone=None,
        address_style=None,
        channel_type=None,
    )
    try:
        return render_prompt_template(
            CHECK_JUDGE_USER_PROMPT_TEMPLATE,
            {
                "POLICY_BLOCK": policy,
                "CASE_TYPE": case_type,
                "RECENT_CONVERSATION": task_record.recent_conversation_md,
                "GOAL": task_record.goal,
                "ACCEPTANCE_CRITERIA_BASELINE": task_record.get_acceptance_criteria_md(),
                "FACTS_SECTION": task_record.get_facts_md(),
                "PLAN_SECTION": task_record.get_plan_md(),
                "MEMORY_FACTS_SECTION": task_record.get_memory_facts_md(),
                "TOOL_CALL_HISTORY_SECTION": task_record.get_tool_call_history_md(),
            },
        )
    except Exception as exc:
        raise JudgePromptTemplateError(
            template_id="pdca.check.judge.user.j2",
            message=f"failed to render check judge template: {exc}",
        ) from exc


def _conduct_trial(
    *,
    llm_client: Any,
    judge_prompt: str,
    case_type: str,
    task_record: TaskRecord,
) -> dict[str, Any]:
    if llm_client is None:
        return _fallback_trial_judge_result(case_type=case_type, task_record=task_record)
    try:
        raw = _call_judge_llm(llm_client=llm_client, judge_prompt=judge_prompt)
    except JudgePromptTemplateError:
        raise
    parsed = _parse_judge_verdict(raw=raw, case_type=case_type)
    if isinstance(parsed, dict):
        return parsed
    return {
        "kind": "plan",
        "case_type": case_type,
        "reason": "Judge output invalid; continue planning.",
        "confidence": 0.0,
        "criteria_updates": [],
        "evidence_refs": [],
        "failure_class": None,
    }


def _extract_verdict_kind(judge_result: dict[str, Any]) -> str:
    return str(judge_result.get("kind") or "").strip().lower() or "plan"


def _apply_check_updates(
    *,
    task_record: TaskRecord,
    criteria_updates: Any,
    case_type: str,
) -> TaskRecord:
    updated = task_record
    if case_type == "new_request":
        _apply_new_request_updates(updated, criteria_updates)
    else:
        _apply_non_entry_updates(updated, criteria_updates)
    return updated


def _apply_new_request_updates(task_record: TaskRecord, criteria_updates: Any) -> None:
    if not str(task_record.goal or "").strip():
        task_record.goal = "the user request"
    _append_acceptance_criteria_updates(task_record, criteria_updates)


def _apply_non_entry_updates(task_record: TaskRecord, criteria_updates: Any) -> None:
    _append_acceptance_criteria_updates(task_record, criteria_updates)


def _append_acceptance_criteria_updates(task_record: TaskRecord, updates: Any) -> None:
    for update in updates if isinstance(updates, list) else []:
        if not isinstance(update, dict):
            continue
        op = str(update.get("op") or "").strip().lower()
        if op == "append":
            text = str(update.get("text") or "").strip()
            if text:
                task_record.append_acceptance_criterion(text)
        elif op == "mark_satisfied":
            criterion_id = str(update.get("criterion_id") or "").strip()
            if criterion_id:
                task_record.append_acceptance_criterion(f"{criterion_id} marked satisfied")


def _apply_terminal_result(*, task_record: TaskRecord, verdict: str, judge_result: dict[str, Any]) -> TaskRecord:
    reason = str(judge_result.get("reason") or "").strip()
    if verdict == "mission_success":
        task_record.status = "done"
        task_record.outcome = {
            "kind": "task_completed",
            "summary": reason or "Mission completed successfully.",
            "final_text": reason or "Mission completed successfully.",
        }
    elif verdict == "mission_failed":
        task_record.status = "failed"
        task_record.outcome = {
            "kind": "task_failed",
            "summary": reason or "Mission failed.",
            "failure_class": str(judge_result.get("failure_class") or "mission_failed").strip() or "mission_failed",
        }
    else:
        task_record.status = "running"
        task_record.outcome = None
    return task_record


def _build_check_result(
    *,
    task_record: TaskRecord,
    verdict: str,
    judge_result: dict[str, Any],
    case_type: str,
    consumed_inputs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "task_record": task_record,
        "verdict": verdict,
        "judge_result": dict(judge_result),
        "case_type": case_type,
        "status": task_record.status,
        "outcome": task_record.outcome,
        "reason": str(judge_result.get("reason") or "").strip(),
        "confidence": float(judge_result.get("confidence") or 0.0),
        "consumed_inputs": list(consumed_inputs or []),
    }


def _build_judge_prompt_failure_result(
    *,
    task_record: TaskRecord,
    logger: Any,
    log_task_event: Any,
    error: str,
) -> dict[str, Any]:
    judge_result = _mission_failed_judge_result(
        case_type="execution_review",
        reason="The PDCA check templating system failed; contact the admin.",
        failure_class="judge_prompt_template_failed",
    )
    verdict = _extract_verdict_kind(judge_result)
    failed_record = _apply_terminal_result(task_record=task_record, verdict=verdict, judge_result=judge_result)
    log_task_event(
        logger=logger,
        state={"correlation_id": None, "channel_type": None, "actor_person_id": task_record.user_id},
        task_state={"cycle_index": 0, "status": failed_record.status},
        node="check_node",
        event="judge.prompt_template.failed",
        template_id="pdca.check.judge.user.j2",
        error=error,
    )
    return _build_check_result(
        task_record=failed_record,
        verdict=verdict,
        judge_result=judge_result,
        case_type="execution_review",
    )


def _log_check_result(
    *,
    task_record: TaskRecord,
    verdict: str,
    judge_result: dict[str, Any],
    logger: Any,
    log_task_event: Any,
) -> None:
    logger.info(
        "task_mode check verdict=%s status=%s",
        verdict,
        task_record.status,
    )
    log_task_event(
        logger=logger,
        state={"correlation_id": None, "channel_type": None, "actor_person_id": task_record.user_id},
        task_state={"cycle_index": 0, "status": task_record.status},
        node="check_node",
        event="graph.check.judge_verdict",
        verdict_kind=verdict,
        case_type=str(judge_result.get("case_type") or ""),
        confidence=float(judge_result.get("confidence") or 0.0),
        route="next_step_node" if verdict == "plan" else "respond_node",
    )


def run_check_from_state(
    state: dict[str, Any],
    *,
    logger: Any,
    log_task_event: Any,
) -> dict[str, Any]:
    context = _prepare_check_context(state)
    task_state = context["task_state"]
    corr = context["correlation_id"]
    cycle = context["cycle"]

    case_type = _validate_case_type(select_case_deterministically(task_state))
    if case_type is None:
        judge_result = _mission_failed_judge_result(
            case_type="execution_review",
            reason="check_provenance is required and must be one of: entry|do|slice_resume.",
            failure_class="invalid_provenance",
        )
        verdict = _extract_verdict_kind(judge_result)
        task_record = _build_task_record_from_state(state=state, task_state=task_state)
        return _finalize_check_state_result(
            state=state,
            task_state=task_state,
            task_record=task_record,
            verdict=verdict,
            judge_result=judge_result,
            logger=logger,
            log_task_event=log_task_event,
            correlation_id=corr,
            cycle=cycle,
        )

    task_record = _build_task_record_from_state(state=state, task_state=task_state)
    result = check_node_impl(
        task_record,
        llm_client=state.get("_llm_client"),
        logger=logger,
        log_task_event=log_task_event,
    )
    consumed_inputs = result.get("consumed_inputs") if isinstance(result.get("consumed_inputs"), list) else []
    if consumed_inputs:
        _apply_check_consumed_inputs(state=state, task_state=task_state, consumed_inputs=consumed_inputs)
        task_metrics(task_state)["steering_consumed_in_check"] = True
        task_state["goal"] = ""
        task_state["acceptance_criteria"] = []
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="pdca.check.input_consumed",
            cycle=cycle,
            consumed_count=len(consumed_inputs),
            consumed_message_ids=[
                str(item.get("message_id") or "").strip()
                for item in consumed_inputs
                if str(item.get("message_id") or "").strip()
            ][:20],
            consumed_attachment_count=sum(
                len(item.get("attachments")) if isinstance(item.get("attachments"), list) else 0
                for item in consumed_inputs
            ),
        )
    judge_result = result.get("judge_result") if isinstance(result.get("judge_result"), dict) else {}
    verdict = str(result.get("verdict") or "").strip().lower() or _extract_verdict_kind(judge_result)
    task_state["acceptance_criteria"] = apply_criteria_updates(
        existing=[] if consumed_inputs else normalize_acceptance_criteria_records(task_state.get("acceptance_criteria")),
        updates=judge_result.get("criteria_updates"),
        case_type=case_type,
        fallback_user_text=str(
            (
                result.get("task_record").goal
                if isinstance(result.get("task_record"), TaskRecord)
                else task_record.goal
            )
            or ""
        ).strip(),
    )
    return _finalize_check_state_result(
        state=state,
        task_state=task_state,
        task_record=result.get("task_record") if isinstance(result.get("task_record"), TaskRecord) else task_record,
        verdict=verdict,
        judge_result=judge_result,
        logger=logger,
        log_task_event=log_task_event,
        correlation_id=corr,
        cycle=cycle,
    )


def _consume_task_inputs_for_task_record(task_record: TaskRecord) -> list[dict[str, Any]]:
    task_id = str(task_record.task_id or "").strip()
    if not task_id:
        return []
    try:
        raw = consume_task_inputs_for_check(task_id=task_id)
    except Exception:
        return []
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


def _validate_case_type(case_type: str | None) -> str | None:
    rendered = str(case_type or "").strip().lower()
    return rendered if rendered in _CASE_TYPES else None


def route_after_check_impl(state: dict[str, Any]) -> str:
    task_state = state.get("task_state") if isinstance(state.get("task_state"), dict) else {}
    verdict = task_state.get("judge_verdict") if isinstance(task_state.get("judge_verdict"), dict) else {}
    kind = str(verdict.get("kind") or "").strip().lower()
    if kind == "plan":
        return "next_step_node"
    if kind in {"conversation", "mission_success", "mission_failed"}:
        return "respond_node"
    route = str(task_state.get("check_route") or "").strip().lower()
    if route in {"respond_node", "next_step_node"}:
        return route
    status = str(task_state.get("status") or "").strip().lower()
    return "respond_node" if status in {"waiting_user", "failed", "done"} else "next_step_node"


def select_case_deterministically(task_state: dict[str, Any]) -> str | None:
    metrics = task_metrics(task_state)
    provenance = str(metrics.get("check_provenance") or task_state.get("check_provenance") or "").strip().lower()
    mapping = {
        "entry": "new_request",
        "do": "execution_review",
        "slice_resume": "task_resumption",
    }
    return mapping.get(provenance)


def apply_criteria_updates(
    *,
    existing: list[dict[str, Any]],
    updates: Any,
    case_type: str,
    fallback_user_text: str,
) -> list[dict[str, Any]]:
    out = [dict(item) for item in existing]
    index: dict[str, int] = {
        str(item.get("id") or ""): pos for pos, item in enumerate(out) if str(item.get("id") or "").strip()
    }
    for update in updates if isinstance(updates, list) else []:
        if not isinstance(update, dict):
            continue
        op = str(update.get("op") or "").strip().lower()
        if op == "append":
            text = str(update.get("text") or "").strip()
            if not text:
                continue
            if any(str(item.get("text") or "").strip().lower() == text.lower() for item in out):
                continue
            criterion_id = _next_criterion_id(out)
            out.append(
                {
                    "id": criterion_id,
                    "text": text[:180],
                    "status": "pending",
                    "evidence_refs": [],
                    "created_by_case": case_type,
                }
            )
            index[criterion_id] = len(out) - 1
            continue
        if op == "mark_satisfied":
            criterion_id = str(update.get("criterion_id") or "").strip()
            if not criterion_id or criterion_id not in index:
                continue
            pos = index[criterion_id]
            refs = update.get("evidence_refs")
            evidence_refs = [str(ref).strip() for ref in refs if str(ref).strip()][:8] if isinstance(refs, list) else []
            out[pos]["status"] = "satisfied"
            out[pos]["evidence_refs"] = evidence_refs

    if case_type == "new_request" and not out:
        snippet = fallback_user_text.strip()[:120] or "the user request"
        out.append(
            {
                "id": "ac_1",
                "text": f"Advance the request successfully: {snippet}",
                "status": "pending",
                "evidence_refs": [],
                "created_by_case": "new_request",
            }
        )
    return normalize_acceptance_criteria_records(out)


def _build_task_record_from_state(*, state: dict[str, Any], task_state: dict[str, Any]) -> TaskRecord:
    recent_conversation_md = _resolve_recent_conversation_md(state)
    record = TaskRecord(
        task_id=_first_non_empty(task_state.get("task_id"), state.get("task_id")),
        user_id=_first_non_empty(task_state.get("user_id"), state.get("actor_person_id")),
        goal=str(task_state.get("goal") or "").strip(),
        status=str(task_state.get("status") or "").strip() or "running",
        outcome=task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None,
    )
    record.set_recent_conversation_md(recent_conversation_md)
    _append_fact_lines(
        record,
        facts=task_state.get("facts"),
        channel_type=_first_non_empty(task_state.get("channel_type"), state.get("channel_type")),
        channel_target=_first_non_empty(task_state.get("channel_target"), state.get("channel_target")),
        locale=_first_non_empty(task_state.get("locale"), state.get("locale")),
        timezone=_first_non_empty(task_state.get("timezone"), state.get("timezone")),
        message_id=_first_non_empty(task_state.get("message_id"), state.get("message_id")),
        conversation_key=_first_non_empty(task_state.get("conversation_key"), state.get("conversation_key")),
    )
    _append_plan_lines(record, plan=task_state.get("plan"))
    _append_memory_fact_lines(record, memory_facts=task_state.get("memory_facts"))
    _append_tool_call_history_lines(record, tool_call_history=task_state.get("tool_call_history"))
    return record


def _resolve_recent_conversation_md(state: dict[str, Any]) -> str:
    recent = str(state.get("recent_conversation_block") or "").strip()
    if recent:
        return recent
    session_state = state.get("session_state") if isinstance(state.get("session_state"), dict) else None
    if session_state:
        return render_recent_conversation_block(session_state)
    return ""

def _normalize_judge_payload(payload: dict[str, Any], *, case_type: str) -> dict[str, Any] | None:
    raw_kind = str(payload.get("kind") or "").strip().lower()
    kind = raw_kind if raw_kind in _VERDICT_KINDS else "plan"
    if case_type == "new_request":
        kind = "plan"
    if case_type != "new_request" and kind == "conversation":
        kind = "plan"
    confidence = _coerce_confidence(payload.get("confidence"))
    reason = str(payload.get("reason") or "").strip()[:320] or "judge_verdict_issued"
    updates = _normalize_criteria_updates(payload=payload, case_type=case_type)
    evidence_refs = payload.get("evidence_refs")
    normalized_refs = [str(item).strip() for item in evidence_refs if str(item).strip()][:12] if isinstance(evidence_refs, list) else []
    failure_class = payload.get("failure_class")
    normalized_failure_class = str(failure_class).strip()[:120] if isinstance(failure_class, str) and failure_class.strip() else None
    return {
        "kind": kind,
        "case_type": case_type,
        "reason": reason,
        "confidence": confidence,
        "criteria_updates": updates,
        "evidence_refs": normalized_refs,
        "failure_class": normalized_failure_class,
    }


def _normalize_criteria_updates(*, payload: dict[str, Any], case_type: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    raw_updates = payload.get("criteria_updates")
    if isinstance(raw_updates, list):
        for update in raw_updates:
            if not isinstance(update, dict):
                continue
            op = str(update.get("op") or "").strip().lower()
            if op not in {"append", "mark_satisfied"}:
                continue
            if op == "append":
                text = str(update.get("text") or "").strip()
                if not text:
                    continue
                out.append({"op": "append", "text": text[:180]})
                continue
            criterion_id = str(update.get("criterion_id") or "").strip()
            if not criterion_id:
                continue
            refs = update.get("evidence_refs")
            evidence_refs = [str(item).strip() for item in refs if str(item).strip()][:8] if isinstance(refs, list) else []
            out.append({"op": "mark_satisfied", "criterion_id": criterion_id[:40], "evidence_refs": evidence_refs})
    baseline = payload.get("baseline_criteria")
    if case_type == "new_request" and isinstance(baseline, list):
        for item in baseline:
            text = str(item).strip()
            if not text:
                continue
            out.append({"op": "append", "text": text[:180]})
    return out[:24]


def _fallback_trial_judge_result(*, case_type: str, task_record: TaskRecord) -> dict[str, Any]:
    if case_type == "new_request":
        text = str(task_record.goal or "").strip()[:120] or "the user request"
        return {
            "kind": "plan",
            "case_type": "new_request",
            "reason": "No judge model available; applying deterministic baseline.",
            "confidence": 0.0,
            "criteria_updates": [{"op": "append", "text": f"Advance the request successfully: {text}"}],
            "evidence_refs": [],
            "failure_class": None,
        }
    criteria_text = task_record.get_acceptance_criteria_md().lower()
    if criteria_text and "[pending]" not in criteria_text and criteria_text != "- (none)":
        return {
            "kind": "mission_success",
            "case_type": case_type,
            "reason": "All acceptance criteria are already satisfied.",
            "confidence": 0.9,
            "criteria_updates": [],
            "evidence_refs": [],
            "failure_class": None,
        }
    return {
        "kind": "plan",
        "case_type": case_type,
        "reason": "No judge model available; continue with plan.",
        "confidence": 0.0,
        "criteria_updates": [],
        "evidence_refs": [],
        "failure_class": None,
    }


def _call_judge_llm(*, llm_client: object, judge_prompt: str) -> str:
    return _call_llm(
        llm_client=llm_client,
        system_prompt=CHECK_JUDGE_SYSTEM_PROMPT_TEMPLATE,
        user_prompt=judge_prompt,
    )


def _parse_judge_verdict(*, raw: str, case_type: str) -> dict[str, Any] | None:
    parsed = parse_json_object(raw)
    if not isinstance(parsed, dict):
        return None
    return _normalize_judge_payload(parsed, case_type=case_type)


def _finalize_check_state_result(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    task_record: TaskRecord,
    verdict: str,
    judge_result: dict[str, Any],
    logger: Any,
    log_task_event: Any,
    correlation_id: str | None,
    cycle: int,
) -> dict[str, Any]:
    reason = str(judge_result.get("reason") or "").strip()
    task_state["judge_verdict"] = dict(judge_result)
    if verdict == "mission_success":
        task_state["status"] = "done"
        task_state["outcome"] = {
            "kind": "task_completed",
            "summary": reason or "Mission completed successfully.",
            "final_text": reason or "Mission completed successfully.",
            "evidence": {
                "criteria": normalize_acceptance_criteria_records(task_state.get("acceptance_criteria")),
                "verdict_confidence": float(judge_result.get("confidence") or 0.0),
            },
        }
    elif verdict == "mission_failed":
        failure_code = str(judge_result.get("failure_class") or "mission_failed").strip() or "mission_failed"
        failure_reason = reason or "Mission failed."
        retry_exhausted = bool(judge_result.get("retry_exhausted"))
        task_state["status"] = "failed"
        task_state["last_validation_error"] = {
            "reason": failure_code,
            "message": failure_reason,
            "retry_exhausted": retry_exhausted,
        }
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="pdca.failure.classified",
            failure_code=failure_code,
            failure_reason=failure_reason,
            retry_exhausted=retry_exhausted,
            cycle=cycle,
        )
        if retry_exhausted:
            log_task_event(
                logger=logger,
                state=state,
                task_state=task_state,
                node="check_node",
                event="pdca.failure.retry_exhausted",
                failure_code=failure_code,
                failure_reason=failure_reason,
                cycle=cycle,
            )
    elif verdict == "conversation":
        task_state["status"] = "waiting_user"
        task_state["next_user_question"] = reason or "Could you tell me how I can help?"
    else:
        task_state["status"] = "running"
        task_state["next_user_question"] = None
        emit_transition_event(state, "thinking")

    task_record.status = str(task_state.get("status") or "").strip() or "running"
    task_record.outcome = task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None

    check_route = post_check_route(verdict)
    task_state["check_route"] = check_route
    append_trace_event(
        task_state,
        {
            "type": "judge_verdict",
            "summary": f"Check verdict={verdict or 'plan'} case={str(judge_result.get('case_type') or '')}",
            "correlation_id": correlation_id,
        },
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="check_node",
        event="graph.check.judge_verdict",
        verdict_kind=verdict,
        case_type=str(judge_result.get("case_type") or ""),
        confidence=float(judge_result.get("confidence") or 0.0),
        route=check_route,
        cycle=cycle,
    )
    _append_check_criteria_snapshot_event(
        state=state,
        task_state=task_state,
        judge_result=judge_result,
        correlation_id=correlation_id,
        cycle=cycle,
    )
    logger.info(
        "task_mode check correlation_id=%s cycle=%s verdict=%s route=%s",
        correlation_id,
        cycle,
        verdict or "plan",
        check_route,
    )
    return {"task_state": task_state}


def post_check_route(verdict: str) -> str:
    if verdict == "plan":
        return "next_step_node"
    return "respond_node"


def _apply_check_consumed_inputs(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    consumed_inputs: list[dict[str, Any]],
) -> None:
    if not consumed_inputs:
        return
    task_metrics(task_state)["check_provenance"] = "slice_resume"
    latest_text = str(consumed_inputs[-1].get("text") or "").strip()
    if latest_text:
        state["last_user_message"] = latest_text
    facts = task_state.get("facts")
    fact_bucket = dict(facts) if isinstance(facts, dict) else {}
    max_suffix = 0
    for key in fact_bucket:
        rendered = str(key or "").strip()
        if not rendered.startswith("user_reply_"):
            continue
        suffix = rendered.split("user_reply_", 1)[1]
        try:
            max_suffix = max(max_suffix, int(suffix))
        except ValueError:
            continue
    for index, item in enumerate(consumed_inputs, start=1):
        text = str(item.get("text") or "").strip()
        attachments = item.get("attachments")
        normalized_attachments = [dict(att) for att in attachments if isinstance(att, dict)] if isinstance(attachments, list) else []
        if not text and not normalized_attachments:
            continue
        fact_bucket[f"user_reply_{max_suffix + index}"] = {
            "source": "user_reply",
            "text": text,
            "attachments": normalized_attachments,
            "actor_id": str(item.get("actor_id") or "").strip() or None,
            "message_id": str(item.get("message_id") or "").strip() or None,
            "received_at": str(item.get("received_at") or "").strip() or None,
            "consumed_at": str(item.get("consumed_at") or "").strip() or None,
        }
    task_state["facts"] = fact_bucket


def _append_consumed_inputs_to_task_record(
    task_record: TaskRecord,
    *,
    consumed_inputs: list[dict[str, Any]],
) -> None:
    for item in consumed_inputs:
        text = str(item.get("text") or "").strip()
        attachments = item.get("attachments")
        attachment_count = len(attachments) if isinstance(attachments, list) else 0
        if text:
            task_record.append_recent_conversation_line(f"User: {text}")
        elif attachment_count:
            task_record.append_recent_conversation_line(
                f"User sent {attachment_count} attachment{'s' if attachment_count != 1 else ''}."
            )


def _compact_json(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)[:500]
    except Exception:
        return str(value)[:500]


def _next_criterion_id(existing: list[dict[str, Any]]) -> str:
    max_num = 0
    for item in existing:
        criterion_id = str(item.get("id") or "").strip().lower()
        if not criterion_id.startswith("ac_"):
            continue
        tail = criterion_id[3:]
        if tail.isdigit():
            max_num = max(max_num, int(tail))
    return f"ac_{max_num + 1}"


def _call_llm(*, llm_client: object, system_prompt: str, user_prompt: str) -> str:
    try:
        complete = getattr(llm_client, "complete", None)
        if callable(complete):
            return str(complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except Exception:
        return ""
    return ""


def _mission_failed_judge_result(
    *,
    case_type: str,
    reason: str,
    failure_class: str,
    retry_exhausted: bool = False,
) -> dict[str, Any]:
    return {
        "kind": "mission_failed",
        "case_type": case_type if case_type in _CASE_TYPES else "execution_review",
        "reason": reason[:320],
        "confidence": 1.0,
        "criteria_updates": [],
        "evidence_refs": [],
        "failure_class": failure_class[:120],
        "retry_exhausted": bool(retry_exhausted),
    }


def _append_check_criteria_snapshot_event(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    judge_result: dict[str, Any],
    correlation_id: str | None,
    cycle: int,
) -> None:
    task_id = _resolve_task_id(state=state, task_state=task_state, correlation_id=correlation_id)
    if not task_id:
        return
    criteria = normalize_acceptance_criteria_records(task_state.get("acceptance_criteria"))
    history = task_state.get("tool_call_history")
    fact_refs: list[str] = []
    if isinstance(history, list):
        for entry in history[-12:]:
            if not isinstance(entry, dict):
                continue
            if isinstance(entry, dict) and bool(entry.get("internal")):
                continue
            rendered = str(entry.get("step_id") or "").strip()
            if rendered:
                fact_refs.append(f"fact:{rendered[:80]}")
    payload = {
        "case_type": str(judge_result.get("case_type") or "").strip()[:40],
        "cycle": int(cycle),
        "acceptance_criteria": [
            {
                "id": str(item.get("id") or "").strip()[:40],
                "text": str(item.get("text") or "").strip()[:180],
                "status": str(item.get("status") or "pending").strip()[:24],
            }
            for item in criteria[:12]
        ],
        "fact_refs": fact_refs[:12],
        "verdict": {
            "kind": str(judge_result.get("kind") or "").strip()[:32],
            "confidence": float(judge_result.get("confidence") or 0.0),
        },
    }
    reason = str(judge_result.get("reason") or "").strip()
    if reason:
        payload["verdict"]["reason"] = reason[:220]
    try:
        append_pdca_event(
            task_id=task_id,
            event_type="check.criteria_snapshot",
            payload=payload,
            correlation_id=correlation_id,
        )
    except Exception:
        return


def _resolve_task_id(*, state: dict[str, Any], task_state: dict[str, Any], correlation_id: str | None) -> str:
    for candidate in (task_state.get("task_id"), state.get("task_id"), state.get("pdca_task_id")):
        rendered = str(candidate or "").strip()
        if rendered:
            return rendered
    corr = str(correlation_id or state.get("correlation_id") or "").strip()
    if corr.startswith("pdca.slice.requested:"):
        parts = corr.split(":")
        if len(parts) >= 3:
            return str(parts[1]).strip()
    return ""


def _coerce_confidence(value: Any) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    if raw < 0.0:
        return 0.0
    if raw > 1.0:
        return 1.0
    return raw


def _append_fact_lines(
    record: TaskRecord,
    *,
    facts: Any,
    channel_type: str | None,
    channel_target: str | None,
    locale: str | None,
    timezone: str | None,
    message_id: str | None,
    conversation_key: str | None,
) -> None:
    facts = facts if isinstance(facts, dict) else {}
    for key, value in list(facts.items())[:20]:
        record.append_fact(f"{str(key)}: {_compact_json(value)}")
    for key, value in (
        ("channel_type", channel_type),
        ("channel_target", channel_target),
        ("locale", locale),
        ("timezone", timezone),
        ("message_id", message_id),
        ("conversation_key", conversation_key),
    ):
        if value:
            record.append_fact(f"{key}: {_compact_json(value)}")


def _append_plan_lines(record: TaskRecord, *, plan: Any) -> None:
    plan = plan if isinstance(plan, dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    for step in steps[-10:]:
        if not isinstance(step, dict):
            continue
        proposal = step.get("proposal") if isinstance(step.get("proposal"), dict) else {}
        tool_name = str(proposal.get("tool_name") or "").strip() or "(none)"
        args = proposal.get("args") if isinstance(proposal.get("args"), dict) else {}
        record.append_plan_line(
            f"{str(step.get('step_id') or '').strip() or '(unknown)'} "
            f"[{str(step.get('status') or '').strip() or 'unknown'}] "
            f"{tool_name} args={_compact_json(args)}"
        )


def _append_memory_fact_lines(record: TaskRecord, *, memory_facts: Any) -> None:
    items = memory_facts
    if not isinstance(items, list):
        return
    for entry in items[-8:]:
        if not isinstance(entry, dict):
            continue
        record.append_memory_fact(
            f"{str(entry.get('tool_name') or '').strip() or 'memory'} "
            f"output={_compact_json(entry.get('output'))} "
            f"exception={_compact_json(entry.get('exception'))}"
        )


def _append_tool_call_history_lines(record: TaskRecord, *, tool_call_history: Any) -> None:
    history = tool_call_history
    if not isinstance(history, list):
        return
    for entry in history[-12:]:
        if not isinstance(entry, dict):
            continue
        record.append_tool_call_history_entry(
            f"{str(entry.get('step_id') or '').strip() or '(no-step)'} "
            f"{str(entry.get('tool_name') or entry.get('tool') or '').strip() or '(unknown)'} "
            f"args={_compact_json(entry.get('params') if isinstance(entry.get('params'), dict) else entry.get('args'))} "
            f"output={_compact_json(entry.get('output'))} "
            f"exception={_compact_json(entry.get('exception'))}"
        )


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        rendered = str(value or "").strip()
        if rendered:
            return rendered
    return None
