from __future__ import annotations

from typing import Any, TypedDict

from alphonse.agent.cognition.prompt_templates_runtime import CHECK_JUDGE_SYSTEM_PROMPT_TEMPLATE
from alphonse.agent.cognition.prompt_templates_runtime import CHECK_JUDGE_USER_PROMPT_TEMPLATE
from alphonse.agent.cognition.prompt_templates_runtime import render_prompt_template
from alphonse.agent.cognition.utterance_policy import render_utterance_policy_block
from alphonse.agent.cortex.llm_output.json_parse import parse_json_object
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.services.pdca_task_inputs import consume_task_inputs_for_check

_CASE_TYPES = {"new_request", "execution_review", "task_resumption"}
_VERDICT_KINDS = {"conversation", "plan", "mission_success", "mission_failed"}


class CheckResult(TypedDict):
    task_record: TaskRecord
    verdict: str
    judge_result: dict[str, Any]
    case_type: str
    status: str
    outcome: dict[str, Any] | None
    reason: str
    confidence: float
    consumed_inputs: list[dict[str, Any]]

class JudgePromptTemplateError(RuntimeError):
    def __init__(self, *, template_id: str, message: str) -> None:
        super().__init__(message)
        self.template_id = template_id


def check_node_impl(
    task_record: TaskRecord,
    *,
    llm_client: Any,
    logger: Any,
    log_task_event: Any,
) -> CheckResult:
    case_type = _validate_case_type(task_record.check_case_type)
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
    result: CheckResult = _build_check_result(
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


def _build_invalid_case_type_result(*, task_record: TaskRecord) -> CheckResult:
    judge_result = _mission_failed_judge_result(
        case_type=str(task_record.check_case_type or "").strip().lower() or "execution_review",
        reason="task_record.check_case_type is required and must be one of: new_request|execution_review|task_resumption.",
        failure_class="invalid_case_type",
    )
    verdict = _extract_verdict_kind(judge_result)
    failed_record = _apply_terminal_result(task_record=task_record, verdict=verdict, judge_result=judge_result)
    return _build_check_result(task_record=failed_record, verdict=verdict, judge_result=judge_result, case_type=str(judge_result.get("case_type") or "execution_review"))


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
        return _mission_failed_judge_result(
            case_type=case_type,
            reason="No judge model available for PDCA Check.",
            failure_class="judge_llm_unavailable",
        )
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


def post_check_route(verdict: str) -> str:
    if verdict == "plan":
        return "next_step_node"
    return "respond_node"


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


def _call_llm(*, llm_client: object, system_prompt: str, user_prompt: str) -> str:
    try:
        complete = getattr(llm_client, "complete", None)
        if callable(complete):
            return str(complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except Exception:
        return ""
    return ""
