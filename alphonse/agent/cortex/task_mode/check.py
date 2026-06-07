from __future__ import annotations

import json
from typing import Any, TypedDict

from alphonse.agent.cognition.providers.contracts import TextCompletionProvider
from alphonse.agent.cognition.prompt_templates_runtime import CHECK_JUDGE_SYSTEM_PROMPT_TEMPLATE
from alphonse.agent.cognition.prompt_templates_runtime import CHECK_JUDGE_USER_PROMPT_TEMPLATE
from alphonse.agent.cognition.prompt_templates_runtime import render_prompt_template
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.services.pdca_task_inputs import consume_task_inputs_for_check

_PROVENANCE_VALUES = {"entry", "do", "slice_resume"}
_CHECK_VERDICTS = {"new", "steer", "wip", "mission_success", "mission_failed"}
_JUDGE_VERDICTS = {"wip", "mission_success", "mission_failed"}


class CheckResult(TypedDict):
    task_record: TaskRecord
    verdict: str
    judge_result: dict[str, Any]
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
    provenance: str,
    llm_client: TextCompletionProvider | None,
    logger: Any,
    log_task_event: Any,
) -> CheckResult:
    if _validate_provenance(provenance) is None:
        _set_check_result(
            task_record,
            verdict="mission_failed",
            reason="Check provenance is invalid.",
            confidence=1.0,
            evidence_refs=[],
            consumed_count=0,
        )
        return _build_check_result(task_record=task_record, consumed_inputs=[])

    consumed_inputs = _consume_task_inputs_for_task_record(task_record)
    if consumed_inputs:
        _append_consumed_inputs_to_task_record(task_record, consumed_inputs=consumed_inputs)
    _ensure_goal_in_conversation_history(task_record)

    consumed_count = len(consumed_inputs)
    if _has_conversation(task_record) and not _has_acceptance_criteria(task_record):
        _set_check_result(
            task_record,
            verdict="new",
            reason="New user request needs acceptance criteria.",
            confidence=1.0,
            evidence_refs=["recent_conversation"],
            consumed_count=consumed_count,
        )
    elif _has_acceptance_criteria(task_record) and consumed_count > 0:
        _set_check_result(
            task_record,
            verdict="steer",
            reason="New user input should steer the existing mission.",
            confidence=1.0,
            evidence_refs=["recent_conversation", "acceptance_criteria"],
            consumed_count=consumed_count,
        )
    else:
        judge_result = _conduct_trial(llm_client=llm_client, task_record=task_record)
        _set_check_result(
            task_record,
            verdict=str(judge_result["verdict"]),
            reason=str(judge_result.get("reason") or ""),
            confidence=float(judge_result.get("confidence") or 0.0),
            evidence_refs=list(judge_result.get("evidence_refs") or []),
            consumed_count=consumed_count,
        )

    result: CheckResult = _build_check_result(task_record=task_record, consumed_inputs=consumed_inputs)
    _log_check_result(
        task_record=task_record,
        logger=logger,
        log_task_event=log_task_event,
    )
    return result


def _conduct_trial(*, llm_client: TextCompletionProvider | None, task_record: TaskRecord) -> dict[str, Any]:
    if llm_client is None:
        return {
            "verdict": "mission_failed",
            "reason": "No judge model is available for PDCA Check.",
            "confidence": 1.0,
            "evidence_refs": [],
            "failure_class": "judge_llm_unavailable",
        }
    judge_prompt = _get_judge_prompt_from_task_record(task_record=task_record)
    raw = _call_judge_llm(llm_client=llm_client, judge_prompt=judge_prompt)
    parsed = _parse_judge_verdict(raw)
    if parsed is None:
        raise ValueError("check_judge_invalid_output")
    return parsed


def _get_judge_prompt_from_task_record(*, task_record: TaskRecord) -> str:
    try:
        return render_prompt_template(
            CHECK_JUDGE_USER_PROMPT_TEMPLATE,
            {
                "RECENT_CONVERSATION": task_record.recent_conversation_md,
                "GOAL": task_record.goal,
                "ACCEPTANCE_CRITERIA_BASELINE": task_record.get_acceptance_criteria_md(),
                "FACTS_SECTION": task_record.get_facts_md(),
                "PLAN_SECTION": task_record.get_plan_md(),
                "MEMORY_FACTS_SECTION": task_record.get_memory_facts_md(),
                "TOOL_CALL_HISTORY_SECTION": task_record.get_tool_call_history_md(),
                "PDCA_CYCLE_COUNT": task_record.pdca_cycle_count,
            },
        )
    except Exception as exc:
        raise JudgePromptTemplateError(
            template_id="pdca.check.judge.user.j2",
            message=f"failed to render check judge template: {exc}",
        ) from exc


def _parse_judge_verdict(raw: Any) -> dict[str, Any] | None:
    parsed = parse_json_object(raw)
    if not isinstance(parsed, dict):
        return None
    verdict = str(parsed.get("verdict") or "").strip().lower()
    if verdict not in _JUDGE_VERDICTS:
        return None
    refs = parsed.get("evidence_refs")
    failure_class = parsed.get("failure_class")
    return {
        "verdict": verdict,
        "reason": str(parsed.get("reason") or "").strip()[:320],
        "confidence": _coerce_confidence(parsed.get("confidence")),
        "evidence_refs": [str(item).strip() for item in refs if str(item).strip()][:12] if isinstance(refs, list) else [],
        "failure_class": str(failure_class).strip()[:120] if isinstance(failure_class, str) and failure_class.strip() else None,
    }


def parse_json_object(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.startswith("json"):
            candidate = candidate[4:].strip()
    parsed = _json_loads(candidate)
    if isinstance(parsed, dict):
        return parsed
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        parsed = _json_loads(candidate[start : end + 1])
        if isinstance(parsed, dict):
            return parsed
    return None


def post_check_route(verdict: str) -> str:
    _ = verdict
    return "act_node"


def _build_check_result(*, task_record: TaskRecord, consumed_inputs: list[dict[str, Any]]) -> CheckResult:
    verdict = str(task_record.check_verdict or "").strip()
    return {
        "task_record": task_record,
        "verdict": verdict,
        "judge_result": {
            "verdict": verdict,
            "reason": task_record.check_reason,
            "confidence": task_record.check_confidence,
            "evidence_refs": list(task_record.check_evidence_refs or []),
        },
        "status": task_record.status,
        "outcome": task_record.outcome,
        "reason": task_record.check_reason,
        "confidence": task_record.check_confidence,
        "consumed_inputs": list(consumed_inputs),
    }


def _set_check_result(
    task_record: TaskRecord,
    *,
    verdict: str,
    reason: str,
    confidence: float,
    evidence_refs: list[str],
    consumed_count: int,
) -> None:
    if verdict not in _CHECK_VERDICTS:
        raise ValueError(f"invalid_check_verdict: {verdict}")
    task_record.set_check_result(
        verdict=verdict,
        reason=reason,
        confidence=confidence,
        evidence_refs=evidence_refs,
        new_message_count=consumed_count,
    )
    task_record.status = "running"
    task_record.outcome = None


def _consume_task_inputs_for_task_record(task_record: TaskRecord) -> list[dict[str, Any]]:
    task_id = str(task_record.task_id or "").strip()
    if not task_id:
        return []
    raw = consume_task_inputs_for_check(task_id=task_id)
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict)]


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


def _ensure_goal_in_conversation_history(task_record: TaskRecord) -> None:
    goal = str(task_record.goal or "").strip()
    if not goal or _has_conversation(task_record):
        return
    task_record.append_recent_conversation_line(f"User: {goal}")


def _has_conversation(task_record: TaskRecord) -> bool:
    return bool(str(task_record.recent_conversation_md or "").strip() not in {"", "- (none)"})


def _has_acceptance_criteria(task_record: TaskRecord) -> bool:
    return bool(str(task_record.get_acceptance_criteria_md() or "").strip() not in {"", "- (none)"})


def _validate_provenance(provenance: str | None) -> str | None:
    rendered = str(provenance or "").strip().lower()
    return rendered if rendered in _PROVENANCE_VALUES else None


def _call_judge_llm(*, llm_client: object, judge_prompt: str) -> str:
    complete = getattr(llm_client, "complete", None)
    if callable(complete):
        return str(
            complete(
                system_prompt=CHECK_JUDGE_SYSTEM_PROMPT_TEMPLATE,
                user_prompt=judge_prompt,
            )
        )
    raise ValueError("check_judge_llm_missing_complete")


def _log_check_result(
    *,
    task_record: TaskRecord,
    logger: Any,
    log_task_event: Any,
) -> None:
    verdict = str(task_record.check_verdict or "").strip()
    logger.info(
        "task_mode check verdict=%s status=%s",
        verdict,
        task_record.status,
    )
    log_task_event(
        logger=logger,
        state={"correlation_id": None, "channel_type": None, "actor_person_id": task_record.user_id},
        node="check_node",
        event="graph.check.verdict",
        task_record=task_record,
        cycle_index=task_record.pdca_cycle_count,
        verdict_kind=verdict,
        confidence=task_record.check_confidence,
        route="act_node",
    )


def _json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


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
