from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from alphonse.agent.cognition.prompt_templates_runtime import CHECK_JUDGE_USER_TEMPLATE
from alphonse.agent.cognition.prompt_templates_runtime import CHECK_SYSTEM_PROMPT
from alphonse.agent.cognition.prompt_templates_runtime import render_prompt_template
from alphonse.agent.cognition.utterance_policy import render_utterance_policy_block
from alphonse.agent.cortex.llm_output.json_parse import parse_json_object
from alphonse.agent.cortex.task_mode.task_state_helpers import append_trace_event
from alphonse.agent.cortex.task_mode.task_state_helpers import correlation_id
from alphonse.agent.cortex.task_mode.task_state_helpers import normalize_acceptance_criteria_records
from alphonse.agent.cortex.task_mode.task_state_helpers import task_state_with_defaults
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.nervous_system.pdca_queue_store import append_pdca_event
from alphonse.agent.session.day_state import render_recent_conversation_block

_JUDGE_INVALID_BUDGET_DEFAULT = 2
_ZERO_PROGRESS_BUDGET_DEFAULT = 2
_REPEATED_FAILURE_BUDGET_DEFAULT = 2
_PLANNER_INVALID_BUDGET_DEFAULT = 2

_CASE_TYPES = {"new_request", "execution_review", "task_resumption"}
_VERDICT_KINDS = {"conversation", "plan", "mission_success", "mission_failed"}


class JudgePromptTemplateError(RuntimeError):
    def __init__(self, *, template_id: str, message: str) -> None:
        super().__init__(message)
        self.template_id = template_id


def check_node_impl(
    state: dict[str, Any],
    *,
    tool_registry: Any,
    logger: Any,
    log_task_event: Any,
    wip_emit_every_cycles: int,
) -> dict[str, Any]:
    _ = (tool_registry, wip_emit_every_cycles)
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "check"
    corr = correlation_id(state)
    task_state["acceptance_criteria"] = normalize_acceptance_criteria_records(task_state.get("acceptance_criteria"))
    cycle = int(task_state.get("cycle_index") or 0)

    case_type = select_case_deterministically(task_state)
    if case_type is None:
        verdict = _mission_failed_verdict(
            case_type="execution_review",
            reason="check_provenance is required and must be one of: entry|do|slice_resume.",
            failure_class="invalid_provenance",
        )
        return _finalize_check_cycle(
            state=state,
            task_state=task_state,
            verdict=verdict,
            logger=logger,
            log_task_event=log_task_event,
            corr=corr,
            cycle=cycle,
        )

    planner_invalid_streak = _update_planner_invalid_streak(task_state)
    if planner_invalid_streak > 0 and planner_invalid_streak <= _planner_invalid_budget():
        append_trace_event(
            task_state,
            {
                "type": "planner_retry",
                "summary": (
                    f"Planner output invalid; retrying planning "
                    f"({planner_invalid_streak}/{_planner_invalid_budget()})."
                ),
                "correlation_id": corr,
            },
        )
    repeated_failure_streak = _update_repeated_failure_signature(task_state)
    zero_progress_streak = _update_zero_progress_state(task_state)

    try:
        verdict = run_judge(
            state=state,
            task_state=task_state,
            case_type=case_type,
            diagnostic_context={
                "planner_invalid_streak": planner_invalid_streak,
                "planner_invalid_budget": _planner_invalid_budget(),
                "repeated_failure_signature_streak": repeated_failure_streak,
                "repeated_failure_signature_budget": _repeated_failure_budget(),
                "zero_progress_streak": zero_progress_streak,
                "zero_progress_budget": _zero_progress_budget(),
            },
        )
    except JudgePromptTemplateError as exc:
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="judge.prompt_template.failed",
            level="error",
            error_code="judge_prompt_template_failed",
            template_id=exc.template_id,
            error_message=str(exc),
            cycle=cycle,
        )
        verdict = _mission_failed_verdict(
            case_type=case_type,
            reason="The templating system failed. Please contact the admin.",
            failure_class="judge_prompt_template_failed",
            retry_exhausted=True,
        )
        return _finalize_check_cycle(
            state=state,
            task_state=task_state,
            verdict=verdict,
            logger=logger,
            log_task_event=log_task_event,
            corr=corr,
            cycle=cycle,
        )
    if not isinstance(verdict, dict):
        streak = int(task_state.get("judge_invalid_streak") or 0) + 1
        task_state["judge_invalid_streak"] = streak
        if streak > _judge_invalid_budget():
            verdict = _mission_failed_verdict(
                case_type=case_type,
                reason="Judge output remained invalid beyond retry budget.",
                failure_class="judge_output_invalid",
                retry_exhausted=True,
            )
        else:
            verdict = {
                "kind": "plan",
                "case_type": case_type,
                "reason": "Judge output invalid; continuing with deterministic retry policy.",
                "confidence": 0.0,
                "criteria_updates": [],
                "evidence_refs": [],
                "failure_class": None,
            }
    else:
        task_state["judge_invalid_streak"] = 0

    task_state["acceptance_criteria"] = apply_criteria_updates(
        existing=normalize_acceptance_criteria_records(task_state.get("acceptance_criteria")),
        updates=verdict.get("criteria_updates"),
        case_type=case_type,
        fallback_user_text=str(state.get("last_user_message") or "").strip(),
    )
    return _finalize_check_cycle(
        state=state,
        task_state=task_state,
        verdict=verdict,
        logger=logger,
        log_task_event=log_task_event,
        corr=corr,
        cycle=cycle,
    )


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
    provenance = str(task_state.get("check_provenance") or "").strip().lower()
    mapping = {
        "entry": "new_request",
        "do": "execution_review",
        "slice_resume": "task_resumption",
    }
    return mapping.get(provenance)


def run_judge(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    case_type: str,
    diagnostic_context: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if case_type not in _CASE_TYPES:
        return None
    llm_client = state.get("_llm_client")
    if llm_client is None:
        return _fallback_judge_verdict(case_type=case_type, state=state, task_state=task_state)

    user_prompt = _build_judge_user_prompt(
        state=state,
        task_state=task_state,
        case_type=case_type,
        diagnostic_context=diagnostic_context,
    )
    raw = _call_llm(llm_client=llm_client, system_prompt=CHECK_SYSTEM_PROMPT, user_prompt=user_prompt)
    parsed = parse_json_object(raw)
    if not isinstance(parsed, dict):
        return None
    normalized = _normalize_judge_payload(parsed, case_type=case_type)
    if normalized is None:
        return None
    return normalized


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


def _build_judge_user_prompt(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    case_type: str,
    diagnostic_context: dict[str, Any] | None = None,
) -> str:
    recent = str(state.get("recent_conversation_block") or "").strip()
    if not recent:
        session_state = state.get("session_state") if isinstance(state.get("session_state"), dict) else None
        if session_state:
            recent = render_recent_conversation_block(session_state)
    criteria = normalize_acceptance_criteria_records(task_state.get("acceptance_criteria"))
    criteria_lines = "\n".join(
        f"- {str(item.get('id') or '')}: {str(item.get('text') or '')} [{str(item.get('status') or 'pending')}]"
        for item in criteria
    ) or "- (none)"
    fact_lines = _render_fact_evidence_blocks(task_state)
    policy = render_utterance_policy_block(
        locale=state.get("locale"),
        tone=state.get("tone"),
        address_style=state.get("address_style"),
        channel_type=state.get("channel_type"),
    )
    diagnostic = diagnostic_context if isinstance(diagnostic_context, dict) else {}
    diagnostic_lines = (
        f"- planner_invalid_streak: {int(diagnostic.get('planner_invalid_streak') or 0)} / "
        f"{int(diagnostic.get('planner_invalid_budget') or _planner_invalid_budget())}\n"
        f"- repeated_failure_signature_streak: {int(diagnostic.get('repeated_failure_signature_streak') or 0)} / "
        f"{int(diagnostic.get('repeated_failure_signature_budget') or _repeated_failure_budget())}\n"
        f"- zero_progress_streak: {int(diagnostic.get('zero_progress_streak') or 0)} / "
        f"{int(diagnostic.get('zero_progress_budget') or _zero_progress_budget())}"
    )
    try:
        return render_prompt_template(
            CHECK_JUDGE_USER_TEMPLATE,
            {
                "POLICY_BLOCK": policy,
                "CASE_TYPE": case_type,
                "RECENT_CONVERSATION": recent or "- (none)",
                "GOAL": str(task_state.get("goal") or "").strip(),
                "ACCEPTANCE_CRITERIA_BASELINE": criteria_lines,
                "FACT_EVIDENCE_SNAPSHOT": fact_lines,
                "USER_MESSAGE": str(state.get("last_user_message") or "").strip(),
                "DIAGNOSTIC_BUDGET_CONTEXT": diagnostic_lines,
            },
        )
    except Exception as exc:
        raise JudgePromptTemplateError(
            template_id="check.judge.user.j2",
            message=f"failed to render check judge template: {exc}",
        ) from exc

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


def _fallback_judge_verdict(*, case_type: str, state: dict[str, Any], task_state: dict[str, Any]) -> dict[str, Any]:
    if case_type == "new_request":
        text = str(state.get("last_user_message") or "").strip()[:120] or "the user request"
        return {
            "kind": "plan",
            "case_type": "new_request",
            "reason": "No judge model available; applying deterministic baseline.",
            "confidence": 0.0,
            "criteria_updates": [{"op": "append", "text": f"Advance the request successfully: {text}"}],
            "evidence_refs": [],
            "failure_class": None,
        }
    criteria = normalize_acceptance_criteria_records(task_state.get("acceptance_criteria"))
    if criteria and all(str(item.get("status") or "").strip().lower() == "satisfied" for item in criteria):
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


def _finalize_check_cycle(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    verdict: dict[str, Any],
    logger: Any,
    log_task_event: Any,
    corr: str | None,
    cycle: int,
) -> dict[str, Any]:
    kind = str(verdict.get("kind") or "").strip().lower()
    reason = str(verdict.get("reason") or "").strip()
    task_state["judge_verdict"] = dict(verdict)
    if kind == "mission_success":
        task_state["status"] = "done"
        task_state["outcome"] = {
            "kind": "task_completed",
            "summary": reason or "Mission completed successfully.",
            "final_text": reason or "Mission completed successfully.",
            "evidence": {
                "criteria": normalize_acceptance_criteria_records(task_state.get("acceptance_criteria")),
                "verdict_confidence": float(verdict.get("confidence") or 0.0),
            },
        }
    elif kind == "mission_failed":
        failure_code = str(verdict.get("failure_class") or "mission_failed").strip() or "mission_failed"
        failure_reason = reason or "Mission failed."
        retry_exhausted = bool(verdict.get("retry_exhausted"))
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
    elif kind == "conversation":
        task_state["status"] = "waiting_user"
        task_state["next_user_question"] = reason or "Could you tell me how I can help?"
    else:
        task_state["status"] = "running"
        task_state["next_user_question"] = None
        emit_transition_event(state, "thinking")

    check_route = post_check_route(verdict)
    task_state["check_route"] = check_route
    append_trace_event(
        task_state,
        {
            "type": "judge_verdict",
            "summary": f"Check verdict={kind or 'plan'} case={str(verdict.get('case_type') or '')}",
            "correlation_id": corr,
        },
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="check_node",
        event="graph.check.judge_verdict",
        verdict_kind=kind,
        case_type=str(verdict.get("case_type") or ""),
        confidence=float(verdict.get("confidence") or 0.0),
        route=check_route,
        cycle=cycle,
    )
    _append_check_criteria_snapshot_event(
        state=state,
        task_state=task_state,
        verdict=verdict,
        correlation_id=corr,
        cycle=cycle,
    )
    logger.info(
        "task_mode check correlation_id=%s cycle=%s verdict=%s route=%s",
        corr,
        cycle,
        kind or "plan",
        check_route,
    )
    return {"task_state": task_state}


def post_check_route(verdict: dict[str, Any]) -> str:
    kind = str(verdict.get("kind") or "").strip().lower()
    if kind == "plan":
        return "next_step_node"
    return "respond_node"


def _update_planner_invalid_streak(task_state: dict[str, Any]) -> int:
    facts = task_state.get("facts")
    if not isinstance(facts, dict) or not facts:
        task_state["planner_error_streak"] = 0
        task_state["planner_error_last_fact_key"] = None
        return 0
    latest_key, latest_entry = next(reversed(facts.items()))
    if not isinstance(latest_entry, dict) or str(latest_entry.get("tool") or "").strip() != "planner_output":
        task_state["planner_error_streak"] = 0
        task_state["planner_error_last_fact_key"] = None
        return 0
    key_text = str(latest_key)
    last_key = str(task_state.get("planner_error_last_fact_key") or "")
    if key_text == last_key:
        return int(task_state.get("planner_error_streak") or 0)
    streak = int(task_state.get("planner_error_streak") or 0) + 1
    task_state["planner_error_streak"] = streak
    task_state["planner_error_last_fact_key"] = key_text
    return streak


def _update_repeated_failure_signature(task_state: dict[str, Any]) -> int:
    signature = _latest_failure_signature(task_state)
    if not signature:
        task_state["check_failure_signature_last"] = None
        task_state["check_failure_signature_streak"] = 0
        return 0
    last = str(task_state.get("check_failure_signature_last") or "").strip()
    streak = int(task_state.get("check_failure_signature_streak") or 0)
    streak = streak + 1 if signature == last else 1
    task_state["check_failure_signature_last"] = signature
    task_state["check_failure_signature_streak"] = streak
    return streak


def _update_zero_progress_state(task_state: dict[str, Any]) -> int:
    signature = _latest_mission_fact_signature(task_state)
    if not signature:
        task_state["zero_progress_last_signature"] = None
        task_state["zero_progress_streak"] = 0
        return 0
    last = str(task_state.get("zero_progress_last_signature") or "").strip()
    streak = int(task_state.get("zero_progress_streak") or 0)
    streak = streak + 1 if signature == last else 0
    task_state["zero_progress_last_signature"] = signature
    task_state["zero_progress_streak"] = streak
    return streak


def _render_fact_evidence_blocks(task_state: dict[str, Any]) -> str:
    facts = task_state.get("facts")
    if not isinstance(facts, dict) or not facts:
        return "- (none)"
    lines: list[str] = []
    for key, entry in list(facts.items())[-10:]:
        if not isinstance(entry, dict):
            continue
        tool = str(entry.get("tool_name") or entry.get("tool") or "").strip() or "unknown_tool"
        params = entry.get("params")
        if not isinstance(params, dict):
            raw_args = entry.get("args")
            params = raw_args if isinstance(raw_args, dict) else {}
        output = entry.get("output")
        if output is None and isinstance(entry.get("result"), dict):
            output = entry["result"].get("output")
        exception = entry.get("exception")
        if exception is None and isinstance(entry.get("result"), dict):
            exception = entry["result"].get("exception")
        lines.append(f"- fact:{str(key)}")
        lines.append(f"  tool_name: {tool}")
        lines.append(f"  params: {_compact_json(params)}")
        lines.append(f"  output: {_compact_json(output)}")
        lines.append(f"  exception: {_compact_json(exception)}")
    return "\n".join(lines) if lines else "- (none)"


def _latest_failure_signature(task_state: dict[str, Any]) -> str:
    facts = task_state.get("facts")
    if not isinstance(facts, dict) or not facts:
        return ""
    for _, entry in reversed(list(facts.items())):
        if not isinstance(entry, dict):
            continue
        if bool(entry.get("internal")):
            continue
        exception = _fact_exception_payload(entry)
        if not _has_exception_payload(exception):
            continue
        tool = str(entry.get("tool_name") or entry.get("tool") or "").strip()
        params = entry.get("params")
        if not isinstance(params, dict):
            params = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        raw = f"{tool}|{_compact_json(params)}|{_compact_json(exception)}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return ""


def _latest_mission_fact_signature(task_state: dict[str, Any]) -> str:
    facts = task_state.get("facts")
    if not isinstance(facts, dict) or not facts:
        return ""
    for _, entry in reversed(list(facts.items())):
        if not isinstance(entry, dict):
            continue
        if bool(entry.get("internal")):
            continue
        tool = str(entry.get("tool_name") or entry.get("tool") or "").strip()
        params = entry.get("params")
        if not isinstance(params, dict):
            params = entry.get("args") if isinstance(entry.get("args"), dict) else {}
        payload = entry.get("output")
        if payload is None and isinstance(entry.get("result"), dict):
            payload = entry["result"].get("output")
        exception = _fact_exception_payload(entry)
        raw = f"{tool}|{_compact_json(params)}|{_compact_json(payload)}|{_compact_json(exception)}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return ""


def _fact_exception_payload(entry: dict[str, Any]) -> Any:
    exception = entry.get("exception")
    if exception is not None:
        return exception
    result = entry.get("result")
    if isinstance(result, dict):
        return result.get("exception")
    return None


def _has_exception_payload(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        if str(value.get("message") or "").strip():
            return True
        if str(value.get("code") or "").strip():
            return True
        return bool(value)
    return True


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


def _mission_failed_verdict(
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
    verdict: dict[str, Any],
    correlation_id: str | None,
    cycle: int,
) -> None:
    task_id = _resolve_task_id(state=state, task_state=task_state, correlation_id=correlation_id)
    if not task_id:
        return
    criteria = normalize_acceptance_criteria_records(task_state.get("acceptance_criteria"))
    facts = task_state.get("facts")
    fact_refs: list[str] = []
    if isinstance(facts, dict):
        for key, entry in list(facts.items())[-12:]:
            if isinstance(entry, dict) and bool(entry.get("internal")):
                continue
            rendered = str(key).strip()
            if rendered:
                fact_refs.append(f"fact:{rendered[:80]}")
    payload = {
        "case_type": str(verdict.get("case_type") or "").strip()[:40],
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
            "kind": str(verdict.get("kind") or "").strip()[:32],
            "confidence": float(verdict.get("confidence") or 0.0),
        },
    }
    reason = str(verdict.get("reason") or "").strip()
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


def _judge_invalid_budget() -> int:
    raw = str(os.getenv("ALPHONSE_CHECK_JUDGE_INVALID_BUDGET") or "").strip()
    if not raw:
        return _JUDGE_INVALID_BUDGET_DEFAULT
    try:
        parsed = int(raw)
    except ValueError:
        return _JUDGE_INVALID_BUDGET_DEFAULT
    return max(0, parsed)


def _zero_progress_budget() -> int:
    raw = str(os.getenv("ALPHONSE_CHECK_ZERO_PROGRESS_BUDGET") or "").strip()
    if not raw:
        return _ZERO_PROGRESS_BUDGET_DEFAULT
    try:
        parsed = int(raw)
    except ValueError:
        return _ZERO_PROGRESS_BUDGET_DEFAULT
    return max(1, parsed)


def _repeated_failure_budget() -> int:
    raw = str(os.getenv("ALPHONSE_CHECK_REPEATED_FAILURE_BUDGET") or "").strip()
    if not raw:
        return _REPEATED_FAILURE_BUDGET_DEFAULT
    try:
        parsed = int(raw)
    except ValueError:
        return _REPEATED_FAILURE_BUDGET_DEFAULT
    return max(1, parsed)


def _planner_invalid_budget() -> int:
    raw = str(os.getenv("ALPHONSE_TASK_MODE_PLANNER_RETRY_BUDGET") or "").strip()
    if not raw:
        return _PLANNER_INVALID_BUDGET_DEFAULT
    try:
        parsed = int(raw)
    except ValueError:
        return _PLANNER_INVALID_BUDGET_DEFAULT
    return max(0, parsed)


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
