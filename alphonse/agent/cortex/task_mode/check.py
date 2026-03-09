from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from alphonse.agent.cognition.prompt_templates_runtime import CHECK_SYSTEM_PROMPT
from alphonse.agent.cognition.utterance_policy import render_utterance_policy_block
from alphonse.agent.cortex.llm_output.json_parse import parse_json_object
from alphonse.agent.cortex.task_mode.task_state_helpers import append_trace_event
from alphonse.agent.cortex.task_mode.task_state_helpers import correlation_id
from alphonse.agent.cortex.task_mode.task_state_helpers import normalize_acceptance_criteria_records
from alphonse.agent.cortex.task_mode.task_state_helpers import task_state_with_defaults
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.session.day_state import render_recent_conversation_block

_JUDGE_INVALID_BUDGET_DEFAULT = 2
_ZERO_PROGRESS_BUDGET_DEFAULT = 2
_REPEATED_FAILURE_BUDGET_DEFAULT = 2
_PLANNER_INVALID_BUDGET_DEFAULT = 2

_CASE_TYPES = {"new_request", "execution_review", "task_resumption"}
_VERDICT_KINDS = {"conversation", "plan", "mission_success", "mission_failed"}


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
    if planner_invalid_streak > _planner_invalid_budget():
        verdict = _mission_failed_verdict(
            case_type=case_type,
            reason="Planner output remained invalid beyond retry budget.",
            failure_class="invalid_planner_output",
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

    hard_stop = _deterministic_hard_stop(task_state=task_state, case_type=case_type)
    if isinstance(hard_stop, dict):
        return _finalize_check_cycle(
            state=state,
            task_state=task_state,
            verdict=hard_stop,
            logger=logger,
            log_task_event=log_task_event,
            corr=corr,
            cycle=cycle,
        )

    verdict = run_judge(
        state=state,
        task_state=task_state,
        case_type=case_type,
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
) -> dict[str, Any] | None:
    if case_type not in _CASE_TYPES:
        return None
    llm_client = state.get("_llm_client")
    if llm_client is None:
        return _fallback_judge_verdict(case_type=case_type, state=state, task_state=task_state)

    user_prompt = _build_judge_user_prompt(state=state, task_state=task_state, case_type=case_type)
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


def _build_judge_user_prompt(*, state: dict[str, Any], task_state: dict[str, Any], case_type: str) -> str:
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
    fact_lines = _compact_fact_lines(task_state)
    policy = render_utterance_policy_block(
        locale=state.get("locale"),
        tone=state.get("tone"),
        address_style=state.get("address_style"),
        channel_type=state.get("channel_type"),
    )
    dynamic_contract = _dynamic_case_contract(case_type)
    return (
        "## ROLE\n"
        "You are the CAPD Check Judge. Judge only. Do not plan or execute.\n\n"
        "## POLICY\n"
        f"{policy}\n\n"
        "## CASE TYPE\n"
        f"{case_type}\n\n"
        "## CASE PROTOCOL\n"
        f"{dynamic_contract}\n\n"
        "## RECENT CONVERSATION\n"
        f"{recent or '- (none)'}\n\n"
        "## GOAL\n"
        f"{str(task_state.get('goal') or '').strip()}\n\n"
        "## ACCEPTANCE CRITERIA BASELINE\n"
        f"{criteria_lines}\n\n"
        "## FACT EVIDENCE SNAPSHOT\n"
        f"{fact_lines}\n\n"
        "## USER MESSAGE\n"
        f"{str(state.get('last_user_message') or '').strip()}\n\n"
        "## STRICT JSON OUTPUT\n"
        "{\n"
        '  "kind": "conversation|plan|mission_success|mission_failed",\n'
        '  "case_type": "new_request|execution_review|task_resumption",\n'
        '  "reason": "short reason",\n'
        '  "confidence": 0.0,\n'
        '  "criteria_updates": [{"op":"append","text":"..."},{"op":"mark_satisfied","criterion_id":"ac_1","evidence_refs":["fact:step_1"]}],\n'
        '  "evidence_refs": ["fact:step_1"],\n'
        '  "failure_class": null\n'
        "}\n"
    )


def _dynamic_case_contract(case_type: str) -> str:
    if case_type == "new_request":
        return (
            "- Always produce PLAN verdict.\n"
            "- Produce a global acceptance criteria baseline via criteria_updates append operations.\n"
            "- Treat conversational asks as low-complexity goals but still PLAN."
        )
    return (
        "- Review all facts against acceptance criteria.\n"
        "- Mark satisfied criteria by id when evidence exists.\n"
        "- Use mission_success only if all criteria are satisfied.\n"
        "- Use mission_failed if clearly blocked or terminally failed."
    )


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


def _deterministic_hard_stop(*, task_state: dict[str, Any], case_type: str) -> dict[str, Any] | None:
    if case_type == "new_request":
        return None

    repeated = _update_repeated_failure_signature(task_state)
    if repeated > _repeated_failure_budget():
        return _mission_failed_verdict(
            case_type=case_type,
            reason="Repeated identical failure signature exceeded hard-stop limit.",
            failure_class="repeated_failure_signature",
            retry_exhausted=True,
        )

    zero_progress = _update_zero_progress_state(task_state)
    if zero_progress > _zero_progress_budget():
        return _mission_failed_verdict(
            case_type=case_type,
            reason="Zero-progress streak exceeded hard-stop limit.",
            failure_class="zero_progress_streak",
            retry_exhausted=True,
        )

    return None


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


def _compact_fact_lines(task_state: dict[str, Any]) -> str:
    facts = task_state.get("facts")
    if not isinstance(facts, dict) or not facts:
        return "- (none)"
    lines: list[str] = []
    for key, entry in list(facts.items())[-10:]:
        if not isinstance(entry, dict):
            continue
        tool = str(entry.get("tool") or "").strip() or "unknown_tool"
        status = str(entry.get("status") or "").strip().lower() or _status_from_result(entry.get("result"))
        lines.append(f"- fact:{str(key)} tool={tool} status={status or 'unknown'}")
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
        status = str(entry.get("status") or "").strip().lower() or _status_from_result(entry.get("result"))
        if status not in {"failed", "error"}:
            continue
        tool = str(entry.get("tool") or "").strip()
        error = entry.get("error") if isinstance(entry.get("error"), dict) else {}
        if not error and isinstance(entry.get("result"), dict):
            result_error = entry["result"].get("error")
            error = result_error if isinstance(result_error, dict) else {}
        code = str(error.get("code") or "").strip()
        message = str(error.get("message") or "").strip()
        raw = f"{tool}|{status}|{code}|{message}"
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
        tool = str(entry.get("tool") or "").strip()
        status = str(entry.get("status") or "").strip().lower() or _status_from_result(entry.get("result"))
        payload = entry.get("result_payload")
        if payload is None:
            payload = entry.get("result")
        try:
            payload_text = json.dumps(payload, ensure_ascii=False, sort_keys=True)[:400]
        except Exception:
            payload_text = str(payload)[:400]
        raw = f"{tool}|{status}|{payload_text}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return ""


def _status_from_result(result: Any) -> str:
    if isinstance(result, dict):
        status = str(result.get("status") or "").strip().lower()
        if status:
            return status
        if isinstance(result.get("error"), dict):
            return "failed"
    return ""


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
