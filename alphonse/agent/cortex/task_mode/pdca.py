from __future__ import annotations

import json
from typing import Any, Callable

from alphonse.agent.cortex.task_mode.check import check_node_impl
from alphonse.agent.cortex.task_mode.check import post_check_route
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.cortex.task_mode.plan import build_next_step_node_impl
from alphonse.agent.cortex.task_mode.plan import route_after_next_step_impl
from alphonse.agent.cortex.task_mode.observability import log_task_event
from alphonse.agent.cortex.task_mode.execute_step import execute_step_node_impl
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.cortex.task_mode.task_state_helpers import append_trace_event
from alphonse.agent.cortex.task_mode.task_state_helpers import correlation_id
from alphonse.agent.cortex.task_mode.task_state_helpers import current_step
from alphonse.agent.cortex.task_mode.task_state_helpers import has_acceptance_criteria
from alphonse.agent.cortex.task_mode.task_state_helpers import next_step_id
from alphonse.agent.cortex.task_mode.task_state_helpers import normalize_acceptance_criteria_records
from alphonse.agent.cortex.task_mode.task_state_helpers import normalize_acceptance_criteria_values
from alphonse.agent.cortex.task_mode.task_state_helpers import task_plan
from alphonse.agent.cortex.task_mode.task_state_helpers import task_metrics
from alphonse.agent.cortex.task_mode.task_state_helpers import task_state_with_defaults
from alphonse.agent.cortex.task_mode.task_state_helpers import task_trace
from alphonse.agent.nervous_system.pdca_queue_store import append_pdca_event
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.session.day_state import render_recent_conversation_block
logger = get_component_logger("task_mode.pdca")

_TRANSITION_EVENT_THINKING = "thinking"
_TRANSITION_EVENT_EXECUTING = "executing"

def build_next_step_node(*, tool_registry: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _node(state: dict[str, Any]) -> dict[str, Any]:
        emit_transition_event(state, _TRANSITION_EVENT_THINKING)
        return build_next_step_node_impl(
            state=state,
            tool_registry=tool_registry,
            task_state_with_defaults=task_state_with_defaults,
            correlation_id=correlation_id,
            next_step_id=next_step_id,
            task_plan=task_plan,
            has_acceptance_criteria=has_acceptance_criteria,
            normalize_acceptance_criteria_values=normalize_acceptance_criteria_values,
            append_trace_event=append_trace_event,
            logger=logger,
            log_task_event=log_task_event,
        )

    return _node


def route_after_next_step(state: dict[str, Any]) -> str:
    return route_after_next_step_impl(
        state,
        correlation_id=correlation_id,
        logger=logger,
    )


def validate_step_node(state: dict[str, Any], *, tool_registry: Any) -> dict[str, Any]:
    # Compatibility wrapper: validation stage was removed from runtime wiring.
    _ = tool_registry
    return {"task_state": task_state_with_defaults(state)}


def route_after_validate_step(state: dict[str, Any]) -> str:
    # Compatibility wrapper: runtime path now routes through check -> act.
    task_state = state.get("task_state")
    if isinstance(task_state, dict) and str(task_state.get("status") or "").strip().lower() == "waiting_user":
        return "respond_node"
    return "execute_step_node"


def execute_step_node(state: dict[str, Any], *, tool_registry: Any) -> dict[str, Any]:
    emit_transition_event(state, _TRANSITION_EVENT_EXECUTING)
    return execute_step_node_impl(
        state,
        tool_registry=tool_registry,
        task_state_with_defaults=task_state_with_defaults,
        correlation_id=correlation_id,
        current_step=current_step,
        append_trace_event=append_trace_event,
        logger=logger,
        log_task_event=log_task_event,
    )


def check_node(state: dict[str, Any], *, tool_registry: Any) -> dict[str, Any]:
    _ = tool_registry
    task_state = task_state_with_defaults(state)
    corr = correlation_id(state)
    cycle = int(task_state.get("cycle_index") or 0)
    task_record = _build_task_record_from_state(state=state, task_state=task_state)
    result = check_node_impl(
        task_record,
        llm_client=state.get("_llm_client"),
        logger=logger,
        log_task_event=log_task_event,
    )
    return _finalize_native_check_result(
        state=state,
        task_state=task_state,
        task_record=result.get("task_record") if isinstance(result.get("task_record"), TaskRecord) else task_record,
        verdict=str(result.get("verdict") or "").strip().lower() or "plan",
        judge_result=result.get("judge_result") if isinstance(result.get("judge_result"), dict) else {},
        consumed_inputs=result.get("consumed_inputs") if isinstance(result.get("consumed_inputs"), list) else [],
        cycle=cycle,
        corr=corr,
    )


def route_after_check(state: dict[str, Any]) -> str:
    check_result = state.get("check_result") if isinstance(state.get("check_result"), dict) else {}
    verdict = str(check_result.get("verdict") or "").strip().lower()
    if verdict == "plan":
        return "next_step_node"
    if verdict in {"mission_success", "mission_failed"}:
        return "respond_node"
    task_record = state.get("task_record")
    status = str(task_record.status or "").strip().lower() if isinstance(task_record, TaskRecord) else ""
    return "respond_node" if status in {"waiting_user", "failed", "done"} else "next_step_node"


def update_state_node(state: dict[str, Any]) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_record = state.get("task_record") if isinstance(state.get("task_record"), TaskRecord) else None
    task_state["pdca_phase"] = "check"
    corr = correlation_id(state)
    task_state["cycle_index"] = int(task_state.get("cycle_index") or 0) + 1
    outcome = task_record.outcome if isinstance(task_record, TaskRecord) and isinstance(task_record.outcome, dict) else None
    task_state["outcome"] = outcome
    if isinstance(task_record, TaskRecord):
        task_state["status"] = str(task_record.status or "").strip() or task_state.get("status") or "running"
    trace = task_trace(task_state)
    trace["summary"] = f"PDCA cycle {task_state['cycle_index']} complete."
    append_trace_event(
        task_state,
        {
            "type": "state_updated",
            "summary": f"State updated at cycle {task_state['cycle_index']}.",
            "correlation_id": corr,
        },
    )
    logger.info(
        "task_mode update_state correlation_id=%s cycle=%s status=%s has_outcome=%s",
        corr,
        int(task_state.get("cycle_index") or 0),
        str(task_state.get("status") or ""),
        bool(outcome),
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="update_state_node",
        event="graph.state.updated",
        has_outcome=bool(outcome),
    )
    return {"task_state": task_state}


def act_node(state: dict[str, Any]) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "act"
    return {"task_state": task_state}


def route_after_act(state: dict[str, Any]) -> str:
    check_result = state.get("check_result") if isinstance(state.get("check_result"), dict) else {}
    verdict_kind = str(check_result.get("verdict") or "").strip().lower()
    if verdict_kind == "plan":
        logger.info(
            "task_mode route_after_act correlation_id=%s route=next_step_node verdict=%s",
            correlation_id(state),
            verdict_kind,
        )
        return "next_step_node"
    if verdict_kind in {"mission_success", "mission_failed"}:
        logger.info(
            "task_mode route_after_act correlation_id=%s route=respond_node verdict=%s",
            correlation_id(state),
            verdict_kind,
        )
        return "respond_node"
    task_record = state.get("task_record") if isinstance(state.get("task_record"), TaskRecord) else None
    status = str(task_record.status or "").strip().lower() if isinstance(task_record, TaskRecord) else ""
    if status in {"done", "failed", "waiting_user"}:
        logger.info(
            "task_mode route_after_act correlation_id=%s route=respond_node status=%s",
            correlation_id(state),
            status,
        )
        return "respond_node"
    logger.info(
        "task_mode route_after_act correlation_id=%s route=next_step_node missing_check_route=true",
        correlation_id(state),
    )
    return "next_step_node"


def _select_check_case_type(task_state: dict[str, Any]) -> str | None:
    provenance = str(
        (task_metrics(task_state).get("check_provenance") or task_state.get("check_provenance") or "")
    ).strip().lower()
    mapping = {
        "entry": "new_request",
        "do": "execution_review",
        "slice_resume": "task_resumption",
    }
    return mapping.get(provenance)


def _build_task_record_from_state(*, state: dict[str, Any], task_state: dict[str, Any]) -> TaskRecord:
    record = TaskRecord(
        task_id=_first_non_empty(task_state.get("task_id"), state.get("task_id")),
        user_id=_first_non_empty(task_state.get("user_id"), state.get("actor_person_id")),
        check_case_type=_select_check_case_type(task_state) or "",
        goal=str(task_state.get("goal") or "").strip(),
        status=str(task_state.get("status") or "").strip() or "running",
        outcome=task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None,
    )
    record.set_recent_conversation_md(_resolve_recent_conversation_md(state))
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


def _finalize_native_check_result(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    task_record: TaskRecord,
    verdict: str,
    judge_result: dict[str, Any],
    consumed_inputs: list[dict[str, Any]],
    cycle: int,
    corr: str | None,
) -> dict[str, Any]:
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

    task_state["acceptance_criteria"] = _apply_criteria_updates(
        existing=[] if consumed_inputs else normalize_acceptance_criteria_records(task_state.get("acceptance_criteria")),
        updates=judge_result.get("criteria_updates"),
        case_type=str(judge_result.get("case_type") or "").strip().lower() or "execution_review",
        fallback_user_text=str(task_record.goal or "").strip(),
    )

    reason = str(judge_result.get("reason") or "").strip()
    check_result = {
        "verdict": verdict,
        "judge_result": dict(judge_result),
    }
    task_state["judge_verdict"] = dict(judge_result)
    task_state["check_result"] = dict(check_result)
    if verdict == "mission_success":
        task_record.status = "done"
        task_record.outcome = {
            "kind": "task_completed",
            "summary": reason or "Mission completed successfully.",
            "final_text": reason or "Mission completed successfully.",
            "evidence": {
                "criteria": normalize_acceptance_criteria_records(task_state.get("acceptance_criteria")),
                "verdict_confidence": float(judge_result.get("confidence") or 0.0),
            },
        }
        task_state["status"] = task_record.status
        task_state["outcome"] = dict(task_record.outcome)
    elif verdict == "mission_failed":
        failure_code = str(judge_result.get("failure_class") or "mission_failed").strip() or "mission_failed"
        retry_exhausted = bool(judge_result.get("retry_exhausted"))
        task_record.status = "failed"
        task_record.outcome = {
            "kind": "task_failed",
            "summary": reason or "Mission failed.",
            "failure_class": failure_code,
        }
        task_state["status"] = task_record.status
        task_state["outcome"] = dict(task_record.outcome)
        task_state["last_validation_error"] = {
            "reason": failure_code,
            "message": reason or "Mission failed.",
            "retry_exhausted": retry_exhausted,
        }
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="pdca.failure.classified",
            failure_code=failure_code,
            failure_reason=reason or "Mission failed.",
            retry_exhausted=retry_exhausted,
            cycle=cycle,
        )
    else:
        task_record.status = "running"
        task_record.outcome = None
        task_state["status"] = task_record.status
        task_state["outcome"] = None
        task_state["next_user_question"] = None
        emit_transition_event(state, "thinking")
    check_route = post_check_route(verdict)
    append_trace_event(
        task_state,
        {
            "type": "judge_verdict",
            "summary": f"Check verdict={verdict or 'plan'} case={str(judge_result.get('case_type') or '')}",
            "correlation_id": corr,
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
        correlation_id=corr,
        cycle=cycle,
    )
    return {"task_state": task_state, "task_record": task_record, "check_result": check_result}


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
        if rendered.startswith("user_reply_"):
            suffix = rendered.split("user_reply_", 1)[1]
            try:
                max_suffix = max(max_suffix, int(suffix))
            except ValueError:
                pass
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


def _apply_criteria_updates(
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
                {"id": criterion_id, "text": text[:180], "status": "pending", "evidence_refs": [], "created_by_case": case_type}
            )
            index[criterion_id] = len(out) - 1
        elif op == "mark_satisfied":
            criterion_id = str(update.get("criterion_id") or "").strip()
            if criterion_id and criterion_id in index:
                pos = index[criterion_id]
                refs = update.get("evidence_refs")
                out[pos]["status"] = "satisfied"
                out[pos]["evidence_refs"] = [str(ref).strip() for ref in refs if str(ref).strip()][:8] if isinstance(refs, list) else []
    if case_type == "new_request" and not out:
        snippet = fallback_user_text.strip()[:120] or "the user request"
        out.append({"id": "ac_1", "text": f"Advance the request successfully: {snippet}", "status": "pending", "evidence_refs": [], "created_by_case": "new_request"})
    return normalize_acceptance_criteria_records(out)

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
            if isinstance(entry, dict) and not bool(entry.get("internal")):
                rendered = str(entry.get("step_id") or "").strip()
                if rendered:
                    fact_refs.append(f"fact:{rendered[:80]}")
    payload = {
        "case_type": str(judge_result.get("case_type") or "").strip()[:40],
        "cycle": int(cycle),
        "acceptance_criteria": [
            {"id": str(item.get("id") or "").strip()[:40], "text": str(item.get("text") or "").strip()[:180], "status": str(item.get("status") or "pending").strip()[:24]}
            for item in criteria[:12]
        ],
        "fact_refs": fact_refs[:12],
        "verdict": {"kind": str(judge_result.get("kind") or "").strip()[:32], "confidence": float(judge_result.get("confidence") or 0.0)},
    }
    reason = str(judge_result.get("reason") or "").strip()
    if reason:
        payload["verdict"]["reason"] = reason[:220]
    try:
        append_pdca_event(task_id=task_id, event_type="check.criteria_snapshot", payload=payload, correlation_id=correlation_id)
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


def _compact_json(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)[:500]
    except Exception:
        return str(value)[:500]


def _append_fact_lines(record: TaskRecord, *, facts: Any, channel_type: str | None, channel_target: str | None, locale: str | None, timezone: str | None, message_id: str | None, conversation_key: str | None) -> None:
    facts = facts if isinstance(facts, dict) else {}
    for key, value in list(facts.items())[:20]:
        record.append_fact(f"{str(key)}: {_compact_json(value)}")
    for key, value in (("channel_type", channel_type), ("channel_target", channel_target), ("locale", locale), ("timezone", timezone), ("message_id", message_id), ("conversation_key", conversation_key)):
        if value:
            record.append_fact(f"{key}: {_compact_json(value)}")


def _append_plan_lines(record: TaskRecord, *, plan: Any) -> None:
    plan = plan if isinstance(plan, dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    for step in steps[-10:]:
        if isinstance(step, dict):
            proposal = step.get("proposal") if isinstance(step.get("proposal"), dict) else {}
            tool_name = str(proposal.get("tool_name") or "").strip() or "(none)"
            args = proposal.get("args") if isinstance(proposal.get("args"), dict) else {}
            record.append_plan_line(f"{str(step.get('step_id') or '').strip() or '(unknown)'} [{str(step.get('status') or '').strip() or 'unknown'}] {tool_name} args={_compact_json(args)}")


def _append_memory_fact_lines(record: TaskRecord, *, memory_facts: Any) -> None:
    if isinstance(memory_facts, list):
        for entry in memory_facts[-8:]:
            if isinstance(entry, dict):
                record.append_memory_fact(f"{str(entry.get('tool_name') or '').strip() or 'memory'} output={_compact_json(entry.get('output'))} exception={_compact_json(entry.get('exception'))}")


def _append_tool_call_history_lines(record: TaskRecord, *, tool_call_history: Any) -> None:
    if isinstance(tool_call_history, list):
        for entry in tool_call_history[-12:]:
            if isinstance(entry, dict):
                record.append_tool_call_history_entry(
                    f"{str(entry.get('step_id') or '').strip() or '(no-step)'} {str(entry.get('tool_name') or entry.get('tool') or '').strip() or '(unknown)'} args={_compact_json(entry.get('params') if isinstance(entry.get('params'), dict) else entry.get('args'))} output={_compact_json(entry.get('output'))} exception={_compact_json(entry.get('exception'))}"
                )


def _next_criterion_id(existing: list[dict[str, Any]]) -> str:
    max_num = 0
    for item in existing:
        criterion_id = str(item.get("id") or "").strip().lower()
        if criterion_id.startswith("ac_") and criterion_id[3:].isdigit():
            max_num = max(max_num, int(criterion_id[3:]))
    return f"ac_{max_num + 1}"


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        rendered = str(value or "").strip()
        if rendered:
            return rendered
    return None
