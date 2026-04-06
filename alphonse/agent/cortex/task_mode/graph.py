from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END
from langgraph.graph import StateGraph

from alphonse.agent.cortex.task_mode.observability import log_task_event
from alphonse.agent.cortex.task_mode.pdca import act_node
from alphonse.agent.cortex.task_mode.pdca import build_next_step_node
from alphonse.agent.cortex.task_mode.pdca import check_node
from alphonse.agent.cortex.task_mode.pdca import execute_step_node
from alphonse.agent.cortex.task_mode.pdca import route_after_act
from alphonse.agent.cortex.task_mode.pdca import route_after_next_step
from alphonse.agent.cortex.task_mode.pdca import update_state_node
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.cortex.task_mode.task_state_helpers import append_trace_event
from alphonse.agent.cortex.task_mode.task_state_helpers import correlation_id
from alphonse.agent.cortex.task_mode.task_state_helpers import normalize_acceptance_criteria_records
from alphonse.agent.cortex.task_mode.task_state_helpers import task_metrics
from alphonse.agent.cortex.task_mode.task_state_helpers import task_state_with_defaults
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.nervous_system.pdca_queue_store import append_pdca_event
from alphonse.agent.observability.log_manager import get_component_logger
from alphonse.agent.session.day_state import render_recent_conversation_block

logger = get_component_logger("task_mode.graph")


def wire_task_mode_pdca(graph: StateGraph, *, tool_registry: Any) -> None:
    """PDCA loop: hydrate TaskRecord -> Check -> Act -> Plan -> Do -> Update -> Check."""
    graph.add_node("task_record_entry_node", task_record_entry_node)
    graph.add_node(
        "next_step_node",
        build_next_step_node(tool_registry=tool_registry),
    )
    graph.add_node(
        "execute_step_node",
        lambda state: execute_step_node(state, tool_registry=tool_registry),
    )
    graph.add_node("check_node", check_node_state_adapter)
    graph.add_node("update_state_node", update_state_node)
    graph.add_node("act_node", act_node)

    graph.add_edge("task_record_entry_node", "check_node")
    graph.add_edge("check_node", "act_node")
    graph.add_conditional_edges(
        "act_node",
        route_after_act,
        {
            "next_step_node": "next_step_node",
            "respond_node": "respond_node",
            "end": END,
        },
    )
    graph.add_conditional_edges(
        "next_step_node",
        route_after_next_step,
        {
            "execute_step_node": "execute_step_node",
            "check_node": "check_node",
            "respond_node": "respond_node",
        },
    )
    graph.add_edge("execute_step_node", "update_state_node")
    graph.add_edge("update_state_node", "check_node")


def task_record_entry_node(state: dict[str, Any]) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_record = _hydrate_task_record_from_state(state=state, task_state=task_state)
    provenance = "slice_resume" if _has_existing_task_context(state=state, task_state=task_state) else "entry"
    task_metrics(task_state)["check_provenance"] = provenance
    task_state["goal"] = str(task_record.goal or "").strip()
    task_state["task_id"] = task_record.task_id
    task_state["user_id"] = task_record.user_id
    return {
        "task_state": task_state,
        "task_record": task_record,
        "recent_conversation_block": task_record.recent_conversation_md,
    }


def check_node_state_adapter(state: dict[str, Any]) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_record = _hydrate_task_record_from_state(state=state, task_state=task_state)
    provenance = _select_check_provenance(state=state, task_state=task_state)
    cycle = int(task_state.get("cycle_index") or 0)
    corr = task_record.correlation_id or correlation_id(state)
    result = check_node(task_record, provenance=provenance)
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


def _has_existing_task_context(*, state: dict[str, Any], task_state: dict[str, Any]) -> bool:
    if isinstance(state.get("task_record"), TaskRecord):
        return True
    task_id = str(task_state.get("task_id") or state.get("task_id") or "").strip()
    if task_id:
        return True
    return bool(task_state.get("plan") or task_state.get("facts") or task_state.get("tool_call_history"))


def _select_check_provenance(*, state: dict[str, Any], task_state: dict[str, Any]) -> str:
    provenance = str(
        task_metrics(task_state).get("check_provenance")
        or task_state.get("check_provenance")
        or state.get("check_provenance")
        or ""
    ).strip().lower()
    if provenance in {"entry", "do", "slice_resume"}:
        return provenance
    if provenance:
        return provenance
    return "slice_resume" if _has_existing_task_context(state=state, task_state=task_state) else "entry"


def _hydrate_task_record_from_state(*, state: dict[str, Any], task_state: dict[str, Any]) -> TaskRecord:
    existing = state.get("task_record")
    if isinstance(existing, TaskRecord):
        record = existing
    else:
        record = TaskRecord()
    record.task_id = _first_non_empty(record.task_id, task_state.get("task_id"), state.get("task_id"))
    record.user_id = _first_non_empty(record.user_id, task_state.get("user_id"), state.get("actor_person_id"))
    record.set_correlation_id(
        _first_non_empty(record.correlation_id, state.get("correlation_id")) or ""
    )
    record.goal = str(record.goal or task_state.get("goal") or "").strip()
    if not record.goal:
        record.goal = _resolve_goal_text(state)
    record.status = str(record.status or task_state.get("status") or "").strip() or "running"
    record.outcome = (
        dict(record.outcome)
        if isinstance(record.outcome, dict)
        else task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None
    )
    record.set_recent_conversation_md(_resolve_recent_conversation_md(state=state, task_record=record))
    _append_fact_lines(
        record,
        facts=task_state.get("facts"),
        channel_type=_first_non_empty(state.get("channel_type")),
        channel_target=_first_non_empty(state.get("channel_target")),
        locale=_first_non_empty(state.get("locale")),
        timezone=_first_non_empty(state.get("timezone")),
        message_id=_first_non_empty(task_state.get("message_id"), state.get("message_id")),
        conversation_key=_first_non_empty(task_state.get("conversation_key"), state.get("conversation_key")),
        actor_person_id=_first_non_empty(state.get("actor_person_id")),
    )
    _append_plan_lines(record, plan=task_state.get("plan"))
    _append_memory_fact_lines(record, memory_facts=task_state.get("memory_facts"))
    _append_tool_call_history_lines(record, tool_call_history=task_state.get("tool_call_history"))
    _sync_acceptance_criteria_md(record=record, criteria=task_state.get("acceptance_criteria"))
    return record


def _resolve_recent_conversation_md(*, state: dict[str, Any], task_record: TaskRecord) -> str:
    existing = str(task_record.recent_conversation_md or "").strip()
    if existing and existing != "- (none)":
        return existing
    recent = str(state.get("recent_conversation_block") or "").strip()
    if recent:
        return recent
    session_state = state.get("session_state") if isinstance(state.get("session_state"), dict) else None
    if session_state:
        rendered = render_recent_conversation_block(session_state)
        if rendered:
            return rendered
    last_user_message = str(state.get("last_user_message") or "").strip()
    if last_user_message:
        return f"- User: {last_user_message}"
    return "- (none)"


def _resolve_goal_text(state: dict[str, Any]) -> str:
    incoming = state.get("incoming_raw_message")
    if isinstance(incoming, dict):
        extracted = _extract_goal_from_payload(incoming)
        if extracted:
            return extracted
    return _extract_goal_from_packed_message(str(state.get("last_user_message") or ""))


def _extract_goal_from_payload(payload: dict[str, Any]) -> str:
    direct = str(payload.get("text") or "").strip()
    if direct:
        return direct
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    nested = str(message.get("text") or "").strip()
    if nested:
        return nested
    provider_event = payload.get("provider_event") if isinstance(payload.get("provider_event"), dict) else {}
    provider_message = provider_event.get("message") if isinstance(provider_event.get("message"), dict) else {}
    provider_text = str(provider_message.get("text") or "").strip()
    if provider_text:
        return provider_text
    return ""


def _extract_goal_from_packed_message(last_user_message: str) -> str:
    rendered = str(last_user_message or "").strip()
    if not rendered:
        return ""
    marker = "```json"
    if marker in rendered:
        after = rendered.split(marker, 1)[1]
        json_payload = after.split("```", 1)[0].strip()
        try:
            parsed = json.loads(json_payload)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            extracted = _extract_goal_from_payload(parsed)
            if extracted:
                return extracted
    for line in rendered.splitlines():
        if line.lower().startswith("- text:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate:
                return candidate
    return rendered[:240]


def _sync_acceptance_criteria_md(*, record: TaskRecord, criteria: Any) -> None:
    record.clear_acceptance_criteria()
    for item in normalize_acceptance_criteria_records(criteria):
        status = str(item.get("status") or "pending").strip().lower()
        prefix = "[x]" if status == "satisfied" else "[ ]"
        record.append_acceptance_criterion(f"{prefix} {str(item.get('text') or '').strip()}")


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
        )

    task_state["judge_verdict"] = dict(judge_result)
    task_state["check_result"] = {"verdict": verdict, "judge_result": dict(judge_result)}
    task_state["status"] = str(task_record.status or "").strip() or "running"
    task_state["outcome"] = dict(task_record.outcome) if isinstance(task_record.outcome, dict) else None
    task_state["goal"] = str(task_record.goal or "").strip()
    task_state["next_user_question"] = None
    task_state["acceptance_criteria"] = _render_acceptance_criteria_records(task_record)

    if verdict == "mission_failed":
        failure_code = str(judge_result.get("failure_class") or "mission_failed").strip() or "mission_failed"
        task_state["last_validation_error"] = {
            "reason": failure_code,
            "message": str(judge_result.get("reason") or "Mission failed.").strip() or "Mission failed.",
            "retry_exhausted": bool(judge_result.get("retry_exhausted")),
        }
    else:
        task_state["last_validation_error"] = None

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
        route="next_step_node" if verdict == "plan" else "respond_node",
        cycle=cycle,
    )
    _append_check_criteria_snapshot_event(
        state=state,
        task_state=task_state,
        judge_result=judge_result,
        correlation_id=corr,
        cycle=cycle,
    )
    if verdict == "plan":
        emit_transition_event(state, "thinking")
    return {
        "task_state": task_state,
        "task_record": task_record,
        "check_result": {"verdict": verdict, "judge_result": dict(judge_result)},
        "recent_conversation_block": task_record.recent_conversation_md,
    }


def _render_acceptance_criteria_records(task_record: TaskRecord) -> list[dict[str, Any]]:
    lines = [line.strip() for line in str(task_record.get_acceptance_criteria_md() or "").splitlines() if line.strip()]
    if not lines or lines == ["- (none)"]:
        return []
    out: list[dict[str, Any]] = []
    for index, line in enumerate(lines, start=1):
        text = line.removeprefix("- ").strip()
        status = "pending"
        if text.startswith("[x] "):
            status = "satisfied"
            text = text[4:].strip()
        elif text.startswith("[ ] "):
            text = text[4:].strip()
        if not text:
            continue
        out.append(
            {
                "id": f"ac_{index}",
                "text": text[:180],
                "status": status,
                "evidence_refs": [],
                "created_by_case": "new_request",
            }
        )
    return out


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
        "verdict": {
            "kind": str(judge_result.get("kind") or "").strip()[:32],
            "confidence": float(judge_result.get("confidence") or 0.0),
        },
        "fact_refs": _fact_refs_from_tool_history(task_state.get("tool_call_history")),
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


def _fact_refs_from_tool_history(tool_call_history: Any) -> list[str]:
    if not isinstance(tool_call_history, list):
        return []
    out: list[str] = []
    for entry in tool_call_history[-12:]:
        if not isinstance(entry, dict) or bool(entry.get("internal")):
            continue
        rendered = str(entry.get("step_id") or "").strip()
        if rendered:
            out.append(f"fact:{rendered[:80]}")
    return out[:12]


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
    actor_person_id: str | None,
) -> None:
    fact_map = facts if isinstance(facts, dict) else {}
    for key, value in list(fact_map.items())[:20]:
        record.append_fact(f"{str(key)}: {_compact_json(value)}")
    for key, value in (
        ("channel_type", channel_type),
        ("channel_target", channel_target),
        ("locale", locale),
        ("timezone", timezone),
        ("message_id", message_id),
        ("conversation_key", conversation_key),
        ("actor_person_id", actor_person_id),
    ):
        if value:
            record.append_fact(f"{key}: {_compact_json(value)}")


def _append_plan_lines(record: TaskRecord, *, plan: Any) -> None:
    plan_dict = plan if isinstance(plan, dict) else {}
    steps = plan_dict.get("steps") if isinstance(plan_dict.get("steps"), list) else []
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
    if not isinstance(memory_facts, list):
        return
    for entry in memory_facts[-8:]:
        if isinstance(entry, dict):
            record.append_memory_fact(
                f"{str(entry.get('tool_name') or '').strip() or 'memory'} "
                f"output={_compact_json(entry.get('output'))} exception={_compact_json(entry.get('exception'))}"
            )


def _append_tool_call_history_lines(record: TaskRecord, *, tool_call_history: Any) -> None:
    if not isinstance(tool_call_history, list):
        return
    for entry in tool_call_history[-12:]:
        if isinstance(entry, dict):
            record.append_tool_call_history_entry(
                f"{str(entry.get('step_id') or '').strip() or '(no-step)'} "
                f"{str(entry.get('tool_name') or entry.get('tool') or '').strip() or '(unknown)'} "
                f"args={_compact_json(entry.get('params') if isinstance(entry.get('params'), dict) else entry.get('args'))} "
                f"output={_compact_json(entry.get('output'))} exception={_compact_json(entry.get('exception'))}"
            )


def _compact_json(value: Any) -> str:
    if value is None:
        return "null"
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)[:500]
    except Exception:
        return str(value)[:500]


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        rendered = str(value or "").strip()
        if rendered:
            return rendered
    return None
