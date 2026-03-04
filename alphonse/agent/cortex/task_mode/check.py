from __future__ import annotations

import os
import json
import hashlib
from typing import Any

from alphonse.agent.cognition.tool_schemas import canonical_tool_names
from alphonse.agent.cortex.task_mode.act_node import evaluate_tool_execution
from alphonse.agent.cortex.task_mode.check_decider import decide_check_action
from alphonse.agent.cortex.task_mode.check_state import evaluate_success_from_evidence
from alphonse.agent.cortex.task_mode.progress_critic_node import build_wip_update_detail
from alphonse.agent.cortex.task_mode.task_state_helpers import append_trace_event
from alphonse.agent.cortex.task_mode.task_state_helpers import correlation_id
from alphonse.agent.cortex.task_mode.task_state_helpers import current_step
from alphonse.agent.cortex.task_mode.task_state_helpers import has_acceptance_criteria
from alphonse.agent.cortex.task_mode.task_state_helpers import normalize_acceptance_criteria_values
from alphonse.agent.cortex.task_mode.task_state_helpers import task_plan
from alphonse.agent.cortex.task_mode.task_state_helpers import task_state_with_defaults
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.session.day_state import render_recent_conversation_block
from alphonse.config import settings

_PLANNER_RETRY_BUDGET_DEFAULT = 2
_ZERO_PROGRESS_BUDGET_DEFAULT = 2


def check_node_impl(
    state: dict[str, Any],
    *,
    tool_registry: Any,
    logger: Any,
    log_task_event: Any,
    wip_emit_every_cycles: int,
) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "check"
    corr = correlation_id(state)
    cycle = int(task_state.get("cycle_index") or 0)

    status = str(task_state.get("status") or "").strip().lower()
    if cycle == 0 and status not in {"done", "failed", "waiting_user"}:
        is_continuation = _is_continuation(task_state)
        has_acceptance = has_acceptance_criteria(task_state)
        llm_client = state.get("_llm_client")
        text = str(state.get("last_user_message") or "").strip()
        recent = str(state.get("recent_conversation_block") or "").strip()
        if not recent:
            session_state = state.get("session_state") if isinstance(state.get("session_state"), dict) else None
            if session_state:
                recent = render_recent_conversation_block(session_state)
        decision = decide_check_action(
            text=text,
            llm_client=llm_client,
            locale=state.get("locale"),
            tone=state.get("tone"),
            address_style=state.get("address_style"),
            channel_type=state.get("channel_type"),
            available_tool_names=canonical_tool_names(tool_registry),
            recent_conversation_block=recent,
            goal=str(task_state.get("goal") or "").strip(),
            status=status or "running",
            cycle_index=cycle,
            is_continuation=is_continuation,
            has_acceptance=has_acceptance,
            facts=task_state.get("facts") if isinstance(task_state.get("facts"), dict) else None,
            plan=task_state.get("plan") if isinstance(task_state.get("plan"), dict) else None,
        )
        task_state["check_decision_last"] = {
            "route": str(decision.get("route") or ""),
            "intent": str(decision.get("intent") or ""),
            "confidence": float(decision.get("confidence") or 0.0),
            "parse_ok": bool(decision.get("parse_ok", False)),
            "retried": bool(decision.get("retried", False)),
        }
        if bool(decision.get("retried", False)):
            log_task_event(
                logger=logger,
                state=state,
                task_state=task_state,
                node="check_node",
                event="graph.check.decision_retry",
                level="warning",
            )
        if bool(decision.get("invalid_json_fallback", False)):
            log_task_event(
                logger=logger,
                state=state,
                task_state=task_state,
                node="check_node",
                event="graph.check.decision_clarify_on_invalid_json",
                level="warning",
            )

        route = str(decision.get("route") or "").strip().lower()
        criteria_created = False
        if route == "tool_plan" and not is_continuation and not has_acceptance:
            criteria = normalize_acceptance_criteria_values(decision.get("acceptance_criteria"))
            if criteria:
                task_state["acceptance_criteria"] = criteria
                criteria_created = True
                append_trace_event(
                    task_state,
                    {
                        "type": "acceptance_criteria_derived",
                        "summary": "Check created acceptance criteria from decision payload.",
                        "correlation_id": corr,
                    },
                )
            else:
                task_state["status"] = "waiting_user"
                task_state["next_user_question"] = (
                    str(decision.get("clarify_question") or "").strip()
                    or "Before I proceed, what acceptance criteria should define completion?"
                )
                route = "clarify"
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="graph.check.decision",
            route=route,
            parse_ok=bool(decision.get("parse_ok", False)),
            is_continuation=is_continuation,
            criteria_created=criteria_created,
            criteria_count=len(normalize_acceptance_criteria_values(task_state.get("acceptance_criteria"))),
        )
        if route == "direct_reply":
            reply_text = str(decision.get("reply_text") or "").strip()
            if reply_text:
                task_state["status"] = ""
                task_state["check_route"] = "respond_node"
                task_state["surface_planner_intent"] = False
                return {"task_state": task_state, "response_text": reply_text}
            task_state["status"] = "waiting_user"
            task_state["next_user_question"] = (
                str(decision.get("clarify_question") or "").strip()
                or "Could you clarify what you need?"
            )
        elif route == "clarify":
            task_state["status"] = "waiting_user"
            task_state["next_user_question"] = (
                str(decision.get("clarify_question") or "").strip()
                or "Could you clarify what you need?"
            )
        elif route == "tool_plan":
            task_state["status"] = "running"
            task_state["next_user_question"] = None
            task_state["surface_planner_intent"] = _should_surface_planner_intent(
                upcoming_cycle=cycle + 1
            )
            emit_transition_event(state, "thinking")

    planner_error = task_state.get("planner_error_last") if isinstance(task_state.get("planner_error_last"), dict) else None
    if planner_error:
        streak = int(task_state.get("planner_error_streak") or 0) + 1
        task_state["planner_error_streak"] = streak
        budget = _planner_retry_budget()
        if streak <= budget:
            task_state["status"] = "running"
            task_state["next_user_question"] = None
            task_state["last_validation_error"] = None
            append_trace_event(
                task_state,
                {
                    "type": "planner_retry",
                    "summary": f"Planner output invalid; retrying planning ({streak}/{budget}).",
                    "correlation_id": corr,
                },
            )
            log_task_event(
                logger=logger,
                state=state,
                task_state=task_state,
                node="check_node",
                event="graph.check.planner_retry",
                planner_error_streak=streak,
                planner_retry_budget=budget,
                error_code=str(planner_error.get("code") or ""),
                level="warning",
            )
            task_state["check_route"] = "next_step_node"
            task_state["surface_planner_intent"] = _should_surface_planner_intent(
                upcoming_cycle=cycle + 1,
                planner_error=True,
            )
            return {"task_state": task_state}

        task_state["status"] = "failed"
        task_state["next_user_question"] = None
        task_state["last_validation_error"] = dict(planner_error)
        append_trace_event(
            task_state,
            {
                "type": "status_changed",
                "summary": "Check marked task failed after planner retry budget exhausted.",
                "correlation_id": corr,
            },
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="graph.check.planner_failed",
            planner_error_streak=streak,
            planner_retry_budget=budget,
            error_code=str(planner_error.get("code") or ""),
            level="warning",
        )
    else:
        if int(task_state.get("planner_error_streak") or 0) != 0:
            task_state["planner_error_streak"] = 0
        if task_state.get("planner_error_last") is not None:
            task_state["planner_error_last"] = None

    current = current_step(task_state)
    current_status = str((current or {}).get("status") or "").strip().lower()
    if current_status == "failed":
        evaluation = evaluate_tool_execution(task_state=task_state, current_step=current, task_plan=task_plan)
        task_state["execution_eval"] = evaluation if isinstance(evaluation, dict) else {}
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="graph.check.failure_evaluated",
            step_id=str((current or {}).get("step_id") or ""),
            reason=str((evaluation or {}).get("reason") or ""),
            total_failures=int((evaluation or {}).get("total_failures") or 0),
            same_signature_failures=int((evaluation or {}).get("same_signature_failures") or 0),
            unique_signatures=int((evaluation or {}).get("unique_signatures") or 0),
            level="warning",
        )

    if int(task_state.get("cycle_index") or 0) > 0:
        emit_every = _wip_emit_every_for_mode(default_every=wip_emit_every_cycles)
        step_status = str((current or {}).get("status") or "").strip().lower()
        if step_status == "proposed" and int(task_state.get("cycle_index") or 0) % emit_every == 0:
            detail = build_wip_update_detail(
                task_state=task_state,
                cycle=int(task_state.get("cycle_index") or 0),
                current_step=current,
            )
            emit_transition_event(state, "wip_update", detail)
            log_task_event(
                logger=logger,
                state=state,
                task_state=task_state,
                node="check_node",
                event="graph.check.wip_update",
                cycle=int(detail.get("cycle") or 0),
                tool=str(detail.get("tool") or ""),
                intention=str(detail.get("intention") or ""),
            )

    evaluation = evaluate_success_from_evidence(task_state=task_state)
    task_state["success_evaluation_last"] = dict(evaluation)
    completion_decision = {
        "is_done": bool(evaluation.get("is_done", False)),
        "reason": str(evaluation.get("reason") or ""),
        "supporting_facts": list(evaluation.get("supporting_facts") or []),
        "confidence": float(evaluation.get("confidence") or 0.0),
        "missing_evidence": list(evaluation.get("missing_evidence") or []),
    }
    task_state["completion_decision"] = completion_decision
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="check_node",
        event="graph.check.success_evaluated",
        is_done=bool(evaluation.get("is_done", False)),
        reason=str(evaluation.get("reason") or ""),
        confidence=float(evaluation.get("confidence") or 0.0),
        missing_evidence_count=len(list(evaluation.get("missing_evidence") or [])),
    )
    if bool(evaluation.get("is_done", False)) and has_acceptance_criteria(task_state):
        if str(task_state.get("status") or "").strip().lower() != "done":
            append_trace_event(
                task_state,
                {
                    "type": "status_changed",
                    "summary": "Check marked task as done from evidence adjudication.",
                    "correlation_id": corr,
                },
            )
        task_state["status"] = "done"
        final_hint = str(evaluation.get("final_response_hint") or "").strip()
        outcome_kind = str(evaluation.get("outcome_kind") or "").strip() or "task_completed"
        task_state["outcome"] = {
            "kind": outcome_kind,
            "summary": str(evaluation.get("reason") or "criteria_satisfied"),
            "final_text": final_hint,
            "evidence": {
                "supporting_facts": list(evaluation.get("supporting_facts") or []),
                "confidence": float(evaluation.get("confidence") or 0.0),
            },
        }
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="graph.task.completed",
        )
    else:
        _update_zero_progress_state(
            state=state,
            task_state=task_state,
            logger=logger,
            log_task_event=log_task_event,
        )

    status = str(task_state.get("status") or "").strip().lower()
    if status in {"done", "waiting_user", "failed"}:
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="graph.check.skipped",
            status=status,
            cycle=cycle,
            level="debug",
        )
    else:
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="check_node",
            event="graph.check.continue",
            cycle=cycle,
            level="debug",
        )
    check_route = "respond_node" if status in {"waiting_user", "failed", "done"} else "next_step_node"
    if check_route == "next_step_node":
        task_state["surface_planner_intent"] = _should_surface_planner_intent(
            upcoming_cycle=cycle + 1
        )
    else:
        task_state["surface_planner_intent"] = False
    task_state["check_route"] = check_route
    logger.info(
        "task_mode check correlation_id=%s cycle=%s status=%s route=%s",
        corr,
        cycle,
        status or "running",
        check_route,
    )
    return {"task_state": task_state}


def route_after_check_impl(state: dict[str, Any]) -> str:
    task_state = state.get("task_state") if isinstance(state.get("task_state"), dict) else {}
    route = str(task_state.get("check_route") or "").strip().lower()
    if route in {"respond_node", "next_step_node"}:
        return route
    status = str(task_state.get("status") or "").strip().lower()
    return "respond_node" if status in {"waiting_user", "failed", "done"} else "next_step_node"


def _planner_retry_budget() -> int:
    raw = str(os.getenv("ALPHONSE_TASK_MODE_PLANNER_RETRY_BUDGET") or "").strip()
    if not raw:
        return _PLANNER_RETRY_BUDGET_DEFAULT
    try:
        parsed = int(raw)
    except ValueError:
        return _PLANNER_RETRY_BUDGET_DEFAULT
    return parsed if parsed >= 0 else 0


def _zero_progress_budget() -> int:
    raw = str(os.getenv("ALPHONSE_TASK_MODE_ZERO_PROGRESS_BUDGET") or "").strip()
    if not raw:
        return _ZERO_PROGRESS_BUDGET_DEFAULT
    try:
        parsed = int(raw)
    except ValueError:
        return _ZERO_PROGRESS_BUDGET_DEFAULT
    return parsed if parsed >= 1 else 1


def _is_continuation(task_state: dict[str, Any]) -> bool:
    if int(task_state.get("cycle_index") or 0) > 0:
        return True
    plan = task_state.get("plan") if isinstance(task_state.get("plan"), dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    current_id = str(plan.get("current_step_id") or "").strip()
    if steps or current_id:
        return True
    facts = task_state.get("facts")
    return bool(facts) if isinstance(facts, dict) else False


def _should_surface_planner_intent(*, upcoming_cycle: int, planner_error: bool = False) -> bool:
    mode = settings.get_execution_mode()
    cycle = max(1, int(upcoming_cycle))
    if mode == "ops":
        return True
    if mode == "dev":
        return planner_error or cycle == 1 or cycle % 2 == 0
    return planner_error or cycle == 1


def _wip_emit_every_for_mode(*, default_every: int) -> int:
    mode = settings.get_execution_mode()
    if mode == "ops":
        return 1
    if mode == "dev":
        return 2
    return max(3, int(default_every))


def _update_zero_progress_state(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    logger: Any,
    log_task_event: Any,
) -> None:
    signature = _latest_mission_fact_signature(task_state)
    if not signature:
        return
    last_signature = str(task_state.get("zero_progress_last_signature") or "").strip()
    streak = int(task_state.get("zero_progress_streak") or 0)
    if signature == last_signature:
        streak += 1
    else:
        streak = 0
    task_state["zero_progress_last_signature"] = signature
    task_state["zero_progress_streak"] = streak
    if streak <= 0:
        return
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="check_node",
        event="graph.check.zero_progress_detected",
        level="warning",
        repeat_count=streak,
        signature=signature,
    )
    if streak > _zero_progress_budget():
        task_state["status"] = "failed"
        task_state["check_route"] = "respond_node"
        task_state["next_user_question"] = None


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
        status = str(entry.get("status") or "").strip().lower()
        result_payload = entry.get("result_payload")
        payload_str = ""
        try:
            payload_str = json.dumps(result_payload, ensure_ascii=False, sort_keys=True)[:400]
        except Exception:
            payload_str = str(result_payload)[:400]
        raw = f"{tool}|{status}|{payload_str}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return ""
