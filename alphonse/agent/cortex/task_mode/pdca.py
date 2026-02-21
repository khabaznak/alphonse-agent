from __future__ import annotations

import inspect
import json
import logging
from datetime import datetime
from typing import Any, Callable

from alphonse.agent.cognition.tool_schemas import planner_tool_schemas
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.cortex.task_mode.progress_critic import progress_critic_node as _progress_critic_node_impl
from alphonse.agent.cortex.task_mode.progress_critic import route_after_progress_critic as _route_after_progress_critic_impl
from alphonse.agent.cortex.task_mode.observability import log_task_event
from alphonse.agent.cortex.task_mode.execute_step import execute_step_node_impl
from alphonse.agent.cortex.task_mode.state import build_default_task_state
from alphonse.agent.cortex.task_mode.validate_step import validate_step_node_impl
from alphonse.agent.cortex.task_mode.types import NextStepProposal
from alphonse.agent.cortex.task_mode.types import TraceEvent
from alphonse.agent.session.day_state import render_recent_conversation_block
from alphonse.agent.tools.scheduler import SchedulerTool
from alphonse.agent.tools.scheduler import SchedulerToolError

logger = logging.getLogger(__name__)

_NEXT_STEP_DEVELOPER_PROMPT = (
    "You are an iterative tool-using agent in a PDCA loop.\n"
    "Propose only the single next action.\n"
    "- Treat `RECENT CONVERSATION` as authoritative context for this session/day.\n"
    "- Use `RECENT CONVERSATION` to resolve follow-up references before asking the user again.\n"
    "- Choose one action kind: ask_user, call_tool, or finish.\n"
    "- Prefer tools that directly satisfy the user goal in one step.\n"
    "- Prefer direct-goal tools over informational tools.\n"
    "- Use informational tools only as prerequisites for missing required arguments of a direct tool.\n"
    "- If a tool accepts a natural time expression (e.g., 'in 1 minute'), pass it through; do NOT call getTime just to interpret it.\n"
    "- Call getTime only when a tool explicitly requires an absolute timestamp and you must compute it.\n"
    "- Prefer actions that reduce uncertainty early.\n"
    "- Use only tools listed in the tool menu.\n"
    "- If required info is missing, ask a concise clarifying question.\n"
    "- Review acceptance criteria from working state.\n"
    "- If acceptance criteria is missing, derive concise criteria from the goal and return them in `acceptance_criteria`.\n"
    "- Ask the user for acceptance criteria only when criteria cannot be inferred safely.\n"
    "- Never propose multi-step plans.\n"
    "- Keep user-facing text concise and natural."
)

_PARSE_FALLBACK_QUESTION = "I can help—what task would you like me to do?"
_PROGRESS_CHECK_CYCLE_THRESHOLD = 25
_WIP_EMIT_EVERY_CYCLES = 5

# Strict JSON Schema for NextStepProposal structured output
_NEXT_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["kind"],
    "properties": {
        "kind": {"type": "string", "enum": ["ask_user", "call_tool", "finish"]},
        "question": {"type": "string"},
        "tool_name": {"type": "string"},
        "args": {"type": "object"},
        "final_text": {"type": "string"},
        "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
    },
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"const": "ask_user"},
                "question": {"type": "string", "minLength": 1},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["kind", "question"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"const": "call_tool"},
                "tool_name": {"type": "string", "minLength": 1},
                "args": {"type": "object"},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["kind", "tool_name", "args"],
        },
        {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "kind": {"const": "finish"},
                "final_text": {"type": "string", "minLength": 1},
                "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["kind", "final_text"],
        },
    ],
}


def build_next_step_node(*, tool_registry: Any) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def _node(state: dict[str, Any]) -> dict[str, Any]:
        emit_transition_event(state, "thinking")
        task_state = _task_state_with_defaults(state)
        task_state["pdca_phase"] = "plan"
        correlation_id = _correlation_id(state)
        llm_client = state.get("_llm_client")
        proposal, parse_failed = _propose_next_step_with_llm(
            llm_client=llm_client,
            state=state,
            task_state=task_state,
            tool_registry=tool_registry,
        )
        if not _has_acceptance_criteria(task_state):
            proposed_criteria = _normalize_acceptance_criteria_values(
                proposal.get("acceptance_criteria") if isinstance(proposal, dict) else None
            )
            if proposed_criteria:
                task_state["acceptance_criteria"] = proposed_criteria
                _append_trace_event(
                    task_state,
                    {
                        "type": "acceptance_criteria_derived",
                        "summary": "Derived acceptance criteria from planning context.",
                        "correlation_id": correlation_id,
                    },
                )

        step_id = _next_step_id(task_state)
        step_entry = {
            "step_id": step_id,
            "proposal": proposal,
            "status": "proposed",
        }
        plan = _task_plan(task_state)
        plan["steps"].append(step_entry)
        plan["current_step_id"] = step_id
        task_state["next_user_question"] = None
        task_state["last_validation_error"] = None
        if parse_failed:
            task_state["status"] = "waiting_user"
            task_state["next_user_question"] = _PARSE_FALLBACK_QUESTION
            _append_trace_event(
                task_state,
                {
                    "type": "parse_failed",
                    "summary": "Next-step parse failed; using safe ask_user fallback.",
                    "correlation_id": _correlation_id(state),
                },
            )
        _append_trace_event(
            task_state,
            {
                "type": "proposal_created",
                "summary": f"Created {_proposal_summary(proposal)} ({step_id}).",
                "correlation_id": correlation_id,
            },
        )
        logger.info(
            "task_mode next_step proposal correlation_id=%s step_id=%s kind=%s summary=%s parse_failed=%s",
            correlation_id,
            step_id,
            str(proposal.get("kind") or ""),
            _proposal_summary(proposal),
            parse_failed,
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="next_step_node",
            event="graph.next_step.proposed",
            step_id=step_id,
            kind=str(proposal.get("kind") or ""),
            parse_failed=parse_failed,
            summary=_proposal_summary(proposal),
        )
        if parse_failed:
            return {"task_state": task_state}
        return {"task_state": task_state}

    return _node


def route_after_next_step(state: dict[str, Any]) -> str:
    task_state = state.get("task_state")
    if isinstance(task_state, dict) and str(task_state.get("status") or "") == "waiting_user":
        logger.info(
            "task_mode route_after_next_step correlation_id=%s route=respond_node reason=waiting_user",
            _correlation_id(state),
        )
        return "respond_node"
    logger.info(
        "task_mode route_after_next_step correlation_id=%s route=execute_step_node",
        _correlation_id(state),
    )
    return "execute_step_node"


def validate_step_node(state: dict[str, Any], *, tool_registry: Any) -> dict[str, Any]:
    return validate_step_node_impl(
        state,
        tool_registry=tool_registry,
        task_state_with_defaults=_task_state_with_defaults,
        correlation_id=_correlation_id,
        task_plan=_task_plan,
        current_step=_current_step,
        validate_proposal=_validate_proposal,
        append_trace_event=_append_trace_event,
        logger=logger,
        log_task_event=log_task_event,
    )


def route_after_validate_step(state: dict[str, Any]) -> str:
    task_state = state.get("task_state")
    if isinstance(task_state, dict) and str(task_state.get("status") or "") == "waiting_user":
        logger.info(
            "task_mode route_after_validate correlation_id=%s route=respond_node reason=waiting_user",
            _correlation_id(state),
        )
        return "respond_node"
    if isinstance(task_state, dict) and task_state.get("last_validation_error") is not None:
        logger.info(
            "task_mode route_after_validate correlation_id=%s route=next_step_node reason=validation_error",
            _correlation_id(state),
        )
        return "next_step_node"
    logger.info(
        "task_mode route_after_validate correlation_id=%s route=execute_step_node reason=validated",
        _correlation_id(state),
    )
    return "execute_step_node"


def execute_step_node(state: dict[str, Any], *, tool_registry: Any) -> dict[str, Any]:
    emit_transition_event(state, "executing")
    return execute_step_node_impl(
        state,
        tool_registry=tool_registry,
        task_state_with_defaults=_task_state_with_defaults,
        correlation_id=_correlation_id,
        current_step=_current_step,
        append_trace_event=_append_trace_event,
        serialize_result=_serialize_result,
        execute_tool_call=_execute_tool_call,
        logger=logger,
        log_task_event=log_task_event,
    )


def update_state_node(state: dict[str, Any]) -> dict[str, Any]:
    task_state = _task_state_with_defaults(state)
    task_state["pdca_phase"] = "check"
    correlation_id = _correlation_id(state)
    task_state["cycle_index"] = int(task_state.get("cycle_index") or 0) + 1
    outcome = _derive_outcome_from_state(state=state, task_state=task_state)
    task_state["outcome"] = outcome
    trace = _task_trace(task_state)
    trace["summary"] = f"PDCA cycle {task_state['cycle_index']} complete."
    _append_trace_event(
        task_state,
        {
            "type": "state_updated",
            "summary": f"State updated at cycle {task_state['cycle_index']}.",
            "correlation_id": correlation_id,
        },
    )
    logger.info(
        "task_mode update_state correlation_id=%s cycle=%s status=%s has_outcome=%s",
        correlation_id,
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


def progress_critic_node(state: dict[str, Any]) -> dict[str, Any]:
    return _progress_critic_node_impl(
        state,
        task_state_with_defaults=_task_state_with_defaults,
        correlation_id=_correlation_id,
        current_step=_current_step,
        goal_satisfied=_goal_satisfied,
        evaluate_tool_execution=_evaluate_tool_execution,
        append_trace_event=_append_trace_event,
        build_progress_checkin_question=_build_progress_checkin_question,
        maybe_emit_periodic_wip_update=_maybe_emit_periodic_wip_update,
        progress_check_cycle_threshold=_PROGRESS_CHECK_CYCLE_THRESHOLD,
    )


def _maybe_emit_periodic_wip_update(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    cycle: int,
    current_step: dict[str, Any] | None,
) -> None:
    if cycle <= 0 or cycle % _WIP_EMIT_EVERY_CYCLES != 0:
        return
    detail = {
        "text": _build_wip_update_text(task_state=task_state, cycle=cycle, current_step=current_step),
        "cycle": cycle,
        "goal": str(task_state.get("goal") or "").strip(),
        "tool": _current_tool_name(current_step),
    }
    emit_transition_event(state, "wip_update", detail)
    logger.info(
        "task_mode progress_critic wip_update correlation_id=%s cycle=%s",
        _correlation_id(state),
        cycle,
    )


def _build_wip_update_text(
    *,
    task_state: dict[str, Any],
    cycle: int,
    current_step: dict[str, Any] | None,
) -> str:
    goal = str(task_state.get("goal") or "").strip() or "the current task"
    tool = _current_tool_name(current_step)
    if tool:
        return f"Working on: {goal}. Cycle {cycle}. Current step: using `{tool}`."
    return f"Working on: {goal}. Cycle {cycle}."


def _current_tool_name(current_step: dict[str, Any] | None) -> str:
    if not isinstance(current_step, dict):
        return ""
    proposal = current_step.get("proposal") if isinstance(current_step.get("proposal"), dict) else {}
    return str(proposal.get("tool_name") or "").strip()


def route_after_progress_critic(state: dict[str, Any]) -> str:
    return _route_after_progress_critic_impl(state, correlation_id=_correlation_id)


def act_node(state: dict[str, Any]) -> dict[str, Any]:
    task_state = _task_state_with_defaults(state)
    task_state["pdca_phase"] = "act"
    correlation_id = _correlation_id(state)
    status = str(task_state.get("status") or "running")
    current = _current_step(task_state)
    current_status = str((current or {}).get("status") or "").strip().lower()

    if status == "waiting_user":
        logger.info(
            "task_mode act waiting_user correlation_id=%s step_id=%s",
            correlation_id,
            str((current or {}).get("step_id") or ""),
        )
        return {"task_state": task_state}

    if status == "failed":
        logger.info(
            "task_mode act failed correlation_id=%s step_id=%s",
            correlation_id,
            str((current or {}).get("step_id") or ""),
        )
        return {"task_state": task_state}

    if status == "done":
        logger.info(
            "task_mode act already_done correlation_id=%s step_id=%s",
            correlation_id,
            str((current or {}).get("step_id") or ""),
        )
        return {"task_state": task_state}

    if current_status == "executed":
        logger.info(
            "task_mode act continue_unsatisfied correlation_id=%s step_id=%s",
            correlation_id,
            str((current or {}).get("step_id") or ""),
        )
        task_state["status"] = "running"
        return {"task_state": task_state}

    if current_status == "failed":
        evaluation = _evaluate_tool_execution(task_state=task_state, current_step=current)
        task_state["execution_eval"] = evaluation
        if evaluation.get("should_pause"):
            task_state["status"] = "waiting_user"
            task_state["next_user_question"] = _build_execution_pause_prompt(evaluation)
            _append_trace_event(
                task_state,
                {
                    "type": "status_changed",
                    "summary": str(evaluation.get("summary") or "Status changed to waiting_user after repeated failures."),
                    "correlation_id": correlation_id,
                },
            )
            logger.info(
                "task_mode act waiting_user_execution_eval correlation_id=%s step_id=%s tool=%s reason=%s total_failures=%s same_signature=%s",
                correlation_id,
                str((current or {}).get("step_id") or ""),
                str(evaluation.get("tool_name") or ""),
                str(evaluation.get("reason") or ""),
                int(evaluation.get("total_failures") or 0),
                int(evaluation.get("same_signature_failures") or 0),
            )
            return {"task_state": task_state}
        task_state["status"] = "running"
        logger.info(
            "task_mode act continue_after_failure correlation_id=%s step_id=%s tool=%s reason=%s total_failures=%s same_signature=%s",
            correlation_id,
            str((current or {}).get("step_id") or ""),
            str(evaluation.get("tool_name") or ""),
            str(evaluation.get("reason") or ""),
            int(evaluation.get("total_failures") or 0),
            int(evaluation.get("same_signature_failures") or 0),
        )
        return {"task_state": task_state}

    task_state["status"] = "running"
    logger.info(
        "task_mode act continue correlation_id=%s step_id=%s",
        correlation_id,
        str((current or {}).get("step_id") or ""),
    )
    return {"task_state": task_state}


def route_after_act(state: dict[str, Any]) -> str:
    task_state = state.get("task_state")
    status = str((task_state or {}).get("status") or "").strip().lower() if isinstance(task_state, dict) else ""
    logger.info(
        "task_mode route_after_act correlation_id=%s route=next_step_node status=%s",
        _correlation_id(state),
        status or "running",
    )
    return "next_step_node"


def _validate_proposal(*, proposal: Any, tool_registry: Any) -> dict[str, Any]:
    if not isinstance(proposal, dict):
        return {"ok": False, "executable": False, "reason": "missing_proposal"}
    kind = str(proposal.get("kind") or "").strip()
    if kind == "ask_user":
        question = str(proposal.get("question") or "").strip()
        if not question:
            return {"ok": False, "executable": False, "reason": "missing_question"}
        return {"ok": True, "executable": True, "reason": None}
    if kind == "finish":
        final_text = str(proposal.get("final_text") or "").strip()
        if not final_text:
            return {"ok": False, "executable": False, "reason": "missing_final_text"}
        return {"ok": True, "executable": True, "reason": None}
    if kind != "call_tool":
        return {"ok": False, "executable": False, "reason": "unknown_kind"}

    tool_name = str(proposal.get("tool_name") or "").strip()
    if not tool_name:
        return {"ok": False, "executable": False, "reason": "missing_tool_name"}
    if not _tool_exists(tool_registry, tool_name):
        return {"ok": False, "executable": False, "reason": f"tool_not_found:{tool_name}"}

    args = proposal.get("args")
    if args is not None and not isinstance(args, dict):
        return {"ok": False, "executable": False, "reason": "invalid_args_type"}

    required = _required_args_for_tool(tool_name)
    provided = set(dict(args or {}).keys())
    missing = [key for key in required if key not in provided]
    if missing:
        missing_keys = ",".join(missing)
        return {"ok": False, "executable": False, "reason": f"missing_required_args:{missing_keys}"}
    invalid_keys = _invalid_args_for_tool(tool_name=tool_name, args=dict(args or {}))
    if invalid_keys:
        keys = ",".join(invalid_keys)
        return {"ok": False, "executable": False, "reason": f"invalid_args_keys:{keys}"}
    return {"ok": True, "executable": True, "reason": None}


def _execute_tool_call(
    *,
    state: dict[str, Any],
    tool_registry: Any,
    tool_name: str,
    args: dict[str, Any],
) -> Any:
    tool = tool_registry.get(tool_name) if hasattr(tool_registry, "get") else None
    if tool_name == "getTime":
        if tool is None and hasattr(tool_registry, "get"):
            tool = tool_registry.get("clock")
        if tool is None or not callable(getattr(tool, "get_time", None)):
            raise RuntimeError("getTime unavailable")
        now = tool.get_time()
        if isinstance(now, datetime):
            return now.isoformat()
        return now

    if tool_name == "createTimeEventTrigger":
        if tool is None or not callable(getattr(tool, "create_time_event_trigger", None)):
            raise RuntimeError("createTimeEventTrigger unavailable")
        return tool.create_time_event_trigger(time=str(args.get("time") or ""))

    if tool_name == "createReminder":
        if tool is None or not callable(getattr(tool, "create_reminder", None)):
            raise RuntimeError("createReminder unavailable")
        llm_client = state.get("_llm_client")
        if isinstance(tool, SchedulerTool) and getattr(tool, "llm_client", None) is None and llm_client is not None:
            tool = SchedulerTool(llm_client=llm_client)
        for_whom = str(args.get("ForWhom") or args.get("for_whom") or args.get("To") or "").strip()
        time_value = str(args.get("Time") or args.get("time") or "").strip()
        message_value = str(args.get("Message") or args.get("message") or "").strip()
        if not for_whom:
            for_whom = str(state.get("channel_target") or "").strip()
        from_value = str(state.get("channel_type") or "assistant")
        timezone_name = str(state.get("timezone") or "UTC")
        correlation_id = _correlation_id(state)
        try:
            reminder_id = tool.create_reminder(
                for_whom=for_whom,
                time=time_value,
                message=message_value,
                timezone_name=timezone_name,
                correlation_id=correlation_id,
                from_=from_value,
                channel_target=str(state.get("channel_target") or ""),
            )
        except SchedulerToolError as exc:
            return {
                "status": "failed",
                "result": None,
                "error": exc.as_payload(),
                "metadata": {"tool": "createReminder"},
            }
        except Exception as exc:
            return {
                "status": "failed",
                "result": None,
                "error": {
                    "code": "create_reminder_exception",
                    "message": str(exc) or type(exc).__name__,
                    "retryable": True,
                    "details": {"exception_type": type(exc).__name__},
                },
                "metadata": {"tool": "createReminder"},
            }
        if isinstance(reminder_id, dict):
            return reminder_id
        return {
            "reminder_id": str(reminder_id),
            "fire_at": time_value,
            "delivery_target": for_whom,
            "message": message_value,
        }

    if tool is None:
        raise RuntimeError(f"tool_not_found:{tool_name}")
    if callable(getattr(tool, "execute", None)):
        execute = getattr(tool, "execute")
        signature = inspect.signature(execute)
        try:
            if "state" in signature.parameters:
                return execute(**args, state=state)
            return execute(**args)
        except TypeError as exc:
            return {
                "status": "failed",
                "result": None,
                "error": {
                    "code": "invalid_tool_arguments",
                    "message": str(exc) or "invalid tool arguments",
                    "retryable": False,
                    "details": {
                        "tool": tool_name,
                        "args_keys": sorted(list(args.keys())),
                    },
                },
                "metadata": {"tool": tool_name},
            }
    raise RuntimeError(f"tool_not_executable:{tool_name}")


def _propose_next_step_with_llm(
    *,
    llm_client: Any,
    state: dict[str, Any],
    task_state: dict[str, Any],
    tool_registry: Any,
) -> tuple[NextStepProposal, bool]:
    goal = str(task_state.get("goal") or "").strip()
    if not goal:
        question = _build_goal_clarification_question(state=state, llm_client=llm_client)
        return {"kind": "ask_user", "question": question}, False

    tool_menu = _build_tool_menu(tool_registry)
    working_view = _build_working_state_view(task_state)
    user_prompt_body = (
        "## Tool Menu\n"
        f"{tool_menu}\n\n"
        "## Working State View\n"
        f"{json.dumps(working_view, ensure_ascii=False)}\n\n"
        "## Output Contract\n"
        "Return ONLY the JSON object that matches the schema. No markdown, no extra text.\n"
        "- kind ask_user requires question.\n"
        "- kind call_tool requires tool_name and args.\n"
        "- kind finish requires final_text."
    )
    recent_conversation_block = str(state.get("recent_conversation_block") or "").strip()
    if not recent_conversation_block:
        session_state = state.get("session_state") if isinstance(state.get("session_state"), dict) else None
        if session_state:
            recent_conversation_block = render_recent_conversation_block(session_state)
    user_prompt = (
        f"{recent_conversation_block}\n\n{user_prompt_body}".strip()
        if recent_conversation_block
        else user_prompt_body
    )

    structured_supported, parsed_structured = _call_llm_structured(
        llm_client=llm_client,
        system_prompt=_NEXT_STEP_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
    )
    if structured_supported:
        normalized = _normalize_next_step_proposal(parsed_structured)
        if normalized is not None:
            return normalized, False
        logger.info("task_mode next_step structured parse failure correlation_id=%s", _correlation_id(state))
        return {"kind": "ask_user", "question": _PARSE_FALLBACK_QUESTION}, True

    raw = _call_llm_text(
        llm_client=llm_client,
        system_prompt=_NEXT_STEP_DEVELOPER_PROMPT,
        user_prompt=user_prompt,
    )
    parsed = _parse_json_payload(raw)
    normalized = _normalize_next_step_proposal(parsed)
    if normalized is not None:
        return normalized, False

    logger.info("task_mode next_step parse failure correlation_id=%s", _correlation_id(state))
    return {"kind": "ask_user", "question": _PARSE_FALLBACK_QUESTION}, True


def _build_goal_clarification_question(*, state: dict[str, Any], llm_client: Any) -> str:
    system_prompt = (
        "You are Alphonse speaking to your human.\n"
        "Generate exactly one concise clarification question in the user's language.\n"
        "Ask what concrete task they want you to execute now.\n"
        "Output plain text only."
    )
    user_prompt = (
        "Context:\n"
        f"- locale: {str(state.get('locale') or '')}\n"
        f"- latest_user_message: {str(state.get('last_user_message') or '').strip()}\n"
        "\nWrite one question to clarify the task goal."
    )
    question = _call_llm_text(
        llm_client=llm_client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    rendered = str(question or "").strip()
    if rendered:
        return rendered
    latest = str(state.get("last_user_message") or "").strip()
    if latest:
        return f"To help correctly, what exact task should I do with: \"{latest}\"?"
    return "What exact task should I execute now?"


def _build_working_state_view(task_state: dict[str, Any]) -> dict[str, Any]:
    facts = task_state.get("facts")
    relevant_facts = dict(facts) if isinstance(facts, dict) else {}
    if len(relevant_facts) > 8:
        keys = sorted(relevant_facts.keys())[-8:]
        relevant_facts = {key: relevant_facts[key] for key in keys}
    acceptance_criteria = _normalize_acceptance_criteria_values(task_state.get("acceptance_criteria"))
    return {
        "goal": str(task_state.get("goal") or "").strip(),
        "acceptance_criteria": acceptance_criteria,
        "relevant_facts": relevant_facts,
        "last_validation_error": task_state.get("last_validation_error"),
        "repair_attempts": int(task_state.get("repair_attempts") or 0),
        "pending_question": str(task_state.get("next_user_question") or "").strip() or None,
    }


def _build_tool_menu(tool_registry: Any) -> str:
    descriptions = _tool_descriptions()
    keys = sorted(descriptions.keys())
    lines: list[str] = []
    for name in keys:
        if not _tool_exists(tool_registry, name):
            continue
        summary = descriptions.get(name) or "Tool available."
        lines.append(f"- `{name}`: {summary}")
    return "\n".join(lines[:24]) or "- (no tools)"


def _tool_descriptions() -> dict[str, str]:
    menu: dict[str, str] = {}
    for schema in planner_tool_schemas():
        fn = schema.get("function")
        if not isinstance(fn, dict):
            continue
        name = str(fn.get("name") or "").strip()
        if not name:
            continue
        description = str(fn.get("description") or "Tool available.").strip()
        menu[name] = description
    return menu


def _call_llm_structured(
    *,
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
) -> tuple[bool, dict[str, Any] | None]:
    complete_json = getattr(llm_client, "complete_json", None)
    if callable(complete_json):
        try:
            payload = complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_NEXT_STEP_SCHEMA,
            )
            if isinstance(payload, dict):
                return True, payload
        except Exception:
            return True, None
        return True, None
    complete_with_schema = getattr(llm_client, "complete_with_schema", None)
    if callable(complete_with_schema):
        try:
            payload = complete_with_schema(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_NEXT_STEP_SCHEMA,
            )
            if isinstance(payload, dict):
                return True, payload
        except Exception:
            return True, None
        return True, None
    return False, None


def _call_llm_text(*, llm_client: Any, system_prompt: str, user_prompt: str) -> str:
    complete = getattr(llm_client, "complete", None)
    if not callable(complete):
        return ""
    try:
        return str(complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except TypeError:
        try:
            return str(complete(system_prompt, user_prompt))
        except Exception:
            return ""
    except Exception:
        return ""


def _build_progress_checkin_question(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    evaluation: dict[str, Any],
) -> str:
    llm_client = state.get("_llm_client")
    goal = str(task_state.get("goal") or "").strip()
    cycle = int(task_state.get("cycle_index") or 0)
    current = _current_step(task_state)
    proposal = (current or {}).get("proposal") if isinstance(current, dict) else {}
    current_kind = str((proposal or {}).get("kind") or "").strip()
    current_tool = str((proposal or {}).get("tool_name") or "").strip()
    summary = str((evaluation or {}).get("summary") or "").strip()
    acceptance_criteria = task_state.get("acceptance_criteria")
    criteria_lines = (
        "\n".join(f"- {str(item).strip()}" for item in acceptance_criteria if str(item).strip())
        if isinstance(acceptance_criteria, list)
        else ""
    )
    if not criteria_lines:
        criteria_lines = "- (not provided)"
    system_prompt = (
        "You are Alphonse speaking to your human.\n"
        "Generate exactly one concise question in the user's language.\n"
        "The question must disclose current work-in-progress and ask whether to continue or stop.\n"
        "Do not mention internal technical terms, stack traces, nodes, or recursion.\n"
        "Output plain text only."
    )
    user_prompt = (
        "Context:\n"
        f"- locale: {str(state.get('locale') or '')}\n"
        f"- cycle_count: {cycle}\n"
        f"- task_goal: {goal}\n"
        f"- current_step_kind: {current_kind}\n"
        f"- current_tool: {current_tool}\n"
        f"- progress_summary: {summary}\n\n"
        "Acceptance criteria:\n"
        f"{criteria_lines}\n\n"
        "Write one question to the human asking whether to continue or stop."
    )
    question = _call_llm_text(
        llm_client=llm_client,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )
    rendered = str(question or "").strip()
    if rendered:
        return rendered
    fallback_goal = goal or "this task"
    fallback_wip = current_tool or "the current plan"
    return (
        f"I have been working on {fallback_goal} for {cycle} cycles and I am currently using {fallback_wip}. "
        "Would you like me to continue or stop?"
    )


def _parse_json_payload(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
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


def _json_loads(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return None


def _normalize_next_step_proposal(payload: Any) -> NextStepProposal | None:
    if not isinstance(payload, dict):
        return None
    kind = str(payload.get("kind") or "").strip()
    criteria = _normalize_acceptance_criteria_values(payload.get("acceptance_criteria"))
    if kind == "ask_user":
        question = str(payload.get("question") or "").strip()
        if not question:
            return None
        out: NextStepProposal = {"kind": "ask_user", "question": question}
        if criteria:
            out["acceptance_criteria"] = criteria
        return out
    if kind == "call_tool":
        tool_name = str(payload.get("tool_name") or "").strip()
        args = payload.get("args")
        if not tool_name or not isinstance(args, dict):
            return None
        out = {"kind": "call_tool", "tool_name": tool_name, "args": dict(args)}
        if criteria:
            out["acceptance_criteria"] = criteria
        return out
    if kind == "finish":
        final_text = str(payload.get("final_text") or "").strip()
        if not final_text:
            return None
        out = {"kind": "finish", "final_text": final_text}
        if criteria:
            out["acceptance_criteria"] = criteria
        return out
    return None


def _goal_satisfied(task_state: dict[str, Any]) -> bool:
    outcome = task_state.get("outcome")
    if not isinstance(outcome, dict) or not outcome:
        return False
    kind = str(outcome.get("kind") or "").strip().lower()
    if kind == "task_completed":
        summary = str(
            outcome.get("final_text")
            or outcome.get("summary")
            or ""
        ).strip()
        if not summary or _looks_like_question(summary):
            return False
        return _has_acceptance_criteria(task_state)
    return True


def _looks_like_question(text: str) -> bool:
    rendered = str(text or "").strip().lower()
    if not rendered:
        return False
    if "?" in rendered:
        return True
    starters = (
        "can ",
        "could ",
        "would ",
        "should ",
        "do ",
        "did ",
        "is ",
        "are ",
        "what ",
        "when ",
        "where ",
        "why ",
        "how ",
    )
    return rendered.startswith(starters)


def _derive_outcome_from_state(*, state: dict[str, Any], task_state: dict[str, Any]) -> dict[str, Any] | None:
    current = _current_step(task_state)
    if not isinstance(current, dict):
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None
    step_id = str(current.get("step_id") or "").strip()
    if not step_id:
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None
    facts = task_state.get("facts")
    if not isinstance(facts, dict):
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None
    entry = facts.get(step_id)
    if not isinstance(entry, dict):
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None

    tool_name = str(entry.get("tool") or "").strip()
    result = entry.get("result")
    if tool_name != "createReminder" or not isinstance(result, dict):
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None

    reminder_id = str(result.get("reminder_id") or "").strip()
    fire_at = str(result.get("fire_at") or "").strip()
    message = str(result.get("message") or "").strip()
    if not reminder_id or not fire_at:
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None

    for_whom = str(result.get("for_whom") or state.get("channel_target") or "").strip()
    if not for_whom:
        for_whom = str(result.get("delivery_target") or "").strip()
    return {
        "kind": "reminder_created",
        "evidence": {
            "reminder_id": reminder_id,
            "fire_at": fire_at,
            "message": message,
            "for_whom": for_whom,
        },
    }


def _proposal_summary(proposal: dict[str, Any]) -> str:
    kind = str(proposal.get("kind") or "").strip()
    if kind == "call_tool":
        tool = str(proposal.get("tool_name") or "").strip()
        return f"call_tool:{tool or 'unknown'}"
    if kind == "ask_user":
        question = str(proposal.get("question") or "").strip()
        return f"ask_user:{question[:48]}"
    if kind == "finish":
        text = str(proposal.get("final_text") or "").strip()
        return f"finish:{text[:48]}"
    return kind or "unknown"


def _task_state_with_defaults(state: dict[str, Any]) -> dict[str, Any]:
    existing = state.get("task_state")
    task_state = dict(existing) if isinstance(existing, dict) else {}
    defaults = build_default_task_state()
    for key, value in defaults.items():
        if key not in task_state:
            task_state[key] = value
    _task_plan(task_state)
    _task_trace(task_state)
    task_state.setdefault("facts", {})
    task_state.setdefault("status", "running")
    task_state.setdefault("repair_attempts", 0)
    task_state.setdefault("acceptance_criteria", [])
    return task_state


def _has_acceptance_criteria(task_state: dict[str, Any]) -> bool:
    return bool(_normalize_acceptance_criteria_values(task_state.get("acceptance_criteria")))


def _normalize_acceptance_criteria_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        out.append(text[:180])
        if len(out) >= 8:
            break
    return out


def _task_plan(task_state: dict[str, Any]) -> dict[str, Any]:
    plan = task_state.get("plan")
    if not isinstance(plan, dict):
        plan = {"version": 1, "steps": [], "current_step_id": None}
        task_state["plan"] = plan
    if not isinstance(plan.get("steps"), list):
        plan["steps"] = []
    if "version" not in plan:
        plan["version"] = 1
    if "current_step_id" not in plan:
        plan["current_step_id"] = None
    return plan


def _task_trace(task_state: dict[str, Any]) -> dict[str, Any]:
    trace = task_state.get("trace")
    if not isinstance(trace, dict):
        trace = {"summary": "", "recent": []}
        task_state["trace"] = trace
    if not isinstance(trace.get("recent"), list):
        trace["recent"] = []
    if "summary" not in trace:
        trace["summary"] = ""
    return trace


def _append_trace_event(task_state: dict[str, Any], event: TraceEvent) -> None:
    trace = _task_trace(task_state)
    recent = trace["recent"]
    recent.append(
        {
            "type": str(event.get("type") or "event"),
            "summary": str(event.get("summary") or "").strip()[:180],
            "correlation_id": event.get("correlation_id"),
        }
    )
    trace["recent"] = recent[-25:]


def _tool_exists(tool_registry: Any, tool_name: str) -> bool:
    if hasattr(tool_registry, "get"):
        return tool_registry.get(tool_name) is not None
    return False


def _required_args_for_tool(tool_name: str) -> list[str]:
    for schema in planner_tool_schemas():
        fn = schema.get("function")
        if not isinstance(fn, dict):
            continue
        if str(fn.get("name") or "") != tool_name:
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            return []
        required = params.get("required")
        if isinstance(required, list):
            return [str(item) for item in required if str(item)]
        return []
    return []


def _invalid_args_for_tool(*, tool_name: str, args: dict[str, Any]) -> list[str]:
    params = _tool_parameters_for_tool(tool_name)
    if not isinstance(params, dict):
        return []
    additional_properties = params.get("additionalProperties")
    if additional_properties is not False:
        return []
    properties = params.get("properties")
    if not isinstance(properties, dict):
        return []
    allowed = {str(k) for k in properties.keys()}
    invalid = [key for key in args.keys() if str(key) not in allowed]
    return sorted([str(item) for item in invalid if str(item)])


def _tool_parameters_for_tool(tool_name: str) -> dict[str, Any] | None:
    for schema in planner_tool_schemas():
        fn = schema.get("function")
        if not isinstance(fn, dict):
            continue
        if str(fn.get("name") or "") != tool_name:
            continue
        params = fn.get("parameters")
        if isinstance(params, dict):
            return params
        return None
    return None


def _next_step_id(task_state: dict[str, Any]) -> str:
    steps = _task_plan(task_state).get("steps")
    index = len(steps) + 1 if isinstance(steps, list) else 1
    return f"step_{index}"


def _current_step(task_state: dict[str, Any]) -> dict[str, Any] | None:
    plan = _task_plan(task_state)
    current_id = str(plan.get("current_step_id") or "").strip()
    steps = plan.get("steps")
    if not current_id or not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("step_id") or "") == current_id:
            return step
    return None


def _serialize_result(result: Any) -> Any:
    if isinstance(result, (dict, list, str, int, float, bool)) or result is None:
        return result
    return str(result)


def _correlation_id(state: dict[str, Any]) -> str | None:
    value = state.get("correlation_id")
    if value is None:
        return None
    return str(value)


def _tool_name_for_step(step: dict[str, Any] | None) -> str:
    if not isinstance(step, dict):
        return ""
    proposal = step.get("proposal") if isinstance(step.get("proposal"), dict) else {}
    return str(proposal.get("tool_name") or "").strip()


def _tool_signature_for_step(step: dict[str, Any] | None) -> str:
    if not isinstance(step, dict):
        return ""
    proposal = step.get("proposal") if isinstance(step.get("proposal"), dict) else {}
    tool_name = str(proposal.get("tool_name") or "").strip()
    args = proposal.get("args") if isinstance(proposal.get("args"), dict) else {}
    try:
        args_text = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception:
        args_text = str(args)
    return f"{tool_name}|{args_text}"


def _evaluate_tool_execution(*, task_state: dict[str, Any], current_step: dict[str, Any] | None) -> dict[str, Any]:
    max_evolving_failures = 10
    immediate_repeat_limit = 2
    current_signature = _tool_signature_for_step(current_step)
    current_tool = _tool_name_for_step(current_step)
    plan = _task_plan(task_state)
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []

    failed_steps: list[dict[str, Any]] = [
        step
        for step in steps
        if isinstance(step, dict) and str(step.get("status") or "").strip().lower() == "failed"
    ]
    total_failures = len(failed_steps)
    same_signature_failures = 0
    signatures: list[str] = []
    for step in failed_steps:
        signature = _tool_signature_for_step(step)
        if signature:
            signatures.append(signature)
        if current_signature and signature == current_signature:
            same_signature_failures += 1
    unique_signatures = len({item for item in signatures if item})
    evolving = unique_signatures > 1

    if same_signature_failures >= immediate_repeat_limit:
        return {
            "should_pause": True,
            "reason": "repeated_identical_failure",
            "summary": f"Repeated identical tool attempt failed for {current_tool or 'tool'}.",
            "tool_name": current_tool,
            "total_failures": total_failures,
            "same_signature_failures": same_signature_failures,
            "unique_signatures": unique_signatures,
            "evolving": evolving,
            "max_evolving_failures": max_evolving_failures,
        }
    if total_failures >= max_evolving_failures:
        return {
            "should_pause": True,
            "reason": "failure_budget_exhausted",
            "summary": "Failure budget exhausted while trying multiple strategies.",
            "tool_name": current_tool,
            "total_failures": total_failures,
            "same_signature_failures": same_signature_failures,
            "unique_signatures": unique_signatures,
            "evolving": evolving,
            "max_evolving_failures": max_evolving_failures,
        }
    return {
        "should_pause": False,
        "reason": "continue_learning" if evolving else "single_failure",
        "summary": "Continue with next planning attempt.",
        "tool_name": current_tool,
        "total_failures": total_failures,
        "same_signature_failures": same_signature_failures,
        "unique_signatures": unique_signatures,
        "evolving": evolving,
        "max_evolving_failures": max_evolving_failures,
    }


def _build_execution_pause_prompt(evaluation: dict[str, Any]) -> str:
    reason = str(evaluation.get("reason") or "")
    tool_name = str(evaluation.get("tool_name") or "tool")
    total = int(evaluation.get("total_failures") or 0)
    unique = int(evaluation.get("unique_signatures") or 0)
    if reason == "repeated_identical_failure":
        return (
            f"I got stuck repeating the same failed action with `{tool_name}`. "
            "I paused the plan to avoid waste. Do you want me to keep trying with your approval, "
            "or provide steering on a different approach?"
        )
    return (
        f"I tried {total} times across {unique} strategy variants and I'm still blocked. "
        "I paused the plan. Should I keep pursuing this goal, or do you want to steer me?"
    )
