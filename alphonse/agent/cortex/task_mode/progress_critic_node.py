from __future__ import annotations

import logging
from typing import Any, Callable

from alphonse.agent.cortex.task_mode.progress_critic import progress_critic_node as _progress_critic_node_impl
from alphonse.agent.cortex.task_mode.progress_critic import route_after_progress_critic as _route_after_progress_critic_impl
from alphonse.agent.cortex.task_mode.prompt_templates import PROGRESS_CHECKIN_SYSTEM_PROMPT
from alphonse.agent.cortex.task_mode.prompt_templates import PROGRESS_CHECKIN_USER_TEMPLATE
from alphonse.agent.cortex.task_mode.prompt_templates import render_pdca_prompt
from alphonse.agent.cortex.transitions import emit_transition_event

DEFAULT_PROGRESS_CHECK_CYCLE_THRESHOLD = 25
DEFAULT_WIP_EMIT_EVERY_CYCLES = 5


def progress_critic_node_stateful(
    state: dict[str, Any],
    *,
    task_state_with_defaults: Callable[[dict[str, Any]], dict[str, Any]],
    correlation_id: Callable[[dict[str, Any]], str | None],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
    goal_satisfied: Callable[[dict[str, Any]], bool],
    evaluate_tool_execution: Callable[..., dict[str, Any]],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    logger: logging.Logger,
    progress_check_cycle_threshold: int = DEFAULT_PROGRESS_CHECK_CYCLE_THRESHOLD,
    wip_emit_every_cycles: int = DEFAULT_WIP_EMIT_EVERY_CYCLES,
) -> dict[str, Any]:
    return _progress_critic_node_impl(
        state,
        task_state_with_defaults=task_state_with_defaults,
        correlation_id=correlation_id,
        current_step=current_step,
        goal_satisfied=goal_satisfied,
        evaluate_tool_execution=evaluate_tool_execution,
        append_trace_event=append_trace_event,
        build_progress_checkin_question=lambda *, state, task_state, evaluation: _build_progress_checkin_question(
            state=state,
            task_state=task_state,
            evaluation=evaluation,
            current_step=current_step,
        ),
        maybe_emit_periodic_wip_update=lambda *, state, task_state, cycle, current_step: _maybe_emit_periodic_wip_update(
            state=state,
            task_state=task_state,
            cycle=cycle,
            current_step=current_step,
            correlation_id=correlation_id,
            logger=logger,
            wip_emit_every_cycles=wip_emit_every_cycles,
        ),
        progress_check_cycle_threshold=progress_check_cycle_threshold,
    )


def route_after_progress_critic_stateful(
    state: dict[str, Any],
    *,
    correlation_id: Callable[[dict[str, Any]], str | None],
) -> str:
    return _route_after_progress_critic_impl(state, correlation_id=correlation_id)


def _maybe_emit_periodic_wip_update(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    cycle: int,
    current_step: dict[str, Any] | None,
    correlation_id: Callable[[dict[str, Any]], str | None],
    logger: logging.Logger,
    wip_emit_every_cycles: int,
) -> None:
    if cycle <= 0 or cycle % wip_emit_every_cycles != 0:
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
        correlation_id(state),
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


def _build_progress_checkin_question(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    evaluation: dict[str, Any],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> str:
    llm_client = state.get("_llm_client")
    goal = str(task_state.get("goal") or "").strip()
    cycle = int(task_state.get("cycle_index") or 0)
    current = current_step(task_state)
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
    user_prompt = render_pdca_prompt(
        PROGRESS_CHECKIN_USER_TEMPLATE,
        {
            "LOCALE": str(state.get("locale") or ""),
            "CYCLE_COUNT": cycle,
            "TASK_GOAL": goal,
            "CURRENT_STEP_KIND": current_kind,
            "CURRENT_TOOL": current_tool,
            "PROGRESS_SUMMARY": summary,
            "ACCEPTANCE_CRITERIA_LINES": criteria_lines,
        },
    )
    question = _call_llm_text(
        llm_client=llm_client,
        system_prompt=PROGRESS_CHECKIN_SYSTEM_PROMPT,
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
