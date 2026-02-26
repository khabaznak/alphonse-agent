from __future__ import annotations

import logging
import re
from typing import Any, Callable

from alphonse.agent.cortex.task_mode.progress_critic import progress_critic_node as _progress_critic_node_impl
from alphonse.agent.cortex.task_mode.progress_critic import route_after_progress_critic as _route_after_progress_critic_impl
from alphonse.agent.cortex.task_mode.prompt_templates import PROGRESS_CHECKIN_SYSTEM_PROMPT
from alphonse.agent.cortex.task_mode.prompt_templates import PROGRESS_CHECKIN_USER_TEMPLATE
from alphonse.agent.cortex.task_mode.prompt_templates import render_pdca_prompt
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.config import settings

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
    if cycle <= 0:
        return
    emit_every = 1 if settings.get_execution_mode() == "ops" else max(1, int(wip_emit_every_cycles))
    if cycle % emit_every != 0:
        return
    detail = {
        "text": _build_wip_update_text(task_state=task_state, cycle=cycle, current_step=current_step),
        "cycle": cycle,
        "goal": str(task_state.get("goal") or "").strip(),
        "tool": _current_tool_name(current_step),
        "intention": _current_intention(current_step),
    }
    emit_transition_event(state, "wip_update", detail)
    logger.info(
        "task_mode progress_critic wip_update correlation_id=%s cycle=%s intention=%s text=%s",
        correlation_id(state),
        cycle,
        str(detail.get("intention") or ""),
        str(detail.get("text") or ""),
    )


def _build_wip_update_text(
    *,
    task_state: dict[str, Any],
    cycle: int,
    current_step: dict[str, Any] | None,
) -> str:
    goal = str(task_state.get("goal") or "").strip() or "the current task"
    tool = _current_tool_name(current_step)
    intention = _current_intention(current_step)
    if intention and tool:
        return f"Working on: {goal}. Cycle {cycle}. Intention: {intention}. Current step: using `{tool}`."
    if intention:
        return f"Working on: {goal}. Cycle {cycle}. Intention: {intention}."
    if tool:
        return f"Working on: {goal}. Cycle {cycle}. Current step: using `{tool}`."
    return f"Working on: {goal}. Cycle {cycle}."


def _current_tool_name(current_step: dict[str, Any] | None) -> str:
    if not isinstance(current_step, dict):
        return ""
    proposal = current_step.get("proposal") if isinstance(current_step.get("proposal"), dict) else {}
    return str(proposal.get("tool_name") or "").strip()


def _current_intention(current_step: dict[str, Any] | None) -> str:
    if not isinstance(current_step, dict):
        return ""
    proposal = current_step.get("proposal") if isinstance(current_step.get("proposal"), dict) else {}
    kind = str(proposal.get("kind") or "").strip()
    if kind == "ask_user":
        return "I am preparing a clarifying question"
    if kind == "finish":
        return "I am preparing the final response"
    if kind != "call_tool":
        return ""
    tool = str(proposal.get("tool_name") or "").strip()
    args = proposal.get("args") if isinstance(proposal.get("args"), dict) else {}
    if tool in {"terminal_sync", "terminal_async", "ssh_terminal"}:
        command = str(args.get("command") or "").strip()
        return _terminal_intention(command)
    if tool in {"send_message", "sendMessage"}:
        return "I am preparing and sending an update"
    if tool in {"telegram_download_file"}:
        return "I am downloading the file required for the task"
    if tool == "mcp_call":
        profile = str(args.get("profile") or "").strip()
        operation = str(args.get("operation") or "").strip()
        if profile and operation:
            return f"I am invoking MCP profile `{profile}` operation `{operation}`"
        if profile:
            return f"I am invoking MCP profile `{profile}`"
        return "I am invoking an MCP operation to advance the task"
    if tool:
        return f"I am using `{tool}` to advance the task"
    return ""


def _terminal_intention(command: str) -> str:
    text = str(command or "").strip()
    if not text:
        return "I am running a terminal command to advance the task"
    lowered = text.lower()

    if _contains_any(lowered, ("npm install", "pip install", "apt-get install", "brew install", "npx -y")):
        pkg = _extract_package_name(text)
        if pkg:
            return f"I am trying to install dependency `{pkg}`"
        return "I am trying to install a dependency required by this task"
    if _contains_any(lowered, ("command -v", "which ")):
        target = _extract_probe_target(text)
        if target:
            return f"I am checking whether `{target}` is available in this environment"
        return "I am checking whether a required command is available"
    if _contains_any(lowered, (".sqlite", "sqlite3", "pragma", "select ")):
        return "I am querying the database to extract the requested information"
    if _contains_any(lowered, ("npm search", "pip index", "curl ", "wget ")):
        return "I am searching for the required dependency or source"
    if _contains_any(lowered, ("cat ", "tee ", "echo ", ">>", ">")):
        return "I am creating or updating a file with new information"
    return "I am executing a terminal step to gather evidence for the task"


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(item in haystack for item in needles)


def _extract_package_name(command: str) -> str:
    patterns = [
        r"npm\s+install(?:\s+-g)?\s+([@\w./-]+)",
        r"npx\s+-y\s+([@\w./-]+)",
        r"pip(?:3)?\s+install\s+([@\w./:-]+)",
        r"brew\s+install\s+([@\w./:-]+)",
        r"apt-get\s+install(?:\s+-y)?\s+([@\w./:-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, command, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).strip()
    return ""


def _extract_probe_target(command: str) -> str:
    patterns = [
        r"command\s+-v\s+([@\w./-]+)",
        r"which\s+([@\w./-]+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, command, flags=re.IGNORECASE)
        if match:
            return str(match.group(1)).strip()
    return ""


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
