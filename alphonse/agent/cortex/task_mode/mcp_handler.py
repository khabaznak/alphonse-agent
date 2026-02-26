from __future__ import annotations

import logging
from typing import Any, Callable

from alphonse.agent.tools.mcp_call_tool import normalize_mcp_call_invocation


def mcp_handler_node_impl(
    state: dict[str, Any],
    *,
    task_state_with_defaults: Callable[[dict[str, Any]], dict[str, Any]],
    correlation_id: Callable[[dict[str, Any]], str | None],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    logger: logging.Logger,
    log_task_event: Callable[..., None],
) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "do"
    corr = correlation_id(state)
    current = current_step(task_state)
    proposal = (current or {}).get("proposal") if isinstance(current, dict) else None
    if not isinstance(proposal, dict):
        return {"task_state": task_state}
    if str(proposal.get("kind") or "").strip() != "call_tool":
        return {"task_state": task_state}
    if str(proposal.get("tool_name") or "").strip() != "mcp_call":
        return {"task_state": task_state}

    raw_args = proposal.get("args")
    payload = dict(raw_args) if isinstance(raw_args, dict) else {}
    normalized, report = normalize_mcp_call_invocation(payload)
    proposal["args"] = normalized

    intent = str(
        task_state.get("goal")
        or state.get("intent")
        or state.get("last_user_message")
        or ""
    ).strip()
    task_state["mcp_context"] = {
        "intent": intent,
        "profile": str(normalized.get("profile") or ""),
        "operation": str(normalized.get("operation") or ""),
    }
    append_trace_event(
        task_state,
        {
            "type": "mcp_prepared",
            "summary": (
                "Prepared MCP invocation with canonical arguments."
                if report.get("normalized")
                else "Prepared MCP invocation."
            ),
            "correlation_id": corr,
        },
    )
    logger.info(
        "task_mode mcp_handler prepared correlation_id=%s step_id=%s intent=%s profile=%s operation=%s mapped=%s ignored=%s",
        corr,
        str((current or {}).get("step_id") or ""),
        intent[:120],
        str(normalized.get("profile") or ""),
        str(normalized.get("operation") or ""),
        list(report.get("mapped") or []),
        list(report.get("ignored") or []),
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="mcp_handler_node",
        event="graph.mcp_handler.prepared",
        step_id=str((current or {}).get("step_id") or ""),
        intent=intent[:500],
        profile=str(normalized.get("profile") or ""),
        operation=str(normalized.get("operation") or ""),
        mapped=list(report.get("mapped") or []),
        ignored=list(report.get("ignored") or []),
    )
    return {"task_state": task_state}

