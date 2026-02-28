from __future__ import annotations

import json
import logging
import inspect
import os
from typing import Any, Callable

from alphonse.agent.cognition.tool_schemas import llm_tool_schemas
from alphonse.agent.cortex.llm_output.json_parse import parse_json_object
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_REPAIR_USER_TEMPLATE
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_SYSTEM_PROMPT
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_USER_TEMPLATE
from alphonse.agent.cortex.task_mode.prompt_templates import render_pdca_prompt
from alphonse.agent.cortex.task_mode.progress_critic_node import build_wip_update_detail
from alphonse.agent.cortex.task_mode.types import NextStepProposal
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.session.day_state import render_recent_conversation_block
from alphonse.agent.tools.mcp.loader import default_profiles_dir
from alphonse.agent.tools.mcp.registry import McpProfileRegistry

_NEXT_STEP_MAX_ATTEMPTS_DEFAULT = 2

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


def build_next_step_node_impl(
    state: dict[str, Any],
    *,
    tool_registry: Any,
    task_state_with_defaults: Callable[[dict[str, Any]], dict[str, Any]],
    correlation_id: Callable[[dict[str, Any]], str | None],
    next_step_id: Callable[[dict[str, Any]], str],
    task_plan: Callable[[dict[str, Any]], dict[str, Any]],
    has_acceptance_criteria: Callable[[dict[str, Any]], bool],
    normalize_acceptance_criteria_values: Callable[[Any], list[str]],
    append_trace_event: Callable[[dict[str, Any], dict[str, Any]], None],
    logger: logging.Logger,
    log_task_event: Callable[..., None],
) -> dict[str, Any]:
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "plan"
    corr = correlation_id(state)
    _, mcp_capability_stats = _build_mcp_capability_menu()
    logger.info(
        "task_mode next_step mcp_capabilities correlation_id=%s profiles=%s operations=%s source_dir=%s",
        corr,
        int(mcp_capability_stats.get("profile_count") or 0),
        int(mcp_capability_stats.get("operation_count") or 0),
        str(mcp_capability_stats.get("source_dir") or ""),
    )
    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="next_step_node",
        event="graph.next_step.mcp_capabilities",
        profile_count=int(mcp_capability_stats.get("profile_count") or 0),
        operation_count=int(mcp_capability_stats.get("operation_count") or 0),
        source_dir=str(mcp_capability_stats.get("source_dir") or ""),
    )
    llm_client = state.get("_llm_client")
    proposal, parse_failed, diagnostics = _propose_next_step_with_llm(
        llm_client=llm_client,
        state=state,
        task_state=task_state,
        tool_registry=tool_registry,
        logger=logger,
        correlation=corr,
    )
    if diagnostics:
        for item in diagnostics:
            if not isinstance(item, dict):
                continue
            log_task_event(
                logger=logger,
                state=state,
                task_state=task_state,
                node="next_step_node",
                event="graph.next_step.parse_invalid",
                attempt=int(item.get("attempt") or 0),
                parse_error_type=str(item.get("parse_error_type") or ""),
                validation_errors=item.get("validation_errors"),
                raw_output_preview=str(item.get("raw_output_preview") or ""),
                provider=str(item.get("provider") or ""),
                model=str(item.get("model") or ""),
            )
    if not has_acceptance_criteria(task_state):
        proposed_criteria = normalize_acceptance_criteria_values(
            proposal.get("acceptance_criteria") if isinstance(proposal, dict) else None
        )
        if proposed_criteria:
            task_state["acceptance_criteria"] = proposed_criteria
            append_trace_event(
                task_state,
                {
                    "type": "acceptance_criteria_derived",
                    "summary": "Derived acceptance criteria from planning context.",
                    "correlation_id": corr,
                },
            )

    if parse_failed:
        task_state["status"] = "failed"
        task_state["next_user_question"] = None
        provider, model = _provider_model_from_state(state)
        attempts = int((diagnostics[-1] if diagnostics else {}).get("attempt") or _next_step_max_attempts())
        task_state["last_validation_error"] = {
            "reason": "next_step_parse_failed",
            "attempts": attempts,
            "schema": "NextStepProposal",
            "provider": provider,
            "model": model,
        }
        append_trace_event(
            task_state,
            {
                "type": "parse_failed",
                "summary": "Next-step parse failed; task paused as internal degradation.",
                "correlation_id": corr,
            },
        )
        log_task_event(
            logger=logger,
            state=state,
            task_state=task_state,
            node="next_step_node",
            event="graph.next_step.degraded",
            reason="next_step_parse_failed",
            attempts=attempts,
            provider=provider,
            model=model,
        )
        logger.warning(
            "task_mode next_step degraded correlation_id=%s reason=next_step_parse_failed attempts=%s provider=%s model=%s",
            corr,
            attempts,
            provider,
            model,
        )
        return {"task_state": task_state}
    step_id = next_step_id(task_state)
    step_entry = {
        "step_id": step_id,
        "proposal": proposal,
        "status": "proposed",
    }
    plan = task_plan(task_state)
    plan["steps"].append(step_entry)
    plan["current_step_id"] = step_id
    task_state["next_user_question"] = None
    task_state["last_validation_error"] = None
    append_trace_event(
        task_state,
        {
            "type": "proposal_created",
            "summary": f"Created {_proposal_summary(proposal)} ({step_id}).",
            "correlation_id": corr,
        },
    )
    logger.info(
        "task_mode next_step proposal correlation_id=%s step_id=%s kind=%s summary=%s parse_failed=%s",
        corr,
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
    proposed_cycle = int(task_state.get("cycle_index") or 0) + 1
    wip_detail = build_wip_update_detail(
        task_state=task_state,
        cycle=proposed_cycle,
        current_step=step_entry,
    )
    emit_transition_event(state, "wip_update", wip_detail)
    logger.info(
        "task_mode next_step wip_update correlation_id=%s cycle=%s intention=%s text=%s",
        corr,
        int(wip_detail.get("cycle") or 0),
        str(wip_detail.get("intention") or ""),
        str(wip_detail.get("text") or ""),
    )
    return {"task_state": task_state}


def route_after_next_step_impl(
    state: dict[str, Any],
    *,
    correlation_id: Callable[[dict[str, Any]], str | None],
    logger: logging.Logger,
) -> str:
    task_state = state.get("task_state")
    if isinstance(task_state, dict) and str(task_state.get("status") or "").strip().lower() in {"waiting_user", "failed"}:
        logger.info(
            "task_mode route_after_next_step correlation_id=%s route=respond_node reason=terminal_status",
            correlation_id(state),
        )
        return "respond_node"
    if _current_step_is_mcp_call(state):
        logger.info(
            "task_mode route_after_next_step correlation_id=%s route=mcp_handler_node",
            correlation_id(state),
        )
        return "mcp_handler_node"
    logger.info(
        "task_mode route_after_next_step correlation_id=%s route=execute_step_node",
        correlation_id(state),
    )
    return "execute_step_node"


def _current_step_is_mcp_call(state: dict[str, Any]) -> bool:
    task_state = state.get("task_state")
    if not isinstance(task_state, dict):
        return False
    plan = task_state.get("plan")
    if not isinstance(plan, dict):
        return False
    step_id = str(plan.get("current_step_id") or "").strip()
    if not step_id:
        return False
    steps = plan.get("steps")
    if not isinstance(steps, list):
        return False
    for item in steps:
        if not isinstance(item, dict):
            continue
        if str(item.get("step_id") or "") != step_id:
            continue
        proposal = item.get("proposal")
        if not isinstance(proposal, dict):
            return False
        return (
            str(proposal.get("kind") or "").strip() == "call_tool"
            and str(proposal.get("tool_name") or "").strip() == "mcp_call"
        )
    return False


def _propose_next_step_with_llm(
    *,
    llm_client: Any,
    state: dict[str, Any],
    task_state: dict[str, Any],
    tool_registry: Any,
    logger: logging.Logger,
    correlation: str | None,
) -> tuple[NextStepProposal, bool, list[dict[str, Any]]]:
    diagnostics: list[dict[str, Any]] = []
    max_attempts = _next_step_max_attempts()

    tool_contract_hints = _build_tool_contract_hints(tool_registry)
    mcp_capability_menu, _ = _build_mcp_capability_menu()
    mcp_live_tools_menu = _build_mcp_live_tools_menu(task_state)
    working_view = _build_working_state_view(task_state)
    injected_blocks: list[str] = []
    philosophy = str(state.get("philosophy_block") or "").strip()
    soul = str(state.get("soul_block") or "").strip()
    agents = str(state.get("agents_block") or "").strip()
    if philosophy:
        injected_blocks.append("## Philosophy\n" + philosophy)
    if soul:
        injected_blocks.append("## SOUL\n" + soul)
    if agents:
        injected_blocks.append("## AGENTS\n" + agents)
    injected_guidance = "\n\n".join(injected_blocks).strip()
    user_prompt_body = render_pdca_prompt(
        NEXT_STEP_USER_TEMPLATE,
        {
            "TOOL_CONTRACT_HINTS": tool_contract_hints,
            "MCP_CAPABILITY_MENU": mcp_capability_menu,
            "MCP_LIVE_TOOLS_MENU": mcp_live_tools_menu,
            "INJECTED_GUIDANCE_BLOCK": injected_guidance,
            "WORKING_STATE_VIEW_JSON": json.dumps(working_view, ensure_ascii=False),
        },
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
    logger.info(
        "task_mode next_step planner_prompt_prepared correlation_id=%s system_prompt_chars=%s user_prompt_chars=%s",
        correlation,
        len(NEXT_STEP_SYSTEM_PROMPT),
        len(user_prompt),
    )
    tool_supported, tool_based_proposal = _call_llm_tool_selection(
        llm_client=llm_client,
        system_prompt=NEXT_STEP_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_registry=tool_registry,
    )
    if tool_supported and tool_based_proposal is not None:
        return tool_based_proposal, False, diagnostics

    structured_supported, parsed_structured = _call_llm_structured(
        llm_client=llm_client,
        system_prompt=NEXT_STEP_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    if structured_supported:
        normalized = _normalize_next_step_proposal(parsed_structured)
        if normalized is not None:
            return normalized, False, diagnostics
        diagnostics.append(
            _build_parse_diagnostic(
                state=state,
                attempt=1,
                parse_error_type="structured_payload_invalid",
                validation_errors=["structured_output_failed_schema_validation"],
                raw_output=parsed_structured,
            )
        )
    else:
        raw = _call_llm_text(
            llm_client=llm_client,
            system_prompt=NEXT_STEP_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )
        parsed = parse_json_object(raw)
        normalized = _normalize_next_step_proposal(parsed)
        if normalized is not None:
            return normalized, False, diagnostics
        diagnostics.append(
            _build_parse_diagnostic(
                state=state,
                attempt=1,
                parse_error_type="text_payload_invalid",
                validation_errors=["text_output_not_valid_next_step_json"],
                raw_output=raw,
            )
        )

    for attempt in range(2, max_attempts + 1):
        repair_prompt = _build_next_step_repair_prompt(
            original_user_prompt=user_prompt,
            latest_diagnostic=diagnostics[-1] if diagnostics else {},
        )
        repair_raw = _call_llm_text(
            llm_client=llm_client,
            system_prompt=NEXT_STEP_SYSTEM_PROMPT,
            user_prompt=repair_prompt,
        )
        repair_parsed = parse_json_object(repair_raw)
        repaired = _normalize_next_step_proposal(repair_parsed)
        if repaired is not None:
            return repaired, False, diagnostics
        diagnostics.append(
            _build_parse_diagnostic(
                state=state,
                attempt=attempt,
                parse_error_type="repair_payload_invalid",
                validation_errors=["repair_output_not_valid_next_step_json"],
                raw_output=repair_raw,
            )
        )

    return {"kind": "ask_user", "question": ""}, True, diagnostics


def _next_step_max_attempts() -> int:
    raw = str(os.getenv("ALPHONSE_TASK_MODE_NEXT_STEP_MAX_ATTEMPTS") or "").strip()
    if not raw:
        return _NEXT_STEP_MAX_ATTEMPTS_DEFAULT
    try:
        value = int(raw)
    except ValueError:
        return _NEXT_STEP_MAX_ATTEMPTS_DEFAULT
    return value if value >= 1 else 1


def _provider_model_from_state(state: dict[str, Any]) -> tuple[str | None, str | None]:
    llm_client = state.get("_llm_client")
    provider = str(getattr(llm_client, "provider", "") or "").strip() or None
    model = str(getattr(llm_client, "model", "") or "").strip() or None
    return provider, model


def _build_parse_diagnostic(
    *,
    state: dict[str, Any],
    attempt: int,
    parse_error_type: str,
    validation_errors: list[str],
    raw_output: Any,
) -> dict[str, Any]:
    provider, model = _provider_model_from_state(state)
    preview = _truncate_text(_redact_secrets(str(raw_output or "")), limit=280)
    return {
        "attempt": int(attempt),
        "parse_error_type": str(parse_error_type or "next_step_parse_invalid"),
        "validation_errors": validation_errors,
        "raw_output_preview": preview,
        "provider": provider,
        "model": model,
    }


def _build_next_step_repair_prompt(*, original_user_prompt: str, latest_diagnostic: dict[str, Any]) -> str:
    parse_error_type = str((latest_diagnostic or {}).get("parse_error_type") or "").strip()
    validation_errors = latest_diagnostic.get("validation_errors")
    lines = []
    if isinstance(validation_errors, list):
        lines = [f"- {str(item)}" for item in validation_errors if str(item).strip()]
    if not lines:
        lines = ["- next_step_output_invalid"]
    return render_pdca_prompt(
        NEXT_STEP_REPAIR_USER_TEMPLATE,
        {
            "ORIGINAL_USER_PROMPT": original_user_prompt,
            "PARSE_ERROR_TYPE": parse_error_type or "unknown",
            "VALIDATION_ERRORS_LINES": "\n".join(lines),
        },
    )


def _redact_secrets(text: str) -> str:
    value = str(text or "")
    lowered = value.lower()
    redacted = value
    secret_tokens = ("password", "pass", "token", "api_key", "secret")
    if any(token in lowered for token in secret_tokens):
        for token in secret_tokens:
            redacted = redacted.replace(token, "[redacted_key]")
    return redacted


def _truncate_text(text: str, *, limit: int) -> str:
    value = str(text or "")
    if len(value) <= limit:
        return value
    return value[:limit]


def _build_working_state_view(task_state: dict[str, Any]) -> dict[str, Any]:
    facts = task_state.get("facts")
    relevant_facts = dict(facts) if isinstance(facts, dict) else {}
    if len(relevant_facts) > 8:
        keys = sorted(relevant_facts.keys())[-8:]
        relevant_facts = {key: relevant_facts[key] for key in keys}
    acceptance_criteria = _normalize_acceptance_criteria_values(task_state.get("acceptance_criteria"))
    failure_diagnostics = _latest_failure_diagnostics(task_state)
    return {
        "goal": str(task_state.get("goal") or "").strip(),
        "acceptance_criteria": acceptance_criteria,
        "relevant_facts": relevant_facts,
        "latest_failure_diagnostics": failure_diagnostics,
        "execution_eval": task_state.get("execution_eval") if isinstance(task_state.get("execution_eval"), dict) else {},
        "last_validation_error": task_state.get("last_validation_error"),
        "repair_attempts": int(task_state.get("repair_attempts") or 0),
        "pending_question": str(task_state.get("next_user_question") or "").strip() or None,
    }


def _latest_failure_diagnostics(task_state: dict[str, Any]) -> dict[str, Any]:
    plan = task_state.get("plan") if isinstance(task_state.get("plan"), dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    facts = task_state.get("facts") if isinstance(task_state.get("facts"), dict) else {}
    for raw_step in reversed(steps):
        if not isinstance(raw_step, dict):
            continue
        if str(raw_step.get("status") or "").strip().lower() != "failed":
            continue
        step_id = str(raw_step.get("step_id") or "").strip()
        proposal = raw_step.get("proposal") if isinstance(raw_step.get("proposal"), dict) else {}
        tool_name = str(proposal.get("tool_name") or "").strip()
        args = proposal.get("args") if isinstance(proposal.get("args"), dict) else {}
        fact_entry = facts.get(step_id) if step_id else None
        result = fact_entry.get("result") if isinstance(fact_entry, dict) and isinstance(fact_entry.get("result"), dict) else {}
        error = result.get("error") if isinstance(result.get("error"), dict) else {}
        return {
            "step_id": step_id,
            "tool_name": tool_name,
            "args": args,
            "error_code": str(error.get("code") or "").strip(),
            "error_message": str(error.get("message") or "").strip(),
        }
    return {}


def _build_tool_contract_hints(tool_registry: Any) -> str:
    # Keep this compact: only include high-friction tools that frequently fail
    # from argument-shape mismatches.
    target_tools = {"job_create", "mcp_call", "terminal_sync"}
    lines: list[str] = []
    for schema in llm_tool_schemas(tool_registry):
        function = schema.get("function")
        if not isinstance(function, dict):
            continue
        tool_name = str(function.get("name") or "").strip()
        if tool_name not in target_tools:
            continue
        params = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
        required = params.get("required") if isinstance(params.get("required"), list) else []
        required_fields = [str(item).strip() for item in required if str(item).strip()]
        lines.append(f"- `{tool_name}` required fields: {', '.join(required_fields) or '(none)'}")
        properties = params.get("properties") if isinstance(params.get("properties"), dict) else {}
        payload_type = properties.get("payload_type") if isinstance(properties.get("payload_type"), dict) else {}
        enum_values = payload_type.get("enum") if isinstance(payload_type.get("enum"), list) else []
        if tool_name == "job_create" and enum_values:
            rendered = ", ".join(str(item) for item in enum_values if str(item).strip())
            lines.append(f"  - `payload_type` must be one of: {rendered}")
        if tool_name == "job_create":
            lines.append(
                "  - `schedule` must be an object with `type`, `dtstart`, `rrule` (NOT a plain string)."
            )
        if tool_name == "mcp_call":
            lines.append(
                "  - args shape: `{\"profile\":\"...\",\"operation\":\"...\",\"arguments\":{...}}`."
            )
            lines.append(
                "  - optional `headless` (boolean) controls browser visibility for native browser MCP profiles."
            )
    return "\n".join(lines).strip()


def _build_mcp_capability_menu() -> tuple[str, dict[str, Any]]:
    registry = McpProfileRegistry()
    lines: list[str] = []
    if registry.keys():
        lines.append(
            "- MCP profiles are tool surfaces. Prefer the profile's capability model when planning actions."
        )
    operation_count = 0
    for profile_key in registry.keys():
        profile = registry.get(profile_key)
        if profile is None:
            continue
        metadata = profile.metadata if isinstance(profile.metadata, dict) else {}
        supports_native = bool(metadata.get("native_tools")) or str(metadata.get("capability_model") or "").strip().lower() in {
            "interactive_browser_server",
            "native_mcp",
        }
        lines.append(
            f"- profile `{profile.key}`: {profile.description} "
            f"(allowed_modes: {', '.join(profile.allowed_modes) or 'n/a'})"
        )
        category = str((profile.metadata or {}).get("category") or "").strip().lower()
        if category == "browser":
            lines.append(
                "  - capability_model `interactive_browser`: use this like a normal browser for agent tasks "
                "(open search engines, navigate websites, and read/extract page information)."
            )
        if supports_native:
            lines.append(
                "  - native MCP enabled: use `operation: \"list_tools\"` first to discover live server tools, "
                "then call `mcp_call` again with `operation` equal to the discovered tool name."
            )
        for operation in sorted(profile.operations.values(), key=lambda item: item.key):
            operation_count += 1
            args_hint = ", ".join(operation.required_args) if operation.required_args else "none"
            lines.append(
                f"  - operation `{operation.key}`: {operation.description} "
                f"(required_args: {args_hint})"
            )
    return "\n".join(lines).strip(), {
        "profile_count": len(registry.keys()),
        "operation_count": operation_count,
        "source_dir": str(default_profiles_dir()),
    }


def _build_mcp_live_tools_menu(task_state: dict[str, Any]) -> str:
    facts = task_state.get("facts") if isinstance(task_state.get("facts"), dict) else {}
    if not facts:
        return ""
    entries = list(facts.values())
    for fact in reversed(entries):
        if not isinstance(fact, dict):
            continue
        if str(fact.get("tool") or "").strip() != "mcp_call":
            continue
        result = fact.get("result")
        if not isinstance(result, dict):
            continue
        if str(result.get("status") or "").strip().lower() != "ok":
            continue
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        requested_operation = str(metadata.get("mcp_requested_operation") or metadata.get("mcp_operation") or "").strip()
        if requested_operation not in {"list_tools", "discover_tools"}:
            continue
        payload = result.get("result") if isinstance(result.get("result"), dict) else {}
        tools = payload.get("tools") if isinstance(payload.get("tools"), list) else []
        if not tools:
            continue
        profile = str(metadata.get("mcp_profile") or "").strip() or "unknown"
        lines = [f"- latest discovered native tools for profile `{profile}`:"]
        for item in tools[:20]:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            description = str(item.get("description") or "").strip()
            schema = item.get("inputSchema") if isinstance(item.get("inputSchema"), dict) else {}
            required_raw = schema.get("required") if isinstance(schema.get("required"), list) else []
            required = [str(x).strip() for x in required_raw if str(x).strip()]
            required_hint = ", ".join(required) if required else "none"
            if description:
                lines.append(f"  - `{name}`: {description} (required_args: {required_hint})")
            else:
                lines.append(f"  - `{name}` (required_args: {required_hint})")
        if len(lines) == 1:
            continue
        lines.append("  - invoke with `mcp_call` using `operation` equal to the tool name above.")
        return "\n".join(lines)
    return ""


def _call_llm_tool_selection(
    *,
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
    tool_registry: Any,
) -> tuple[bool, NextStepProposal | None]:
    complete_with_tools = getattr(llm_client, "complete_with_tools", None)
    if not callable(complete_with_tools):
        return False, None
    allowed_tools = llm_tool_schemas(tool_registry)
    payload = complete_with_tools(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=allowed_tools,
        tool_choice="auto",
    )
    return True, _normalize_tool_selection_payload(payload)


def _normalize_tool_selection_payload(payload: Any) -> NextStepProposal | None:
    if not isinstance(payload, dict):
        return None
    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("name") or "").strip()
            if not tool_name:
                continue
            arguments = call.get("arguments")
            args = dict(arguments) if isinstance(arguments, dict) else {}
            if tool_name == "askQuestion":
                question = str(args.get("question") or "").strip()
                if question:
                    return {"kind": "ask_user", "question": question}
            return {"kind": "call_tool", "tool_name": tool_name, "args": args}
    content = payload.get("content")
    if isinstance(content, str) and content.strip():
        parsed = parse_json_object(content)
        return _normalize_next_step_proposal(parsed)
    return None


def _call_llm_structured(
    *,
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
) -> tuple[bool, dict[str, Any] | None]:
    complete_json = getattr(llm_client, "complete_json", None)
    if callable(complete_json):
        payload = complete_json(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=_NEXT_STEP_SCHEMA,
        )
        if isinstance(payload, dict):
            return True, payload
        return True, None
    complete_with_schema = getattr(llm_client, "complete_with_schema", None)
    if callable(complete_with_schema):
        payload = complete_with_schema(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=_NEXT_STEP_SCHEMA,
        )
        if isinstance(payload, dict):
            return True, payload
        return True, None
    return False, None


def _call_llm_text(*, llm_client: Any, system_prompt: str, user_prompt: str) -> str:
    complete = getattr(llm_client, "complete", None)
    if not callable(complete):
        return ""
    try:
        signature = inspect.signature(complete)
    except (TypeError, ValueError):
        signature = None
    if signature and _supports_prompt_keywords(signature):
        return str(complete(system_prompt=system_prompt, user_prompt=user_prompt))
    return str(complete(system_prompt, user_prompt))


def _supports_prompt_keywords(signature: inspect.Signature) -> bool:
    names = set()
    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}:
            names.add(parameter.name)
    return {"system_prompt", "user_prompt"}.issubset(names)


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
