from __future__ import annotations

import inspect
import json
from typing import Any, Callable

from alphonse.agent.cognition.tool_schemas import llm_tool_schemas
from alphonse.agent.cortex.llm_output.json_parse import parse_json_object
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_SYSTEM_PROMPT
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_USER_TEMPLATE
from alphonse.agent.cortex.task_mode.prompt_templates import render_pdca_prompt
from alphonse.agent.cortex.transitions import emit_transition_event
from alphonse.agent.session.day_state import render_recent_conversation_block
from alphonse.agent.tools.mcp.loader import default_profiles_dir
from alphonse.agent.tools.mcp.registry import McpProfileRegistry

# Schema is kept for providers that support schema-native generation.
_NEXT_STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tool_call", "planner_intent"],
    "properties": {
        "tool_call": {
            "type": "object",
            "additionalProperties": False,
            "required": ["kind", "tool_name", "args"],
            "properties": {
                "kind": {"type": "string", "enum": ["call_tool"]},
                "tool_name": {"type": "string", "minLength": 1},
                "args": {"type": "object"},
            },
        },
        "planner_intent": {"type": "string", "minLength": 1},
    },
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
    logger: Any,
    log_task_event: Callable[..., None],
) -> dict[str, Any]:
    _ = has_acceptance_criteria
    _ = normalize_acceptance_criteria_values
    task_state = task_state_with_defaults(state)
    task_state["pdca_phase"] = "plan"
    corr = correlation_id(state)
    llm_client = state.get("_llm_client")

    _, mcp_capability_stats = _build_mcp_capability_menu()
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

    user_prompt = _build_planner_user_prompt(state=state, task_state=task_state, tool_registry=tool_registry)
    raw_candidate, source = _request_raw_candidate(
        llm_client=llm_client,
        system_prompt=NEXT_STEP_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        tool_registry=tool_registry,
    )

    task_state["pending_plan_raw"] = raw_candidate
    task_state.pop("planner_error_last", None)

    step_id = next_step_id(task_state)
    step_entry = {
        "step_id": step_id,
        "status": "proposed",
        "proposal_raw": raw_candidate,
        "raw_source": source,
    }
    plan = task_plan(task_state)
    plan["steps"].append(step_entry)
    plan["current_step_id"] = step_id

    append_trace_event(
        task_state,
        {
            "type": "proposal_created",
            "summary": f"Created raw planner candidate ({step_id}) via {source}.",
            "correlation_id": corr,
        },
    )

    log_task_event(
        logger=logger,
        state=state,
        task_state=task_state,
        node="next_step_node",
        event="graph.next_step.proposed_raw",
        step_id=step_id,
        source=source,
        raw_type=type(raw_candidate).__name__,
    )

    # Keep transition semantics for UI progress pulses; no decision-making here.
    candidate_dict = _extract_candidate_dict(raw_candidate)
    intent_text = _extract_planner_intent(raw_candidate)
    emit_transition_event(
        state,
        "wip_update",
        {
            "cycle": int(task_state.get("cycle_index") or 0) + 1,
            "tool": str((candidate_dict or {}).get("tool_name") or ""),
            "intention": "planning_next_step",
            "text": intent_text or "Estoy analizando el siguiente paso para avanzar la tarea.",
        },
    )

    logger.info(
        "task_mode next_step raw_candidate correlation_id=%s step_id=%s source=%s raw_type=%s",
        corr,
        step_id,
        source,
        type(raw_candidate).__name__,
    )
    return {"task_state": task_state}


def route_after_next_step_impl(
    state: dict[str, Any],
    *,
    correlation_id: Callable[[dict[str, Any]], str | None],
    logger: Any,
) -> str:
    logger.info(
        "task_mode route_after_next_step correlation_id=%s route=execute_step_node",
        correlation_id(state),
    )
    return "execute_step_node"


def _extract_candidate_dict(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        tool_call = raw.get("tool_call")
        if isinstance(tool_call, dict):
            return _extract_candidate_dict(tool_call)
        tool_calls = raw.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                if not isinstance(call, dict):
                    continue
                name = str(call.get("name") or "").strip()
                if not name:
                    continue
                args = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
                return {"kind": "call_tool", "tool_name": name, "args": dict(args)}
        kind = str(raw.get("kind") or "").strip()
        tool_name = str(raw.get("tool_name") or "").strip()
        args = raw.get("args")
        if kind == "call_tool" and tool_name and isinstance(args, dict):
            return {"kind": "call_tool", "tool_name": tool_name, "args": dict(args)}
        content = raw.get("content")
        if isinstance(content, str) and content.strip():
            parsed = parse_json_object(content)
            return _extract_candidate_dict(parsed)
        return None
    if isinstance(raw, str) and raw.strip():
        parsed = parse_json_object(raw)
        return _extract_candidate_dict(parsed)
    return None


def _extract_planner_intent(raw: Any) -> str:
    if isinstance(raw, dict):
        text = str(raw.get("planner_intent") or "").strip()
        if text:
            return text[:160]
        content = raw.get("content")
        if isinstance(content, str) and content.strip():
            return _extract_planner_intent(parse_json_object(content))
        return ""
    if isinstance(raw, str) and raw.strip():
        return _extract_planner_intent(parse_json_object(raw))
    return ""


def _request_raw_candidate(
    *,
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
    tool_registry: Any,
) -> tuple[Any, str]:
    complete_with_tools = getattr(llm_client, "complete_with_tools", None)
    if callable(complete_with_tools):
        return (
            complete_with_tools(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tools=llm_tool_schemas(tool_registry),
                tool_choice="auto",
            ),
            "complete_with_tools",
        )

    complete_json = getattr(llm_client, "complete_json", None)
    if callable(complete_json):
        return (
            complete_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_NEXT_STEP_SCHEMA,
            ),
            "complete_json",
        )

    complete_with_schema = getattr(llm_client, "complete_with_schema", None)
    if callable(complete_with_schema):
        return (
            complete_with_schema(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema=_NEXT_STEP_SCHEMA,
            ),
            "complete_with_schema",
        )

    return _call_llm_text(llm_client=llm_client, system_prompt=system_prompt, user_prompt=user_prompt), "complete"


def _build_planner_user_prompt(*, state: dict[str, Any], task_state: dict[str, Any], tool_registry: Any) -> str:
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
    return (
        f"{recent_conversation_block}\n\n{user_prompt_body}".strip()
        if recent_conversation_block
        else user_prompt_body
    )


def _build_working_state_view(task_state: dict[str, Any]) -> dict[str, Any]:
    facts = task_state.get("facts")
    relevant_facts = dict(facts) if isinstance(facts, dict) else {}
    if len(relevant_facts) > 8:
        keys = sorted(relevant_facts.keys())[-8:]
        relevant_facts = {key: relevant_facts[key] for key in keys}
    return {
        "goal": str(task_state.get("goal") or "").strip(),
        "acceptance_criteria": task_state.get("acceptance_criteria") if isinstance(task_state.get("acceptance_criteria"), list) else [],
        "relevant_facts": relevant_facts,
        "latest_failure_diagnostics": _latest_failure_diagnostics(task_state),
        "execution_eval": task_state.get("execution_eval") if isinstance(task_state.get("execution_eval"), dict) else {},
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
    target_tools = {"job_create", "mcp_call", "terminal_sync"}
    lines: list[str] = []
    for schema in llm_tool_schemas(tool_registry):
        function = schema.get("function") if isinstance(schema, dict) else None
        if not isinstance(function, dict):
            continue
        tool_name = str(function.get("name") or "").strip()
        if tool_name not in target_tools:
            continue
        params = function.get("parameters") if isinstance(function.get("parameters"), dict) else {}
        required = params.get("required") if isinstance(params.get("required"), list) else []
        required_fields = [str(item).strip() for item in required if str(item).strip()]
        lines.append(f"- `{tool_name}` required fields: {', '.join(required_fields) or '(none)' }")
    return "\n".join(lines).strip()


def _build_mcp_capability_menu() -> tuple[str, dict[str, Any]]:
    registry = McpProfileRegistry()
    lines: list[str] = []
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
        result = fact.get("result") if isinstance(fact.get("result"), dict) else {}
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
