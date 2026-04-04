from __future__ import annotations

import json
from typing import Any, Callable

from alphonse.agent.cognition.providers.contracts import require_tool_calling_provider
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_SYSTEM_PROMPT
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_USER_TEMPLATE
from alphonse.agent.cortex.task_mode.prompt_templates import render_pdca_prompt
from alphonse.agent.session.day_state import render_recent_conversation_block
from alphonse.agent.tools.mcp.loader import default_profiles_dir
from alphonse.agent.tools.mcp.registry import McpProfileRegistry
from alphonse.agent.tools.registry import planner_tool_schemas

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


def _request_raw_candidate(
    *,
    llm_client: Any,
    system_prompt: str,
    user_prompt: str,
    tool_registry: Any,
) -> tuple[Any, str]:
    try:
        tool_client = require_tool_calling_provider(
            llm_client,
            source="task_mode.plan._request_raw_candidate",
        )
    except Exception as exc:
        return (
            {
                "error": {
                    "code": "planner_capability_missing",
                    "message": f"LLM client contract invalid for task_mode planning: {exc}",
                }
            },
            "complete_with_tools_unavailable",
        )
    return (
        tool_client.complete_with_tools(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tools=planner_tool_schemas(tool_registry),
            tool_choice="auto",
        ),
        "complete_with_tools",
    )


def _build_planner_user_prompt(*, state: dict[str, Any], task_state: dict[str, Any], tool_registry: Any) -> str:
    mcp_capability_menu, _ = _build_mcp_capability_menu()
    mcp_live_tools_menu = _build_mcp_live_tools_menu(task_state)
    working_view = _build_working_state_view(state=state, task_state=task_state)
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


def _build_working_state_view(*, state: dict[str, Any], task_state: dict[str, Any]) -> dict[str, Any]:
    relevant_facts = _select_relevant_facts(task_state)
    latest_failure = _latest_failure_diagnostics(task_state)
    return {
        "goal": str(task_state.get("goal") or "").strip(),
        "acceptance_criteria": task_state.get("acceptance_criteria") if isinstance(task_state.get("acceptance_criteria"), list) else [],
        "relevant_facts": relevant_facts,
        "reply_context": {
            "actor_person_id": _first_non_empty(task_state.get("actor_person_id"), state.get("actor_person_id")),
            "incoming_user_id": _first_non_empty(task_state.get("incoming_user_id"), state.get("incoming_user_id")),
            "channel_type": _first_non_empty(task_state.get("channel_type"), state.get("channel_type")),
            "channel_target": _first_non_empty(task_state.get("channel_target"), state.get("channel_target")),
            "conversation_key": _first_non_empty(task_state.get("conversation_key"), state.get("conversation_key")),
        },
        "delivery_context": {
            "channel_type": _first_non_empty(task_state.get("channel_type"), state.get("channel_type")),
            "channel_target": _first_non_empty(task_state.get("channel_target"), state.get("channel_target")),
            "message_id": _first_non_empty(task_state.get("message_id"), state.get("message_id")),
        },
        "latest_failure_diagnostics": latest_failure,
        "recipient_repair_context": _recipient_repair_context(
            task_state=task_state,
            relevant_facts=relevant_facts,
            latest_failure=latest_failure,
        ),
        "execution_eval": task_state.get("execution_eval") if isinstance(task_state.get("execution_eval"), dict) else {},
        "repair_attempts": int(task_state.get("repair_attempts") or 0),
        "pending_question": str(task_state.get("next_user_question") or "").strip() or None,
    }


def _first_non_empty(*values: Any) -> str | None:
    for value in values:
        rendered = str(value or "").strip()
        if rendered:
            return rendered
    return None


def _select_relevant_facts(task_state: dict[str, Any]) -> dict[str, Any]:
    facts = task_state.get("facts")
    if not isinstance(facts, dict):
        return {}
    plan = task_state.get("plan") if isinstance(task_state.get("plan"), dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    ordered_keys: list[str] = []
    for raw_step in steps:
        if not isinstance(raw_step, dict):
            continue
        step_id = str(raw_step.get("step_id") or "").strip()
        if step_id and step_id in facts and step_id not in ordered_keys:
            ordered_keys.append(step_id)
    for key in facts.keys():
        rendered = str(key or "").strip()
        if rendered and rendered not in ordered_keys:
            ordered_keys.append(rendered)
    selected = ordered_keys[-8:]
    return {key: facts[key] for key in selected if key in facts}


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
        output = fact_entry.get("output") if isinstance(fact_entry, dict) else None
        exception = fact_entry.get("exception") if isinstance(fact_entry, dict) else None
        if exception is None:
            result = fact_entry.get("result") if isinstance(fact_entry, dict) and isinstance(fact_entry.get("result"), dict) else {}
            exception = result.get("exception") if isinstance(result, dict) else None
            output = result.get("output") if isinstance(result, dict) else output
        error = exception if isinstance(exception, dict) else {"message": str(exception or "").strip()} if exception else {}
        return {
            "step_id": step_id,
            "tool_name": tool_name,
            "args": args,
            "error_code": str(error.get("code") or "").strip(),
            "error_message": str(error.get("message") or "").strip(),
            "output_preview": str(output)[:280] if output is not None else "",
        }
    return {}


def _recipient_repair_context(
    *,
    task_state: dict[str, Any],
    relevant_facts: dict[str, Any],
    latest_failure: dict[str, Any],
) -> dict[str, Any]:
    error_code = str(latest_failure.get("error_code") or "").strip().lower()
    if error_code != "unresolved_recipient":
        return {}
    latest_lookup = _latest_successful_lookup_fact(task_state=task_state, relevant_facts=relevant_facts)
    return {
        "latest_failure_code": error_code,
        "has_successful_lookup": bool(latest_lookup),
        "latest_successful_lookup": latest_lookup,
    }


def _latest_successful_lookup_fact(
    *,
    task_state: dict[str, Any],
    relevant_facts: dict[str, Any],
) -> dict[str, Any] | None:
    plan = task_state.get("plan") if isinstance(task_state.get("plan"), dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    ordered_step_ids = [
        str(raw_step.get("step_id") or "").strip()
        for raw_step in steps
        if isinstance(raw_step, dict) and str(raw_step.get("step_id") or "").strip()
    ]
    lookup_tools = {"get_user_details", "users.search"}
    for step_id in reversed(ordered_step_ids):
        fact = relevant_facts.get(step_id)
        if not isinstance(fact, dict):
            continue
        tool_name = str(fact.get("tool_name") or fact.get("tool") or "").strip()
        if tool_name not in lookup_tools:
            continue
        exception = fact.get("exception")
        if not _fact_is_success(exception):
            continue
        return {
            "step_id": step_id,
            "tool_name": tool_name,
            "output": fact.get("output"),
        }
    return None


def _fact_is_success(exception: Any) -> bool:
    if exception is None:
        return True
    if isinstance(exception, dict):
        return not any(str(exception.get(key) or "").strip() for key in ("code", "message"))
    if isinstance(exception, str):
        return not exception.strip()
    return False


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
                "then call `execution.call_mcp` again with `operation` equal to the discovered tool name."
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
        tool_name = str(fact.get("tool_name") or fact.get("tool") or "").strip()
        if tool_name != "execution.call_mcp":
            continue

        exception = fact.get("exception")
        output = fact.get("output")
        metadata = fact.get("metadata")

        if exception is None and isinstance(fact.get("result"), dict):
            legacy_result = fact.get("result")
            exception = legacy_result.get("exception")
            output = legacy_result.get("output")
            metadata = legacy_result.get("metadata")

        if exception is not None:
            continue
        metadata = metadata if isinstance(metadata, dict) else {}
        payload = output if isinstance(output, dict) else {}
        if isinstance(payload.get("output"), dict):
            nested_metadata = payload.get("metadata")
            if isinstance(nested_metadata, dict) and not metadata:
                metadata = nested_metadata
            payload = payload.get("output")
        requested_operation = str(metadata.get("mcp_requested_operation") or metadata.get("mcp_operation") or "").strip()
        if requested_operation not in {"list_tools", "discover_tools"}:
            continue
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
        lines.append("  - invoke with `execution.call_mcp` using `operation` equal to the tool name above.")
        return "\n".join(lines)
    return ""
