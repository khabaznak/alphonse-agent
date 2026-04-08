from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, TypedDict

from alphonse.agent.cognition.providers.contracts import CanonicalToolCall
from alphonse.agent.cognition.providers.contracts import ToolCallingProvider
from alphonse.agent.cognition.providers.contracts import require_canonical_single_tool_call_result
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_SYSTEM_PROMPT
from alphonse.agent.cortex.task_mode.prompt_templates import NEXT_STEP_USER_TEMPLATE
from alphonse.agent.cortex.task_mode.prompt_templates import render_pdca_prompt
from alphonse.agent.cortex.task_mode.task_record import TaskRecord
from alphonse.agent.tools.mcp.loader import default_profiles_dir
from alphonse.agent.tools.mcp.registry import McpProfileRegistry
from alphonse.agent.tools.registry import ToolRegistry
from alphonse.agent.tools.registry import build_default_tool_registry
from alphonse.agent.tools.registry import planner_tool_schemas


class PlannerOutput(TypedDict):
    tool_call: CanonicalToolCall
    planner_intent: str


def plan_node_impl(
    task_record: TaskRecord,
    *,
    llm_client: ToolCallingProvider | None,
    logger: Any,
    log_task_event: Any,
) -> PlannerOutput:
    user_prompt = _build_planner_user_prompt(task_record=task_record)
    planner_output = _request_planner_output(
        llm_client=llm_client,
        system_prompt=NEXT_STEP_SYSTEM_PROMPT,
        user_prompt=user_prompt,
    )
    _append_planner_output_to_record(task_record, planner_output=planner_output)
    _log_planner_output(
        task_record=task_record,
        planner_output=planner_output,
        logger=logger,
        log_task_event=log_task_event,
    )
    return planner_output


def route_after_next_step_impl(state: dict[str, Any], *, correlation_id: Any, logger: Any) -> str:
    _ = (state, correlation_id, logger)
    return "execute_step_node"


def _request_planner_output(
    *,
    llm_client: ToolCallingProvider | None,
    system_prompt: str,
    user_prompt: str,
) -> PlannerOutput:
    if llm_client is None:
        raise ValueError("planner_capability_missing: ToolCallingProvider is required for plan_node_impl")
    raw = llm_client.complete_with_tools(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=planner_tool_schemas(_planner_tool_registry()),
        tool_choice="auto",
    )
    canonical = require_canonical_single_tool_call_result(
        raw,
        error_prefix="plan_node_impl.invalid_planner_output",
    )
    planner_intent = str(canonical.get("planner_intent") or "").strip()
    return {
        "tool_call": dict(canonical["tool_call"]),
        "planner_intent": planner_intent[:160],
    }


def _build_planner_user_prompt(*, task_record: TaskRecord) -> str:
    mcp_capability_menu, _ = _build_mcp_capability_menu()
    return render_pdca_prompt(
        NEXT_STEP_USER_TEMPLATE,
        {
            "MCP_CAPABILITY_MENU": mcp_capability_menu,
            "MCP_LIVE_TOOLS_MENU": "",
            "INJECTED_GUIDANCE_BLOCK": "",
            "TASK_RECORD_JSON": json.dumps(task_record.to_dict(), ensure_ascii=False),
            "FACTS_SECTION": task_record.get_facts_md(),
            "PLAN_SECTION": task_record.get_plan_md(),
            "ACCEPTANCE_CRITERIA_SECTION": task_record.get_acceptance_criteria_md(),
            "MEMORY_FACTS_SECTION": task_record.get_memory_facts_md(),
            "TOOL_CALL_HISTORY_SECTION": task_record.get_tool_call_history_md(),
        },
    )


def _append_planner_output_to_record(task_record: TaskRecord, *, planner_output: PlannerOutput) -> None:
    tool_call = planner_output["tool_call"]
    tool_name = str(tool_call.get("tool_name") or "").strip() or "(unknown)"
    args = tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {}
    args_summary = _compact_json(args)
    planner_intent = str(planner_output.get("planner_intent") or "").strip()
    line = f"{tool_name} args={args_summary}"
    if planner_intent:
        line = f"{line} intent={planner_intent}"
    task_record.append_plan_line(line)


def _log_planner_output(
    *,
    task_record: TaskRecord,
    planner_output: PlannerOutput,
    logger: Any,
    log_task_event: Any,
) -> None:
    tool_call = planner_output["tool_call"]
    tool_name = str(tool_call.get("tool_name") or "").strip()
    logger.info(
        "task_mode planner proposed tool=%s task_id=%s",
        tool_name,
        str(task_record.task_id or ""),
    )
    log_task_event(
        logger=logger,
        state={
            "correlation_id": task_record.correlation_id or None,
            "channel_type": None,
            "actor_person_id": task_record.user_id,
        },
        node="next_step_node",
        event="graph.next_step.planner_output",
        task_record=task_record,
        cycle_index=0,
        tool_name=tool_name,
        planner_intent=str(planner_output.get("planner_intent") or ""),
    )


@lru_cache(maxsize=1)
def _planner_tool_registry() -> ToolRegistry:
    return build_default_tool_registry()


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


def _compact_json(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)[:500]
    except Exception:
        return str(value)[:500]
