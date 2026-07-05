"""Plan node for the v2 PDCA graph."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from jinja2 import Environment
from jinja2 import FileSystemLoader
from jinja2 import select_autoescape

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.intelligence.task_state import TaskState

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext

_TEMPLATE_DIR = Path(__file__).resolve().parents[2] / "templates"


def plan_node(task: TaskState, context: CoreLoopContext | None = None) -> TaskState:
    """Prepare one future tool call without executing it."""
    tools = _tools_from_context(context)
    prompt = _render_tool_call_plan_prompt(task, tools)
    task.metadata["tool_call_plan_prompt"] = prompt

    planned_tool_call = _normalize_tool_call(_call_tool_planning_llm(prompt), tools)
    if planned_tool_call is None:
        task.metadata["tool_call_planning_llm_stubbed"] = True
        task.append_update("Plan prepared one-tool-call prompt; LLM execution is stubbed.")
        return task

    task.metadata["tool_call_planning_llm_stubbed"] = False
    task.metadata["planned_tool_call"] = planned_tool_call
    task.plan_md = json.dumps(planned_tool_call, indent=2, sort_keys=True)
    task.append_update("Plan selected the next tool call.")
    return task


def _render_tool_call_plan_prompt(task: TaskState, tools: tuple[ToolDescriptor, ...]) -> str:
    env = Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(default_for_string=False),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("tool_call_plan_prompt.j2")
    return template.render(
        acceptance_criteria_md=task.acceptance_criteria_md,
        available_tools_md=_render_tools_md(tools),
        task_state_md=task.to_markdown_prompt(),
    ).strip()


def _render_tools_md(tools: tuple[ToolDescriptor, ...]) -> str:
    if not tools:
        return "- (none)"
    return "\n".join(
        f"- {tool.tool_id} | {tool.name} | {tool.kind.value} | {tool.description or '(no description)'}"
        for tool in tools
    )


def _tools_from_context(context: CoreLoopContext | None) -> tuple[ToolDescriptor, ...]:
    if context is None or context.tools is None:
        return ()
    return tuple(context.tools.list())


def _call_tool_planning_llm(prompt: str) -> dict[str, Any] | None:
    """Stub for the future one-tool-call planning LLM call."""
    _ = prompt
    return None


def _normalize_tool_call(value: dict[str, Any] | None, tools: tuple[ToolDescriptor, ...]) -> dict[str, Any] | None:
    if value is None:
        return None
    tool_id = str(value.get("tool_id") or "").strip()
    tool_name = str(value.get("tool_name") or "").strip()
    arguments = value.get("arguments")
    if not tool_id or not tool_name or not isinstance(arguments, dict):
        return None
    if tools and tool_id not in {tool.tool_id for tool in tools}:
        return None
    return {
        "tool_id": tool_id,
        "tool_name": tool_name,
        "arguments": dict(arguments),
    }
