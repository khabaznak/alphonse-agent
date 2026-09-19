"""Bounded read-only recall from the current project's archived memory."""

from __future__ import annotations

from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

SEARCH_MEMORY_TOOL_ID = "native.search_memory"
SEARCH_MEMORY_TOOL_NAME = "search_memory"


def execute_search_memory(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query: raise ValueError("memory_search_query_required")
    if context is None or context.memory is None: raise RuntimeError("memory_unavailable")
    task = context.task
    search = getattr(context.memory, "search", None)
    if not callable(search): raise RuntimeError("memory_search_unavailable")
    result = str(search(user_id=str(task.user or ""), project_id=str(task.project_id or ""), query=query) or "")
    return {"query": query, "project_id": task.project_id, "matches_markdown": result}


def build_search_memory_tool_definition() -> ToolDefinition:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Words or phrase to find in closed sessions and legacy project memory."}},
        "required": ["query"],
        "additionalProperties": False,
    }
    return ToolDefinition(
        descriptor=ToolDescriptor(
            tool_id=SEARCH_MEMORY_TOOL_ID,
            name=SEARCH_MEMORY_TOOL_NAME,
            kind=ToolKind.NATIVE,
            description="Search archived memory in the current project and return bounded cited excerpts.",
            argument_schema=schema,
            capabilities=("memory", "search"),
            read_only=True,
        ),
        callable=execute_search_memory,
        argument_schema=schema,
        accepts_context=True,
    )
