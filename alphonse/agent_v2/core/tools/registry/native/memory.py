"""Bounded read-only recall from the user's conversation and authorized memory."""

from __future__ import annotations

import hashlib

from typing import Any

from alphonse.agent_v2.core.memory.contracts import AccessScope, ContextItem, SourceReference, RecallResult, conversation_item, visible_unique

from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

SEARCH_MEMORY_TOOL_ID = "native.search_memory"
SEARCH_MEMORY_TOOL_NAME = "search_memory"


def execute_search_memory(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    query = str(arguments.get("query") or "").strip()
    if not query: raise ValueError("memory_search_query_required")
    if context is None or context.memory is None: raise RuntimeError("memory_unavailable")
    task = context.task
    conversation_store = getattr(context, "conversation_store", None)
    conversation_search = getattr(conversation_store, "search", None)
    conversation_failed = False
    if callable(conversation_search):
        try:
            events = conversation_search(owner_user_id=str(task.user or ""), query=query, limit=8)
        except Exception:
            conversation_failed = True
            events = []
        items = visible_unique([item for event in events
                                if (item := conversation_item(event, str(task.user or ""), max_chars=900)) is not None],
                               str(task.user or ""))
        if items:
            events_by_id = {f"conversation:{event.event_id}": event for event in events
                            if conversation_item(event, str(task.user or "")) is not None}
            excerpts = "\n".join(
                f"- [{item.source.observed_at or 'time unknown'}; project={events_by_id[item.source.source_id].project_id or 'conversation'}; {item.source.role}] "
                + ("[legacy import; unverified provenance] " if item.source.provenance != "recorded" else "")
                + item.content for item in items
            )
            return {"query": query, "project_id": task.project_id, "matches_markdown": excerpts,
                    "recall": RecallResult(query, "found", tuple(items)).to_dict()}
    search = getattr(context.memory, "search", None)
    if not callable(search): raise RuntimeError("memory_search_unavailable")
    result = str(search(user_id=str(task.user or ""), project_id=str(task.project_id or ""), query=query) or "")
    # Legacy task-audit search is explicitly weaker evidence, not canonical messages.
    items = ()
    if result and result not in {"- (no matches)", "- (no searchable terms)"}:
        source_id = "legacy-ledger:" + hashlib.sha256(
            (str(task.user) + "\0" + str(task.project_id) + "\0" + result).encode()
        ).hexdigest()
        reference = SourceReference(source_id, "legacy_ledger", AccessScope(str(task.user)),
                                    provenance="legacy_unverified")
        items = (ContextItem(reference, result, "audit"),)
    return {"query": query, "project_id": task.project_id, "matches_markdown": result,
            "recall": RecallResult(query, "found" if items else ("unavailable" if conversation_failed else "empty"), items).to_dict()}


def build_search_memory_tool_definition() -> ToolDefinition:
    schema = {
        "type": "object",
        "properties": {"query": {"type": "string", "description": "Words or phrase to find in the user's conversation history and archived memory."}},
        "required": ["query"],
        "additionalProperties": False,
    }
    return ToolDefinition(
        descriptor=ToolDescriptor(
            tool_id=SEARCH_MEMORY_TOOL_ID,
            name=SEARCH_MEMORY_TOOL_NAME,
            kind=ToolKind.NATIVE,
            description=(
                "Search the user's conversation history across projects and bounded archived memory in the "
                "current task's authorized scope; return cited excerpts."
            ),
            argument_schema=schema,
            capabilities=("memory", "search"),
            read_only=True,
        ),
        callable=execute_search_memory,
        argument_schema=schema,
        accepts_context=True,
    )
