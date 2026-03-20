from __future__ import annotations

from typing import Any, Literal, Protocol, TypedDict, runtime_checkable


class CanonicalToolCall(TypedDict):
    kind: Literal["call_tool"]
    tool_name: str
    args: dict[str, Any]


class CanonicalCompleteWithToolsResult(TypedDict, total=False):
    content: str
    tool_call: CanonicalToolCall
    planner_intent: str


@runtime_checkable
class TextCompletionProvider(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        ...


@runtime_checkable
class ToolCallingProvider(TextCompletionProvider, Protocol):
    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> CanonicalCompleteWithToolsResult:
        ...


def require_text_completion_provider(obj: Any, *, source: str) -> TextCompletionProvider:
    candidate = obj
    complete = getattr(candidate, "complete", None)
    if not callable(complete):
        raise ValueError(
            f"provider_contract_error:text_completion_missing source={source} required=complete(system_prompt,user_prompt)"
        )
    return candidate


def require_tool_calling_provider(obj: Any, *, source: str) -> ToolCallingProvider:
    candidate = require_text_completion_provider(obj, source=source)
    complete_with_tools = getattr(candidate, "complete_with_tools", None)
    if not callable(complete_with_tools):
        raise ValueError(
            "provider_contract_error:tool_calling_missing "
            f"source={source} required=complete_with_tools(messages,tools,tool_choice)"
        )
    return candidate


def require_canonical_single_tool_call_result(
    raw: Any,
    *,
    error_prefix: str,
) -> CanonicalCompleteWithToolsResult:
    source = raw if isinstance(raw, dict) else {}
    tool_call = source.get("tool_call")
    if not isinstance(tool_call, dict):
        raise ValueError(f"{error_prefix}: missing canonical tool_call")
    if str(tool_call.get("kind") or "").strip() != "call_tool":
        raise ValueError(f"{error_prefix}: invalid tool_call.kind")
    tool_name = str(tool_call.get("tool_name") or "").strip()
    if not tool_name:
        raise ValueError(f"{error_prefix}: missing tool_call.tool_name")
    args = tool_call.get("args")
    if not isinstance(args, dict):
        raise ValueError(f"{error_prefix}: invalid tool_call.args")

    out: CanonicalCompleteWithToolsResult = {
        "tool_call": {
            "kind": "call_tool",
            "tool_name": tool_name,
            "args": dict(args),
        }
    }
    content = source.get("content")
    if isinstance(content, str):
        out["content"] = content

    planner_intent = source.get("planner_intent")
    if planner_intent is not None:
        if not isinstance(planner_intent, str):
            raise ValueError(f"{error_prefix}: invalid planner_intent")
        text = planner_intent.strip()
        if text:
            out["planner_intent"] = text[:160]
    return out
