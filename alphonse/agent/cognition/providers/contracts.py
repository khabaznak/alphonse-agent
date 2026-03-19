from __future__ import annotations

from typing import Any, Literal, TypedDict


class CanonicalToolCall(TypedDict):
    kind: Literal["call_tool"]
    tool_name: str
    args: dict[str, Any]


class CanonicalCompleteWithToolsResult(TypedDict, total=False):
    content: str
    tool_call: CanonicalToolCall
    planner_intent: str


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
