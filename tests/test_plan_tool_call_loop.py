from __future__ import annotations

from typing import Any

from alphonse.agent.cortex.nodes.plan import plan_node
from alphonse.agent.tools.registry import build_default_tool_registry


class _ToolCallLlm:
    supports_tool_calls = True

    def __init__(self) -> None:
        self.calls = 0

    def complete_with_tools(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        tool_choice: str = "auto",
    ) -> dict[str, Any]:
        _ = messages
        _ = tools
        _ = tool_choice
        self.calls += 1
        if self.calls == 1:
            return {
                "content": "",
                "tool_calls": [
                    {"id": "tc-1", "name": "getTime", "arguments": {}},
                ],
                "assistant_message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "tc-1",
                            "type": "function",
                            "function": {"name": "getTime", "arguments": "{}"},
                        }
                    ],
                },
            }
        return {"content": "Done.", "tool_calls": [], "assistant_message": {"role": "assistant", "content": "Done."}}


def _run_capability_gap_tool(state: dict[str, Any], llm_client: Any, reason: str) -> dict[str, Any]:
    _ = state
    _ = llm_client
    return {"messages": [], "plans": [{"plan_type": "CAPABILITY_GAP", "reason": reason}]}


def test_plan_node_uses_native_tool_call_loop_when_supported() -> None:
    state = {
        "last_user_message": "What time is it?",
        "timezone": "UTC",
        "locale": "en-US",
        "tone": "friendly",
        "address_style": "tu",
        "channel_type": "telegram",
        "channel_target": "123",
        "correlation_id": "corr-tool-call",
    }
    result = plan_node(
        state,
        llm_client=_ToolCallLlm(),
        tool_registry=build_default_tool_registry(),
        discover_plan=lambda **_: {"messages": [], "plans": []},
        format_available_abilities=lambda: "- getTime() -> current time",
        run_capability_gap_tool=_run_capability_gap_tool,
    )
    assert result.get("response_text") == "Done."
    ability = result.get("ability_state")
    assert isinstance(ability, dict)
    assert ability.get("kind") == "tool_calls"
