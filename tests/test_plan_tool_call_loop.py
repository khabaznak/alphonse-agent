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


class _LoopingTimeToolCallLlm:
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
        return {
            "content": "",
            "tool_calls": [{"id": f"tc-{self.calls}", "name": "getTime", "arguments": {}}],
            "assistant_message": {"role": "assistant", "content": ""},
        }


class _RefusalThenToolCallLlm:
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
                "content": "I tried but the required tool isn't available here.",
                "tool_calls": [],
                "assistant_message": {"role": "assistant", "content": "I tried but the required tool isn't available here."},
            }
        return {
            "content": "",
            "tool_calls": [{"id": "tc-1", "name": "getTime", "arguments": {}}],
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


class _TextPlanOnlyLlm:
    supports_tool_calls = True

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
        return {
            "content": "**Plan**\n\n- Use `getTime` to fetch my current time",
            "tool_calls": [],
            "assistant_message": {"role": "assistant", "content": "**Plan**\n\n- Use `getTime` to fetch my current time"},
        }

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        _ = system_prompt
        _ = user_prompt
        return '{"tool":"getTime","parameters":{}}'


class _JsonExecutionPlanLlm:
    supports_tool_calls = True

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
        return {
            "content": (
                '{"intention":"Provide the current time to the user",'
                '"confidence":"high",'
                '"execution_plan":[{"tool":"getTime","parameters":{}}]}'
            ),
            "tool_calls": [],
            "assistant_message": {
                "role": "assistant",
                "content": (
                    '{"intention":"Provide the current time to the user",'
                    '"confidence":"high",'
                    '"execution_plan":[{"tool":"getTime","parameters":{}}]}'
                ),
            },
        }


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
        format_available_abilities=lambda: "- getTime() -> current time",
        run_capability_gap_tool=_run_capability_gap_tool,
    )
    assert isinstance(result.get("response_text"), str)
    assert result.get("response_text")
    ability = result.get("ability_state")
    assert isinstance(ability, dict)
    assert ability.get("kind") == "tool_calls"


def test_plan_node_short_circuits_terminal_tool_loop() -> None:
    llm = _LoopingTimeToolCallLlm()
    state = {
        "last_user_message": "What time is it?",
        "timezone": "UTC",
        "locale": "en-US",
        "tone": "friendly",
        "address_style": "tu",
        "channel_type": "telegram",
        "channel_target": "123",
        "correlation_id": "corr-terminal-short-circuit",
    }
    result = plan_node(
        state,
        llm_client=llm,
        tool_registry=build_default_tool_registry(),
        format_available_abilities=lambda: "- getTime() -> current time",
        run_capability_gap_tool=_run_capability_gap_tool,
    )
    assert llm.calls == 1
    assert isinstance(result.get("response_text"), str)
    assert result.get("response_text")
    assert not result.get("plans")


def test_plan_node_reports_tool_refusal_without_silent_repair() -> None:
    llm = _RefusalThenToolCallLlm()
    state = {
        "last_user_message": "What time is it?",
        "timezone": "UTC",
        "locale": "en-US",
        "tone": "friendly",
        "address_style": "tu",
        "channel_type": "telegram",
        "channel_target": "123",
        "correlation_id": "corr-force-time",
    }
    result = plan_node(
        state,
        llm_client=llm,
        tool_registry=build_default_tool_registry(),
        format_available_abilities=lambda: "- getTime() -> current time",
        run_capability_gap_tool=_run_capability_gap_tool,
    )
    assert llm.calls == 1
    plans = result.get("plans") if isinstance(result.get("plans"), list) else []
    assert plans
    first = plans[0] if isinstance(plans[0], dict) else {}
    assert first.get("reason") == "model_tool_refusal_no_tool_call"


def test_plan_node_translates_textual_plan_step_to_tool_execution() -> None:
    state = {
        "last_user_message": "claro! qué hora es contigo?",
        "timezone": "UTC",
        "locale": "es-MX",
        "tone": "friendly",
        "address_style": "tu",
        "channel_type": "telegram",
        "channel_target": "123",
        "correlation_id": "corr-text-plan-fallback",
    }
    result = plan_node(
        state,
        llm_client=_TextPlanOnlyLlm(),
        tool_registry=build_default_tool_registry(),
        format_available_abilities=lambda: "- getTime() -> current time",
        run_capability_gap_tool=_run_capability_gap_tool,
    )
    text = str(result.get("response_text") or "")
    assert text.startswith("Son las ")


def test_plan_node_routes_json_execution_plan_to_tool_execution() -> None:
    state = {
        "last_user_message": "Hola Alphonse, qué horas tienes?",
        "timezone": "UTC",
        "locale": "es-MX",
        "tone": "friendly",
        "address_style": "tu",
        "channel_type": "telegram",
        "channel_target": "123",
        "correlation_id": "corr-json-plan-fallback",
    }
    result = plan_node(
        state,
        llm_client=_JsonExecutionPlanLlm(),
        tool_registry=build_default_tool_registry(),
        format_available_abilities=lambda: "- getTime() -> current time",
        run_capability_gap_tool=_run_capability_gap_tool,
    )
    text = str(result.get("response_text") or "")
    assert text.startswith("Son las ")
