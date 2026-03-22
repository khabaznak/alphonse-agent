from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import alphonse.agent.tools.registry as runtime_registry
from alphonse.agent.tools.base import ToolDefinition
from alphonse.agent.tools.registry import ToolRegistry
from alphonse.agent.tools.spec import ToolSpec


@dataclass
class _EchoExecutor:
    value: str = "ok"

    def execute(self, **kwargs: Any) -> dict[str, Any]:
        _ = kwargs
        return {
            "output": {"value": self.value},
            "exception": None,
            "metadata": {"tool": "echo"},
        }


def _spec(name: str, *, aliases: list[str] | None = None) -> ToolSpec:
    return ToolSpec(
        canonical_name=name,
        summary=f"{name} summary",
        description=f"{name} description",
        input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        output_schema={"type": "object", "additionalProperties": True},
        aliases=list(aliases or []),
    )


def test_registry_registers_tool_definition_only() -> None:
    registry = ToolRegistry()
    definition = ToolDefinition(spec=_spec("echo_tool"), executor=_EchoExecutor())
    registry.register(definition)
    assert registry.get("echo_tool") is definition


def test_registry_alias_lookup_returns_same_definition() -> None:
    registry = ToolRegistry()
    definition = ToolDefinition(spec=_spec("send_message", aliases=["sendMessage"]), executor=_EchoExecutor())
    registry.register(definition)
    assert registry.get("send_message") is definition
    assert registry.get("sendMessage") is definition


def test_registry_rejects_duplicate_alias_collision() -> None:
    registry = ToolRegistry()
    registry.register(ToolDefinition(spec=_spec("first", aliases=["alias_a"]), executor=_EchoExecutor()))
    with pytest.raises(ValueError, match="tool_key_collision:alias_a"):
        registry.register(ToolDefinition(spec=_spec("second", aliases=["alias_a"]), executor=_EchoExecutor()))


def test_tool_definition_invoke_normalizes_execute_result() -> None:
    definition = ToolDefinition(spec=_spec("echo_tool"), executor=_EchoExecutor(value="hello"))
    result = definition.invoke({"unused": True})
    assert result["exception"] is None
    assert result["output"] == {"value": "hello"}
    assert isinstance(result["metadata"], dict)


def test_build_default_tool_registry_fails_when_required_spec_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    real_specs = runtime_registry.build_planner_tool_registry().specs()
    filtered = [spec for spec in real_specs if spec.canonical_name != "get_time"]

    class _FakePlannerRegistry:
        def specs(self) -> list[ToolSpec]:
            return filtered

    monkeypatch.setattr(runtime_registry, "build_planner_tool_registry", lambda: _FakePlannerRegistry())
    with pytest.raises(ValueError, match="tool_spec_missing:get_time"):
        runtime_registry.build_default_tool_registry()
