from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

import alphonse.agent.tools.registry as runtime_registry
from alphonse.agent.tools.base import ToolDefinition
from alphonse.agent.tools.registry import planner_canonical_tool_names
from alphonse.agent.tools.registry import planner_tool_schemas
from alphonse.agent.tools.registry import ToolRegistry
from alphonse.agent.tools.spec import ToolSpec


@dataclass
class _EchoExecutor:
    value: str = "ok"
    canonical_name: str = "echo_tool"
    capability: str = "test"

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
    real_specs = runtime_registry._default_specs()
    filtered = [spec for spec in real_specs if spec.canonical_name != "get_time"]
    monkeypatch.setattr(runtime_registry, "_default_specs", lambda: filtered)
    with pytest.raises(ValueError, match="tool_spec_missing:get_time"):
        runtime_registry.build_default_tool_registry()


def test_build_default_tool_registry_fails_when_executor_missing_canonical_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingCanonical:
        capability = "test"

        def execute(self, **kwargs: Any) -> dict[str, Any]:
            _ = kwargs
            return {"output": {}, "exception": None, "metadata": {"tool": "x"}}

    monkeypatch.setattr(runtime_registry, "_build_runtime_executors", lambda **_: [_MissingCanonical()])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tool_executor_missing_canonical_name"):
        runtime_registry.build_default_tool_registry()


def test_build_default_tool_registry_fails_when_executor_missing_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingCapability:
        canonical_name = "get_time"

        def execute(self, **kwargs: Any) -> dict[str, Any]:
            _ = kwargs
            return {"output": {}, "exception": None, "metadata": {"tool": "x"}}

    monkeypatch.setattr(runtime_registry, "_build_runtime_executors", lambda **_: [_MissingCapability()])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tool_executor_missing_capability:get_time"):
        runtime_registry.build_default_tool_registry()


def test_build_default_tool_registry_fails_when_duplicate_executor_canonical_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExecA:
        canonical_name = "get_time"
        capability = "context"

        def execute(self, **kwargs: Any) -> dict[str, Any]:
            _ = kwargs
            return {"output": {}, "exception": None, "metadata": {"tool": "a"}}

    class _ExecB:
        canonical_name = "get_time"
        capability = "context"

        def execute(self, **kwargs: Any) -> dict[str, Any]:
            _ = kwargs
            return {"output": {}, "exception": None, "metadata": {"tool": "b"}}

    monkeypatch.setattr(runtime_registry, "_build_runtime_executors", lambda **_: [_ExecA(), _ExecB()])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tool_executor_duplicate_canonical_name:get_time"):
        runtime_registry.build_default_tool_registry()


def test_build_default_tool_registry_fails_when_executor_spec_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Exec:
        canonical_name = "get_time"
        capability = "context"

        def execute(self, **kwargs: Any) -> dict[str, Any]:
            _ = kwargs
            return {"output": {}, "exception": None, "metadata": {"tool": "x"}}

    monkeypatch.setattr(runtime_registry, "_build_runtime_executors", lambda **_: [_Exec()])  # type: ignore[arg-type]
    monkeypatch.setattr(
        runtime_registry,
        "_require_spec",
        lambda spec_by_name, canonical_name: _spec("wrong_name"),
    )
    with pytest.raises(ValueError, match="tool_executor_spec_mismatch:get_time:wrong_name"):
        runtime_registry.build_default_tool_registry()


def test_planner_schemas_exclude_hidden_tools() -> None:
    registry = ToolRegistry()
    hidden = ToolSpec(
        canonical_name="hidden_tool",
        summary="hidden summary",
        description="hidden description",
        input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        output_schema={"type": "object", "additionalProperties": True},
        visible_to_agent=False,
    )
    registry.register(ToolDefinition(spec=hidden, executor=_EchoExecutor()))
    assert planner_tool_schemas(registry) == []


def test_planner_schemas_use_canonical_name_once_for_aliases() -> None:
    registry = ToolRegistry()
    spec = ToolSpec(
        canonical_name="send_message",
        summary="send message summary",
        description="send message description",
        input_schema={"type": "object", "properties": {"To": {"type": "string"}}, "required": ["To"], "additionalProperties": False},
        output_schema={"type": "object", "additionalProperties": True},
        aliases=["sendMessage", "send_message_alias"],
    )
    registry.register(ToolDefinition(spec=spec, executor=_EchoExecutor()))
    names = planner_canonical_tool_names(registry)
    assert names == ["send_message"]
    schemas = planner_tool_schemas(registry)
    assert len(schemas) == 1
    fn = schemas[0]["function"]
    assert fn["name"] == "send_message"
    assert fn["description"] == "send message summary"
    assert fn["parameters"] == spec.input_schema
