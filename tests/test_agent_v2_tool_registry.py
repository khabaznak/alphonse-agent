from __future__ import annotations

import pytest

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.core.tools.registry import ToolExposurePolicy


def test_tool_registry_registers_lists_gets_and_executes_tool() -> None:
    registry = InMemoryToolRegistry()
    registry.register(_definition())

    assert registry.list()[0].tool_id == "tool-1"
    assert registry.get("write_file").tool_id == "tool-1"
    assert registry.get("tool-1").name == "write_file"
    assert registry.execute("tool-1", {"path": "a.txt"}) == {"path": "a.txt"}


def test_tool_registry_does_not_expose_disabled_tool() -> None:
    registry = InMemoryToolRegistry()
    registry.register(_definition(enabled=False))

    assert registry.list() == ()
    assert registry.get("tool-1") is None
    assert ToolExposurePolicy().select_tools(registry=registry) == ()


def test_tool_registry_unknown_tool_execution_raises_controlled_exception() -> None:
    registry = InMemoryToolRegistry()

    with pytest.raises(KeyError, match="tool_not_found"):
        registry.execute("missing", {})


def test_tool_registry_preserves_argument_schema_on_descriptor() -> None:
    registry = InMemoryToolRegistry()
    registry.register(_definition())

    descriptor = registry.get("tool-1")

    assert descriptor is not None
    assert descriptor.argument_schema["properties"]["path"]["type"] == "string"


def _definition(*, enabled: bool = True) -> ToolDefinition:
    return ToolDefinition(
        descriptor=ToolDescriptor(
            tool_id="tool-1",
            name="write_file",
            kind=ToolKind.NATIVE,
            description="Writes a file",
        ),
        argument_schema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
        callable=lambda arguments: {"path": arguments["path"]},
        enabled=enabled,
    )
