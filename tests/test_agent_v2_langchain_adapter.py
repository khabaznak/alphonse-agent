from __future__ import annotations

from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolKind
from alphonse.agent_v2.core.inference.langchain_adapter import to_langchain_tool
from alphonse.agent_v2.core.tools.registry import ToolDefinition


def test_langchain_adapter_converts_registry_tool_to_structured_tool() -> None:
    tool = to_langchain_tool(_definition())

    assert tool.name == "write_file"
    assert tool.description == "Writes a file"
    assert tool.metadata["tool_id"] == "tool-1"


def test_langchain_adapter_invokes_original_tool_callable() -> None:
    tool = to_langchain_tool(_definition())

    assert tool.invoke({"path": "a.txt"}) == {"path": "a.txt"}


def _definition() -> ToolDefinition:
    return ToolDefinition(
        descriptor=ToolDescriptor(
            tool_id="tool-1",
            name="write_file",
            kind=ToolKind.NATIVE,
            description="Writes a file",
        ),
        callable=lambda arguments: {"path": arguments["path"]},
        enabled=True,
    )
