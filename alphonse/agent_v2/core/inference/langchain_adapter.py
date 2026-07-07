"""LangChain schema adapters for Alphonse v2 tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool

from alphonse.agent_v2.core.tools.registry import ToolDefinition


def to_langchain_tool(definition: ToolDefinition) -> StructuredTool:
    """Convert one Alphonse tool definition into a LangChain StructuredTool."""
    descriptor = definition.descriptor_with_schema()

    def invoke_tool(*args: Any, **kwargs: Any) -> Any:
        arguments: dict[str, Any] = {}
        if args and isinstance(args[0], dict):
            arguments.update(args[0])
        arguments.update(kwargs)
        return definition.callable(arguments)

    invoke_tool.__name__ = descriptor.name
    invoke_tool.__doc__ = descriptor.description or f"Alphonse tool {descriptor.name}."

    kwargs: dict[str, Any] = {
        "func": invoke_tool,
        "name": descriptor.name,
        "description": descriptor.description or f"Alphonse tool {descriptor.name}.",
        "metadata": {"tool_id": descriptor.tool_id, **dict(descriptor.metadata)},
        "args_schema": descriptor.argument_schema
        or {"type": "object", "additionalProperties": True},
        "infer_schema": False,
    }
    return StructuredTool.from_function(**kwargs)


def to_langchain_tools(definitions: tuple[ToolDefinition, ...]) -> tuple[StructuredTool, ...]:
    """Convert multiple Alphonse tool definitions into LangChain tools."""
    return tuple(to_langchain_tool(definition) for definition in definitions if definition.enabled)
