"""Provider-independent in-memory tool registry for Alphonse v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from alphonse.agent_v2.core.core import ToolDescriptor


@dataclass(frozen=True)
class ToolDefinition:
    """Executable tool registered with Alphonse."""

    descriptor: ToolDescriptor
    callable: Callable[[dict[str, Any]], Any]
    argument_schema: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def descriptor_with_schema(self) -> ToolDescriptor:
        schema = dict(self.argument_schema or self.descriptor.argument_schema or {})
        return ToolDescriptor(
            tool_id=self.descriptor.tool_id,
            name=self.descriptor.name,
            kind=self.descriptor.kind,
            description=self.descriptor.description,
            argument_schema=schema,
            capabilities=tuple(self.descriptor.capabilities),
            tags=tuple(self.descriptor.tags),
            metadata=dict(self.descriptor.metadata),
        )


class InMemoryToolRegistry:
    """Simple canonical registry for native and artifact tools."""

    def __init__(self) -> None:
        self._definitions_by_id: dict[str, ToolDefinition] = {}
        self._ids_by_name: dict[str, str] = {}

    def register(self, tool: ToolDefinition | ToolDescriptor) -> None:
        if isinstance(tool, ToolDescriptor):
            raise TypeError("tool_definition_required")
        descriptor = tool.descriptor
        tool_id = _required_identifier(descriptor.tool_id, "tool_id")
        name = _required_identifier(descriptor.name, "name")
        self._definitions_by_id[tool_id] = tool
        self._ids_by_name[name] = tool_id

    def get(self, name: str) -> ToolDescriptor | None:
        definition = self.get_definition(name)
        if definition is None or not definition.enabled:
            return None
        return definition.descriptor_with_schema()

    def get_definition(self, name_or_id: str) -> ToolDefinition | None:
        key = str(name_or_id or "").strip()
        if not key:
            return None
        tool_id = key if key in self._definitions_by_id else self._ids_by_name.get(key)
        return self._definitions_by_id.get(tool_id or "")

    def list(self) -> tuple[ToolDescriptor, ...]:
        return tuple(
            definition.descriptor_with_schema()
            for definition in self._definitions_by_id.values()
            if definition.enabled
        )

    def execute(self, tool_id: str, arguments: dict[str, Any]) -> Any:
        definition = self.get_definition(tool_id)
        if definition is None:
            raise KeyError(f"tool_not_found: {tool_id}")
        if not definition.enabled:
            raise PermissionError(f"tool_disabled: {tool_id}")
        return definition.callable(dict(arguments or {}))


class ToolExposurePolicy:
    """First-pass exposure policy for tools made visible to a model."""

    def select_tools(
        self,
        *,
        registry: InMemoryToolRegistry,
        project_id: str = "",
        user: str | None = None,
        task: object | None = None,
        model_profile: object | None = None,
    ) -> tuple[ToolDescriptor, ...]:
        _ = project_id
        _ = user
        _ = task
        _ = model_profile
        return registry.list()


def _required_identifier(value: str, field_name: str) -> str:
    rendered = str(value or "").strip()
    if not rendered:
        raise ValueError(f"{field_name}_required")
    return rendered
