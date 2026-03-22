from __future__ import annotations

from alphonse.agent.tools.registry2 import ToolRegistry
from alphonse.agent.tools.registry2 import planner_tool_schemas_from_specs
from alphonse.agent.tools.spec import ToolSpec


def _spec(
    canonical_name: str,
    *,
    expose_in_schemas: bool = True,
    visible_to_agent: bool = True,
    deprecated: bool = False,
) -> ToolSpec:
    return ToolSpec(
        canonical_name=canonical_name,
        summary=f"{canonical_name} summary",
        description=f"{canonical_name} long description",
        input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        output_schema={"type": "object", "additionalProperties": True},
        expose_in_schemas=expose_in_schemas,
        visible_to_agent=visible_to_agent,
        deprecated=deprecated,
    )


def test_registry_get_uses_canonical_name() -> None:
    registry = ToolRegistry()
    spec = _spec("send_message")
    registry.register(spec)
    assert registry.get("send_message") is spec


def test_planner_schemas_exclude_not_visible_to_agent() -> None:
    registry = ToolRegistry()
    registry.register(_spec("tool_a", visible_to_agent=False))
    schemas = planner_tool_schemas_from_specs(registry)
    assert schemas == []


def test_planner_schemas_exclude_deprecated() -> None:
    registry = ToolRegistry()
    registry.register(_spec("tool_a", deprecated=True))
    schemas = planner_tool_schemas_from_specs(registry)
    assert schemas == []


def test_planner_schemas_exclude_when_not_exposed() -> None:
    registry = ToolRegistry()
    registry.register(_spec("tool_a", expose_in_schemas=False))
    schemas = planner_tool_schemas_from_specs(registry)
    assert schemas == []


def test_planner_schema_description_uses_summary() -> None:
    registry = ToolRegistry()
    registry.register(
        ToolSpec(
            canonical_name="tool_a",
            summary="short summary",
            description="long description that should not be used",
            input_schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
            output_schema={"type": "object", "additionalProperties": True},
        )
    )
    schemas = planner_tool_schemas_from_specs(registry)
    assert len(schemas) == 1
    fn = schemas[0]["function"]
    assert fn["name"] == "tool_a"
    assert fn["description"] == "short summary"

