from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from alphonse.agent.tools.registry2 import ToolRegistry
from alphonse.agent.tools.spec import ToolSpec


def pretty_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def build_render_context(specs: list[ToolSpec]) -> list[dict[str, Any]]:
    tools: list[dict[str, Any]] = []
    for spec in specs:
        tools.append(
            {
                "key": spec.key,
                "description": spec.description.strip(),
                "domain_tags": spec.domain_tags,
                "safety_level": spec.safety_level.value,
                "requires_confirmation": spec.requires_confirmation,
                "input_schema_pretty": pretty_json(spec.input_schema),
                "examples": spec.examples,
            }
        )
    return tools


def render_tool_catalog(registry: ToolRegistry, template_dir: str | Path) -> str:
    env = Environment(loader=FileSystemLoader(str(template_dir)), autoescape=False)
    template = env.get_template("planning.tools.md.j2")
    specs = registry.specs_for_catalog()
    tools_context = build_render_context(specs)
    return str(template.render(tools=tools_context)).strip()

