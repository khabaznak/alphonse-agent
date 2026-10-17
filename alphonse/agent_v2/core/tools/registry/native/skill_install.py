"""Native tool for authoring and installing instruction-only skills."""
from __future__ import annotations

from typing import Any, Callable

from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.skills import SkillStore

SKILL_INSTALL_TOOL_ID = "native.skill_install"
SKILL_INSTALL_ARGUMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "name": {"type": "string", "minLength": 1, "maxLength": 64, "pattern": "^[a-z0-9]+(?:-[a-z0-9]+)*$"},
        "description": {"type": "string", "minLength": 1, "maxLength": 1024},
        "instructions": {"type": "string", "minLength": 1},
    },
    "required": ["name", "description", "instructions"],
}


def build_skill_install_tool_definition(
    store: SkillStore,
    *,
    is_admin: Callable[[str], bool] | None = None,
) -> ToolDefinition:
    schema = dict(SKILL_INSTALL_ARGUMENT_SCHEMA)
    descriptor = ToolDescriptor(
        SKILL_INSTALL_TOOL_ID,
        "skill_install",
        ToolKind.NATIVE,
        "Install a newly authored reusable Alphonse skill as a SKILL.md instruction package. Use only after the requester has reviewed and approved the proposed skill. The skill cannot grant tools, permissions, or executable capabilities.",
        schema,
        ("skill_management",),
        ("native", "skills", "install"),
        read_only=False,
    )
    return ToolDefinition(
        descriptor,
        lambda arguments, context=None: execute_skill_install(
            arguments, context=context, store=store, is_admin=is_admin,
        ),
        schema,
        accepts_context=True,
    )


def execute_skill_install(
    arguments: dict[str, Any],
    *,
    context: ToolExecutionContext | None,
    store: SkillStore,
    is_admin: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    if context is None:
        raise ValueError("skill_install_context_required")
    actor = str(context.task.user or "").strip()
    if not actor or not (is_admin and is_admin(actor)):
        raise PermissionError("skill_manager_required")
    skill = store.create_skill(
        str(arguments.get("name") or ""),
        str(arguments.get("description") or ""),
        str(arguments.get("instructions") or ""),
    )
    return {
        "skill": skill.candidate(),
        "message": f'Installed skill "{skill.name}" ({skill.skill_id}).',
    }
