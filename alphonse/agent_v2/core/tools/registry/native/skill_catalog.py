"""Read-only, actor-scoped discovery of installed skill metadata.

Discovery is not activation or a relevance judgment. In particular it does
not authorize loading package instructions, resources, or executing code.
"""
from __future__ import annotations

import json
from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.skill_resources import authorized, private_legacy_package

SKILL_CATALOG_TOOL_ID = 'native.skill_catalog'
MAX_SKILL_CATALOG_ITEMS = 100
_SCHEMA = {'type': 'object', 'properties': {}, 'additionalProperties': False}


def execute_skill_catalog(arguments: dict[str, Any], *, context: Any) -> dict[str, Any]:
    """Return a complete set of safe metadata visible to the current actor."""
    if arguments != {} or context is None:
        raise ValueError('skill_catalog_arguments_invalid')
    task = getattr(context, 'task', None)
    actor = str(getattr(task, 'user', '') or '').strip()
    store = getattr(context, 'skill_store', None)
    resources = getattr(context, 'artifact_resources', None)
    cancelled = getattr(context, 'cancellation_checker', None)
    if (not actor or store is None or resources is None
            or not callable(getattr(store, 'list_skills', None))
            or not callable(getattr(store, 'can_use', None))
            or not callable(getattr(resources, 'access_records', None))
            or getattr(task, 'status', None) == 'cancelled'
            or not callable(cancelled) or cancelled() is not False):
        raise PermissionError('skill_catalog_unavailable')
    visible = []
    selected = []
    skills = store.list_skills()
    # The store intentionally skips invalid/unreadable packages.  Returning
    # the remainder as a complete installed-skill list would hide that gap;
    # don't disclose diagnostic paths or private package details either.
    if getattr(store, 'diagnostics', None):
        raise ValueError('skill_catalog_incomplete')
    for skill in skills:
        if not authorized(store, skill.skill_id, actor):
            continue
        if private_legacy_package(skill, actor, resources):
            continue
        selected.append(skill)
        visible.append({'skill_id': skill.skill_id, 'name': skill.name, 'description': skill.description})
        if len(visible) > MAX_SKILL_CATALOG_ITEMS:
            raise ValueError('skill_catalog_too_large')
    # Re-check access at completion, and refuse partial or changed visibility.
    if cancelled() is not False or any(
        not authorized(store, skill.skill_id, actor) or private_legacy_package(skill, actor, resources)
        for skill in selected
    ):
        raise PermissionError('skill_catalog_unavailable')
    return {'schema_version': 1, 'status': 'found' if visible else 'empty', 'skills': visible}


def validate_skill_catalog_observation(actor: str, tool_id: str, arguments: dict[str, Any],
                                       payload: dict[str, Any], *, context: Any) -> bool:
    """Re-read visibility and metadata before exposing or recovering evidence."""
    try:
        if (tool_id != SKILL_CATALOG_TOOL_ID or not actor
                or actor != getattr(getattr(context, 'task', None), 'user', None)
                or not isinstance(payload, dict)):
            return False
        current = execute_skill_catalog(arguments, context=context)
        return json.dumps(current, sort_keys=True, allow_nan=False) == json.dumps(payload, sort_keys=True, allow_nan=False)
    except Exception:
        return False


def build_skill_catalog_tool_definition() -> ToolDefinition:
    descriptor = ToolDescriptor(
        SKILL_CATALOG_TOOL_ID, 'skill_catalog', ToolKind.NATIVE,
        'List installed skill names and descriptions visible to the current owner. '
        'This is discovery only; it does not select or activate a skill, load its '
        'instructions, authorize actions, or guarantee an external capability is ready.',
        _SCHEMA, ('skill_discovery',), ('native', 'skills'), read_only=True,
    )
    return ToolDefinition(descriptor, execute_skill_catalog, _SCHEMA, accepts_context=True)
