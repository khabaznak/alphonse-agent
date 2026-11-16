"""Load one explicitly requested standard skill's instructions, without effects.

This is a narrow direct-owner path, not automatic skill relevance. It accepts
only an unambiguous explicit request naming an installed skill; Jev remains
responsible for relevance-based selection among skills. No script or resource
is executed or authorized by this read.
"""
from __future__ import annotations

import json
import re
from typing import Any

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.core.tools.registry.native.skill_catalog import execute_skill_catalog
from alphonse.agent_v2.skill_resources import authorized, package_revision, private_legacy_package

OWNER_NAMED_SKILL_TOOL_ID = 'native.owner_named_skill'
_SCHEMA = {'type': 'object', 'properties': {'skill_id': {'type': 'string', 'description': 'Exact package ID: skill: followed by the requested skill name, for example skill:meal-planning.'}},
           'required': ['skill_id'], 'additionalProperties': False}
_NAME = r'[a-z0-9]+(?:-[a-z0-9]+)*'
_DIRECT = re.compile(
    r'^\s*(?:(?:please\s+)|(?:(?:can|could|would)\s+you\s+(?:please\s+)?))?'
    r'(?:use|apply|follow)\s+(?:(?:the|my)\s+)?'
    r'(?:(?P<before>' + _NAME + r')\s+skill|skill\s+(?:named\s+)?(?P<after>' + _NAME + r'))'
    r'(?=$|[\s,.:;!?])', re.IGNORECASE)

def explicit_requested_name(goal: str) -> str:
    """Return the exact direct-request name, or empty for unsupported wording."""
    if not isinstance(goal, str):
        return ''
    match = _DIRECT.search(goal)
    # A later sentence or clause may revise the instruction. This narrow
    # direct-request path cannot resolve conversational corrections, so do
    # not activate from a multi-sentence or compound-punctuation request.
    if not match:
        return ''
    remainder = goal[match.end():].strip()
    if remainder.endswith(('.', '!')):
        remainder = remainder[:-1]
    if re.search(r'[.,!?;:\n\r]', remainder):
        return ''
    # Coordinating clauses may reverse or condition the initial request. The
    # bounded parser cannot resolve that relationship, so decline activation.
    if re.search(r'\b(?:and|but|or|yet|so)\b', remainder, re.IGNORECASE):
        return ''
    # Do not activate on questions, reversals, or contingent future requests.
    # This deliberately fails closed on ambiguous wording; it is not a
    # general natural-language intent classifier.
    if ('?' in goal or re.search(r"\b\w+n['’]t\b", goal[match.end():], re.IGNORECASE) or re.search(
            r'\b(?:no|not|never|unless|if|when|once|after|before|until|later|maybe|might)\b',
            goal[match.end():], re.IGNORECASE)):
        return ''
    return (match.group('before') or match.group('after')).casefold()


def explicit_requested_skill(goal: str, skills: list[dict[str, str]]) -> str:
    """Resolve a limited explicit request; never guess among partial names.

    This intentionally does not classify implicit need, translate names, or
    treat an incidental mention as a request. Unsupported wording returns an
    empty string and must not activate anything.
    """
    if not isinstance(skills, list):
        return ''
    name = explicit_requested_name(goal)
    if not name:
        return ''
    matches = []
    for row in skills:
        candidate, identity = row.get('name'), row.get('skill_id')
        if not isinstance(candidate, str) or not isinstance(identity, str) or not identity:
            continue
        if candidate.casefold() == name:
            matches.append(identity)
    return matches[0] if len(matches) == 1 else ''


def execute_owner_named_skill(arguments: dict[str, Any], *, context: Any) -> dict[str, Any]:
    """Load full instructions for an exact user-requested, authorized package."""
    if not isinstance(arguments, dict) or set(arguments) != {'skill_id'} or not isinstance(arguments['skill_id'], str):
        raise ValueError('owner_named_skill_arguments_invalid')
    task = getattr(context, 'task', None)
    actor = str(getattr(task, 'user', '') or '')
    store, resources = getattr(context, 'skill_store', None), getattr(context, 'artifact_resources', None)
    cancelled = getattr(context, 'cancellation_checker', None)
    if not actor or store is None or resources is None or not callable(cancelled) or cancelled() is not False:
        raise PermissionError('owner_named_skill_unavailable')
    name = explicit_requested_name(getattr(task, 'goal', ''))
    if not name or arguments['skill_id'] != 'skill:' + name:
        raise PermissionError('owner_named_skill_not_explicitly_requested')
    def unavailable():
        if cancelled() is not False:
            raise PermissionError('owner_named_skill_unavailable')
        return {'schema_version': 1, 'status': 'unavailable'}
    try:
        catalog = execute_skill_catalog({}, context=context)
    except ValueError as exc:
        if str(exc) not in {'skill_catalog_incomplete', 'skill_catalog_too_large'}:
            raise
        return unavailable()
    selected = explicit_requested_skill(getattr(task, 'goal', ''), catalog['skills'])
    if not selected or not authorized(store, selected, actor):
        return unavailable()
    if selected != arguments['skill_id']:
        raise PermissionError('owner_named_skill_not_explicitly_requested')
    skill = store.get(selected)
    if skill is None or skill.skill_id != selected or private_legacy_package(skill, actor, resources):
        return unavailable()
    try:
        revision = package_revision(skill)
    except (OSError, ValueError):
        return unavailable()
    if not isinstance(skill.instructions, str) or not skill.instructions.strip():
        return unavailable()
    # Never silently clip package instructions. Full instructions must fit the
    # same operational package limit as the standard skill loader.
    from alphonse.agent_v2.skills import MAX_SKILL_INSTRUCTIONS_LENGTH
    if len(skill.instructions) > MAX_SKILL_INSTRUCTIONS_LENGTH:
        return unavailable()
    current = store.get(selected)
    if (current is None or current.skill_id != selected or current.instructions != skill.instructions
            or not authorized(store, selected, actor) or private_legacy_package(current, actor, resources)
            or package_revision(current) != revision or cancelled() is not False):
        raise PermissionError('owner_named_skill_changed')
    return {'schema_version': 1, 'status': 'activated', 'skill_id': selected,
            'name': skill.name, 'revision': revision, 'instructions': skill.instructions,
            'selection_reason': 'explicit_user_request', 'execution_available': False}


def validate_owner_named_skill(actor: str, arguments: dict[str, Any], payload: dict[str, Any], *, context: Any) -> bool:
    try:
        if not actor or actor != getattr(getattr(context, 'task', None), 'user', None) or not isinstance(payload, dict):
            return False
        current = execute_owner_named_skill(arguments, context=context)
        return json.dumps(current, sort_keys=True, allow_nan=False) == json.dumps(payload, sort_keys=True, allow_nan=False)
    except Exception:
        return False


def build_owner_named_skill_tool_definition() -> ToolDefinition:
    descriptor = ToolDescriptor(OWNER_NAMED_SKILL_TOOL_ID, 'owner_named_skill', ToolKind.NATIVE,
        'Read the complete instructions of a single installed, authorized skill explicitly requested by exact name '
        'in the owner’s message. This provides guidance only; resource access, scripts, effects and automatic '
        'relevance selection are unavailable.', _SCHEMA, ('skill_guidance',), ('native', 'skills'), read_only=True)
    return ToolDefinition(descriptor, execute_owner_named_skill, _SCHEMA, accepts_context=True)
