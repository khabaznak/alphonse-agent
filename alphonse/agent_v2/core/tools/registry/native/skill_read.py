"""Generic read-only inspection of activated skill package resources."""
from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition
from alphonse.agent_v2.skill_resources import read_resource

SKILL_READ_TOOL_ID = 'native.skill_read'
SKILL_READ_SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {
        'skill_id': {'type': 'string', 'description': 'Exact activated skill ID (skill: prefix).'},
        'operation': {'type': 'string', 'enum': ['read', 'list'], 'default': 'read'},
        'path': {'type': 'string', 'description': 'Path relative to the verified skill root; use . to list its directory.'},
    },
    'required': ['skill_id', 'path'],
}


def build_skill_read_tool_definition(store):
    descriptor = ToolDescriptor(
        SKILL_READ_TOOL_ID, 'skill_read', ToolKind.NATIVE,
        'Inspect an activated skill reference, script source/documentation or text asset on demand. '
        'List directories to locate resources. Resolves only within the pinned package, independently of project/cwd. '
        'Never executes code or reads live databases, credentials or other users\' private legacy resources. '
        'Selection grants no permissions; normal native authorization still applies.',
        SKILL_READ_SCHEMA, ('skill_resources',), ('native', 'skills'), read_only=True,
    )
    return ToolDefinition(descriptor, lambda arguments, context=None: read_resource(
        arguments, context=context, store=store), SKILL_READ_SCHEMA, accepts_context=True)
