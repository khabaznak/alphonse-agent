"""Unregistered private owner knowledge tool and internal-only read transport.

No model, provider or runtime uses this module yet. The caller must enforce its
own candidate reservation, live-claim, cancellation and projection checks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from .owner_knowledge_snapshot import OWNER_KNOWLEDGE_TOOL_ID, owner_knowledge_projection, snapshot_owner_knowledge


@dataclass(frozen=True, repr=False)
class PrivateKnowledgeRead:
    observation: dict = field(repr=False)
    binding_digest: str = field(repr=False)


def _read(arguments, *, context):
    if arguments != {} or context is None or not callable(context.cancellation_checker):
        raise ValueError('owner_knowledge_unavailable')
    snapshot = snapshot_owner_knowledge(context.messages, context.claim_receipt, context.task,
                                        cancellation_checker=context.cancellation_checker)
    observation = owner_knowledge_projection(snapshot)
    if observation is None:
        # Unavailable/incomplete is not an empty set of knowledge.
        raise ValueError('owner_knowledge_unavailable')
    return {'observation': observation, 'binding_digest': snapshot.digest}


def build_private_knowledge_tool_definition():
    """Create an unregistered tool; normal invocation redacts its result."""
    schema = {'type': 'object', 'properties': {}, 'additionalProperties': False}
    return ToolDefinition(descriptor=ToolDescriptor(tool_id=OWNER_KNOWLEDGE_TOOL_ID, name='owner_knowledge',
        kind=ToolKind.NATIVE, description='Read eligible private owner knowledge.',
        argument_schema=schema, metadata={'sensitive_content': True}, read_only=True),
        callable=_read, argument_schema=schema, accepts_context=True)


def read_private_knowledge(context: CoreLoopContext, task) -> PrivateKnowledgeRead:
    """Return ephemeral evidence to a trusted host without event or memory logging.

    This is not an authorization for model/provider disclosure. The enclosing
    host must run its normal budget, authority and cancellation checks and must
    bind/revalidate the digest at settlement, projection, recovery and pickup.
    """
    if not isinstance(context, CoreLoopContext) or type(context.tools) is not InMemoryToolRegistry:
        raise ValueError('owner_knowledge_unavailable')
    definition = context.tools.get_definition(OWNER_KNOWLEDGE_TOOL_ID)
    if (definition is None or not definition.enabled or definition.callable is not _read
            or definition.accepts_context is not True or definition.descriptor.read_only is not True
            or definition.descriptor.metadata.get('sensitive_content') is not True):
        raise ValueError('owner_knowledge_unavailable')
    # Invoke the pinned function directly: the registry is mutable, so a second
    # registry lookup could substitute a different callable or non-sensitive
    # descriptor after the check and leak data to generic telemetry.
    value = _read({}, context=context.tool_execution_context(task))
    if (not isinstance(value, dict) or set(value) != {'observation', 'binding_digest'}
            or not isinstance(value['observation'], dict) or not isinstance(value['binding_digest'], str)
            or len(value['binding_digest']) != 64
            or any(c not in '0123456789abcdef' for c in value['binding_digest'])):
        raise ValueError('owner_knowledge_unavailable')
    return PrivateKnowledgeRead(value['observation'], value['binding_digest'])
