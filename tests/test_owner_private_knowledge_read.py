"""No provider or household data: transport isolation for disposable knowledge."""
import json
from dataclasses import replace

import pytest

from test_owner_knowledge_snapshot import fixture
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.intelligence.simple_loop.owner_knowledge_snapshot import OWNER_KNOWLEDGE_TOOL_ID
from alphonse.agent_v2.core.intelligence.simple_loop.owner_private_knowledge_read import (
    build_private_knowledge_tool_definition, read_private_knowledge,
)


def test_private_read_has_no_ui_memory_or_telemetry_payload(fixture):
    queue, task, _, record, insert, snapshot = fixture
    insert(record)
    events, telemetry, memory = [], [], []
    class Recorder:
        def event(self, *args):
            memory.append(args)
    registry = InMemoryToolRegistry()
    registry.register(build_private_knowledge_tool_definition())
    ctx = CoreLoopContext(messages=queue, claim_receipt=snapshot.receipt, tools=registry,
                          cancellation_checker=lambda: False, ui_event_sink=events.append,
                          telemetry_sink=telemetry.append, memory=Recorder())
    result = read_private_knowledge(ctx, task)
    assert result.observation['items'][0]['content'] == 'Apples are red'
    assert result.binding_digest == snapshot().digest
    assert 'Apples are red' not in repr(result)
    assert result.binding_digest not in json.dumps(result.observation)
    assert not events and not memory
    assert not telemetry

    # Generic invocation cannot expose its sensitive result to its caller,
    # Desktop events, memory or telemetry.
    generic = ToolInvocationService(context=ctx, task=task).invoke(OWNER_KNOWLEDGE_TOOL_ID, {})
    assert generic['result'] == {'redacted': True}
    assert 'Apples are red' not in str((generic, events, memory, telemetry))
    assert result.binding_digest not in str((generic, events, memory, telemetry))


def test_private_read_rejects_forged_tool_and_cancelled_claim(fixture):
    queue, task, _, record, insert, snapshot = fixture
    insert(record)
    registry = InMemoryToolRegistry()
    definition = build_private_knowledge_tool_definition()
    registry.register(replace(definition, callable=lambda *_args, **_kwargs: {'observation': {}, 'binding_digest': 'a' * 64}))
    ctx = CoreLoopContext(messages=queue, claim_receipt=snapshot.receipt, tools=registry,
                          cancellation_checker=lambda: False)
    with pytest.raises(ValueError, match='owner_knowledge_unavailable'):
        read_private_knowledge(ctx, task)
    registry.register(definition)
    ctx.cancellation_checker = lambda: True
    with pytest.raises(ValueError, match='owner_knowledge_unavailable'):
        read_private_knowledge(ctx, task)


def test_private_read_does_not_relookup_mutable_registry(fixture, monkeypatch):
    queue, task, _, record, insert, snapshot = fixture
    insert(record)
    registry = InMemoryToolRegistry()
    definition = build_private_knowledge_tool_definition()
    registry.register(definition)
    original = registry.get_definition
    def swap_after_lookup(key):
        result = original(key)
        registry.register(replace(definition, callable=lambda *_args, **_kwargs: pytest.fail('substituted callable ran')))
        return result
    monkeypatch.setattr(registry, 'get_definition', swap_after_lookup)
    ctx = CoreLoopContext(messages=queue, claim_receipt=snapshot.receipt, tools=registry,
                          cancellation_checker=lambda: False)
    assert read_private_knowledge(ctx, task).observation['items'][0]['content'] == 'Apples are red'
