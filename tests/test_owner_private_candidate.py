"""Synthetic private candidate dispatch; not a registered runtime feature."""
import json
import pytest

from test_owner_read_recovery import setup
from alphonse.agent_v2.core.memory.contracts import AccessScope, conversation_item
from alphonse.agent_v2.core.memory.knowledge import KnowledgeRecord, KnowledgeStore, _source_snapshot_digest
from alphonse.agent_v2.core.intelligence.simple_loop.owner_knowledge_snapshot import OWNER_KNOWLEDGE_TOOL_ID, snapshot_owner_knowledge
from alphonse.agent_v2.core.intelligence.simple_loop.owner_private_knowledge_read import build_private_knowledge_tool_definition, read_private_knowledge
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateHost, CandidateState, run_candidate
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry


@pytest.mark.parametrize('mode', ['success', 'missing_callback', 'cancel_after_read'])
def test_private_dispatch_keeps_digest_out_of_model_and_emits_no_tool_events(setup, mode):
    atomic, initial, ctx, conversations, owner = setup
    KnowledgeStore(atomic.queue.db_path)
    source = conversations.event_for_source_message_id(owner_user_id=owner.user_id, source_message_id='source-one')
    item = conversation_item(source, owner.user_id)
    record = KnowledgeRecord(memory_id='memory:private', content='Synthetic apples.', access=AccessScope(owner.user_id),
        sources=(item.source,), recorded_at=source.created_at, kind='fact', subjects=('apples',), claim_key='apple',
        basis='user_report', quotes=('Synthetic apples.',), source_binding_version=1,
        source_snapshot_digests=(_source_snapshot_digest(item),))
    with atomic.queue._connect() as conn:
        conn.execute('INSERT INTO memory_knowledge VALUES (?,?,?,?,?,?)',
            (record.memory_id, owner.user_id, record.claim_key, json.dumps(record.to_dict()), record.status, '[]'))
    registry = InMemoryToolRegistry()
    registry.register(build_private_knowledge_tool_definition())
    ctx.tools = registry
    cancelled = [False]
    ctx.cancellation_checker = lambda: cancelled[0]
    events, telemetry, prompts, evidence = [], [], [], {}
    ctx.ui_event_sink = events.append
    ctx.telemetry_sink = telemetry.append
    decisions = iter([
        {'kind': 'tool', 'tool_id': OWNER_KNOWLEDGE_TOOL_ID, 'arguments': {}, 'usage': {'input_tokens': 20, 'output_tokens': 10}},
        {'kind': 'answer', 'text': 'Synthetic apples.', 'usage': {'input_tokens': 20, 'output_tokens': 10}},
    ])
    def choose(text, _cap):
        prompts.append(text)
        return next(decisions)
    def access(actor, ref):
        if actor != owner.user_id:
            return False
        if ref == 'goal':
            return True
        if ref not in evidence:
            return False
        current = snapshot_owner_knowledge(atomic.queue, atomic.receipt, atomic.task,
            cancellation_checker=ctx.cancellation_checker, clock=atomic.clock)
        return current.status == 'available' and current.digest == evidence[ref][1]
    def load(actor, ref):
        assert access(actor, ref)
        return 'Find apples' if ref == 'goal' else json.dumps(evidence[ref][0])
    def record_private(actor, tool, args, result, digest):
        assert actor == owner.user_id and tool == OWNER_KNOWLEDGE_TOOL_ID and args == {}
        evidence['private'] = (result['result'], digest)
        return 'private'
    def private_read():
        result = read_private_knowledge(ctx, atomic.task)
        cancelled[0] = mode == 'cancel_after_read'
        return result
    host = CandidateHost(atomic.task, ctx, None, lambda *_: None, choose, access, load,
        lambda actor, tool, args: actor == owner.user_id and tool == OWNER_KNOWLEDGE_TOOL_ID and args == {},
        lambda *_: '', lambda actor, response, refs: bool(refs),
        lambda: initial.budget.started_at_ms + 100, atomic_checkpoint=atomic,
        input_reservation=65536, output_reservation=16384, projection_budget=65536,
        validate_observation=lambda actor, tool, args, payload: payload.get('status') == 'complete',
        read_private_knowledge=None if mode == 'missing_callback' else private_read, record_private_knowledge=record_private)
    if mode == 'cancel_after_read':
        # The call remains in flight; cancellation never makes its result available
        # or silently replays the read on recovery.
        result = run_candidate(CandidateState(atomic.task.task_id, owner.user_id, 'goal'), host)
        assert result.status == 'cancelled'
        saved, bindings = atomic.load(expected_version=atomic.version)
        assert saved.state.status == 'cancelled'
        assert saved.state.inflight and bindings == {}
        assert len(prompts) == 1 and not evidence and not events and not telemetry
        return
    result = run_candidate(CandidateState(atomic.task.task_id, owner.user_id, 'goal'), host)
    if mode == 'missing_callback':
        assert result.blocker == 'candidate_private_read_unavailable'
        assert not evidence and len(prompts) == 1
        return
    assert result.status == 'completed', result.blocker
    assert 'Synthetic apples.' in prompts[-1]
    assert evidence['private'][1] not in ''.join(prompts)
    assert not events and not telemetry
    _, bindings = atomic.load(expected_version=atomic.version)
    assert bindings['private']['knowledge_snapshot_digest'] == evidence['private'][1]
