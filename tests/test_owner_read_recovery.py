"""Recovery never reuses stale sources or retries uncertain owner actions."""
from datetime import datetime, timezone
import json
import os

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import ProcessingStatus
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable, restore_owner_read_observations
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor, authorize_trusted_owner_claim
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, TaskBudget, SQLiteTaskBudgetStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import (execute_search_memory, SEARCH_MEMORY_TOOL_ID,
    execute_read_memory_source, READ_MEMORY_SOURCE_TOOL_ID,
    build_search_memory_tool_definition, build_read_memory_source_tool_definition)
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    message_id = queue.enqueue_trusted_local_owner(prompt='Find apples', peer_uid=os.getuid())
    acquired = queue.claim_next(lease_seconds=60)
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    conversations = SQLiteConversationStore(path)
    conversations.record(owner_user_id=owner.user_id, role='user', content='Synthetic apples.',
                         source='test', source_message_id='source-one')
    ctx = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt,
                          conversation_store=conversations, cancellation_checker=lambda: False)
    now = datetime.now(timezone.utc)
    atomic = OwnerAtomicCheckpoint(queue, ExperimentalCheckpointStore(SQLiteQuestionStore(path)),
                                   task, acquired.claim_receipt, clock=lambda: now)
    vector = dict(actions=5, model_calls=8, input_tokens=524288, output_tokens=131072)
    zero = dict(actions=0, model_calls=0, input_tokens=0, output_tokens=0)
    initial = atomic.initialize(CandidateState(message_id, owner.user_id, 'goal'),
             TaskBudget(BudgetLimits(vector, zero, 100000, 0), started_at_ms=int(now.timestamp() * 1000)))
    return atomic, initial, ctx, conversations, owner


def settled_read(setup):
    atomic, initial, ctx, _, _ = setup
    args = {'query': 'apples', 'source_types': ['conversation']}
    selected = {'kind': 'tool', 'tool_id': SEARCH_MEMORY_TOOL_ID, 'arguments': args}
    fingerprint = 'a' * 64
    model = atomic.reserve(expected_version=initial.version, reservation_id='model-1', kind='model',
        amounts=dict(actions=0, model_calls=1, input_tokens=100, output_tokens=20), now_ms=initial.budget.started_at_ms)
    decision = atomic.settle_model(expected_version=model.version, reservation_id='model-1', outcome='success',
                                   input_tokens=10, output_tokens=10, decision=selected,
                                   projection_fingerprint=fingerprint)
    reserved = atomic.reserve(expected_version=decision.version, reservation_id='read-1', kind='tool',
        amounts=dict(actions=1, model_calls=0, input_tokens=0, output_tokens=0),
        now_ms=initial.budget.started_at_ms, tool_id=SEARCH_MEMORY_TOOL_ID, arguments=args,
        selected_decision=selected, projection_fingerprint=fingerprint)
    result = execute_search_memory(args, context=ctx.tool_execution_context(atomic.task))
    settled = atomic.settle_read(expected_version=reserved.version, reservation_id='read-1',
        tool_id=SEARCH_MEMORY_TOOL_ID, arguments=args, evidence_ref='owner-read:one', observation=result)
    return settled


def test_recovery_reads_actor_bound_durable_source_and_returns_checkpoint(setup):
    atomic, _, ctx, _, _ = setup
    saved = settled_read(setup)
    checkpoint, bindings = restore_owner_read_observations(atomic, ctx, expected_version=saved.version)
    assert checkpoint.state.evidence_sources == ['owner-read:one']
    assert bindings['owner-read:one']['observation']['results']
    with atomic.store.store._connect() as conn:
        stored = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                              (atomic.task.task_id,)).fetchone()['task_state_json']
    assert 'Synthetic apples.' not in stored
    assert 'observation_sha256' in stored


def test_recovery_revalidates_search_overflow_and_rejects_changed_results(setup):
    atomic, _, ctx, conversations, owner = setup
    for i in range(31):
        conversations.record(owner_user_id=owner.user_id, role='user', content=f'Synthetic apples {i}',
                             source='test', source_message_id=f'overflow-{i}')
    saved = settled_read(setup)
    recovered, bindings = restore_owner_read_observations(atomic, ctx, expected_version=saved.version)
    assert recovered.state.evidence_sources == ['owner-read:one']
    assert bindings['owner-read:one']['observation']['next_offset'] is not None
    conversations.record(owner_user_id=owner.user_id, role='user', content='More synthetic apples.',
                         source='test', source_message_id='changed-overflow')
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=saved.version)


def test_recovered_newer_search_overflow_is_not_cleared_by_older_complete_search(setup, tmp_path):
    atomic, initial, ctx, conversations, owner = setup
    conversations.record(owner_user_id=owner.user_id, role='user', content='Unique synthetic record.',
                         source='test', source_message_id='unique')
    for i in range(31):
        conversations.record(owner_user_id=owner.user_id, role='user', content=f'Synthetic apples {i}',
                             source='test', source_message_id=f'overflow-order-{i}')
    current = initial
    for i, query in enumerate(('unique', 'apples')):
        args = {'query': query, 'source_types': ['conversation'], 'limit': 30}
        decision = {'kind': 'tool', 'tool_id': SEARCH_MEMORY_TOOL_ID, 'arguments': args}
        model = atomic.reserve(expected_version=current.version, reservation_id=f'model-{i}', kind='model',
            amounts=dict(actions=0, model_calls=1, input_tokens=100, output_tokens=20), now_ms=initial.budget.started_at_ms)
        selected = atomic.settle_model(expected_version=model.version, reservation_id=f'model-{i}', outcome='success',
            input_tokens=10, output_tokens=10, decision=decision, projection_fingerprint='a' * 64)
        pending = atomic.reserve(expected_version=selected.version, reservation_id=f'read-{i}', kind='tool',
            amounts=dict(actions=1, model_calls=0, input_tokens=0, output_tokens=0), now_ms=initial.budget.started_at_ms,
            tool_id=SEARCH_MEMORY_TOOL_ID, arguments=args, selected_decision=decision, projection_fingerprint='a' * 64)
        observation = execute_search_memory(args, context=ctx.tool_execution_context(atomic.task))
        assert (observation['next_offset'] is not None) == (i == 1)
        current = atomic.settle_read(expected_version=pending.version, reservation_id=f'read-{i}',
            tool_id=SEARCH_MEMORY_TOOL_ID, arguments=args, evidence_ref=f'owner-read:{i}', observation=observation)
    recovered, _ = restore_owner_read_observations(atomic, ctx, expected_version=current.version)
    assert recovered.state.evidence_sources == ['owner-read:0', 'owner-read:1']
    atomic.task._checkpoint_version = current.version

    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            assert 'next_offset' in request.prompt
            decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 10, 'output_tokens': 10})
    ctx.inference = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline'))
    ctx.tools = InMemoryToolRegistry()
    ctx.tools.register(build_search_memory_tool_definition())
    ctx.tools.register(build_read_memory_source_tool_definition())
    processor = OwnerReadConversationProcessor(users=V2UserStore(atomic.queue.db_path),
        budgets=SQLiteTaskBudgetStore(str(tmp_path / 'unused.sqlite')), authorize_claim=authorize_trusted_owner_claim,
        save_checkpoint=lambda *_: None, atomic_checkpoints=atomic.store,
        now_ms=lambda: initial.budget.started_at_ms)
    result = processor.process(atomic.task, ctx)
    assert result.status == ProcessingStatus.FAILED
    assert result.error == 'candidate_answer_unsupported'


def test_recovery_reconstructs_partial_page_without_persisting_content(setup):
    atomic, initial, ctx, conversations, owner = setup
    event = conversations.record(owner_user_id=owner.user_id, role='user',
        content='Willow picnic Thursday. ' + 'filler ' * 2000 + 'Correction Saturday.',
        source='test', source_message_id='long-source')
    args = {'source_id': 'conversation:' + event.event_id, 'neighbors': 0, 'max_chars': 12000}
    decision = {'kind': 'tool', 'tool_id': READ_MEMORY_SOURCE_TOOL_ID, 'arguments': args}
    model = atomic.reserve(expected_version=initial.version, reservation_id='long-model', kind='model',
        amounts=dict(actions=0, model_calls=1, input_tokens=100, output_tokens=20), now_ms=initial.budget.started_at_ms)
    selected = atomic.settle_model(expected_version=model.version, reservation_id='long-model', outcome='success',
        input_tokens=10, output_tokens=10, decision=decision, projection_fingerprint='a' * 64)
    reserved = atomic.reserve(expected_version=selected.version, reservation_id='long-read', kind='tool',
        amounts=dict(actions=1, model_calls=0, input_tokens=0, output_tokens=0), now_ms=initial.budget.started_at_ms,
        tool_id=READ_MEMORY_SOURCE_TOOL_ID, arguments=args, selected_decision=decision, projection_fingerprint='a' * 64)
    observation = execute_read_memory_source(args, context=ctx.tool_execution_context(atomic.task))
    assert observation['next_offset'] == 12000
    saved = atomic.settle_read(expected_version=reserved.version, reservation_id='long-read',
        tool_id=READ_MEMORY_SOURCE_TOOL_ID, arguments=args, evidence_ref='owner-read:long', observation=observation)
    recovered, bindings = restore_owner_read_observations(atomic, ctx, expected_version=saved.version)
    assert recovered.state.evidence_sources == ['owner-read:long']
    assert bindings['owner-read:long']['observation']['next_offset'] == 12000
    with atomic.store.store._connect() as conn:
        stored = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                              (atomic.task.task_id,)).fetchone()['task_state_json']
    assert 'Willow picnic Thursday' not in stored
    with conversations._connect() as conn:
        conn.execute('UPDATE v2_conversation_events SET content=? WHERE event_id=?',
                     ('Changed picnic Thursday. ' + 'filler ' * 2000 + 'Correction Saturday.', event.event_id))
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=saved.version)


def test_real_owner_processor_continues_recovered_partial_source(setup, tmp_path):
    atomic, initial, ctx, conversations, owner = setup
    event = conversations.record(owner_user_id=owner.user_id, role='user',
        content='Willow picnic Thursday. ' + 'filler ' * 2000 + 'Correction Saturday.',
        source='test', source_message_id='long-source-resume')
    source = 'conversation:' + event.event_id
    args = {'source_id': source, 'neighbors': 0, 'max_chars': 12000}
    decision = {'kind': 'tool', 'tool_id': READ_MEMORY_SOURCE_TOOL_ID, 'arguments': args}
    model = atomic.reserve(expected_version=initial.version, reservation_id='model-partial', kind='model',
        amounts=dict(actions=0, model_calls=1, input_tokens=100, output_tokens=20), now_ms=initial.budget.started_at_ms)
    chosen = atomic.settle_model(expected_version=model.version, reservation_id='model-partial', outcome='success',
        input_tokens=10, output_tokens=10, decision=decision, projection_fingerprint='a' * 64)
    pending = atomic.reserve(expected_version=chosen.version, reservation_id='read-partial', kind='tool',
        amounts=dict(actions=1, model_calls=0, input_tokens=0, output_tokens=0), now_ms=initial.budget.started_at_ms,
        tool_id=READ_MEMORY_SOURCE_TOOL_ID, arguments=args, selected_decision=decision, projection_fingerprint='a' * 64)
    page = execute_read_memory_source(args, context=ctx.tool_execution_context(atomic.task))
    saved = atomic.settle_read(expected_version=pending.version, reservation_id='read-partial',
        tool_id=READ_MEMORY_SOURCE_TOOL_ID, arguments=args, evidence_ref='owner-read:partial', observation=page)
    task = atomic.task
    task._checkpoint_version = saved.version

    class Provider:
        supports_image_input = False
        def __init__(self):
            self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                assert '12000' in request.prompt and 'Willow picnic Thursday' not in request.prompt
                value = {'kind':'tool', 'tool_id':READ_MEMORY_SOURCE_TOOL_ID,
                         'arguments':{'source_id':source, 'neighbors':0, 'max_chars':12000, 'offset':12000}, 'text':None}
            else:
                assert 'Correction Saturday' in request.prompt and 'Willow picnic Thursday' in request.prompt
                value = {'kind':'answer', 'tool_id':None, 'arguments':None, 'text':'Saturday.'}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens':100, 'output_tokens':20})
    provider = Provider()
    ctx.inference = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline'))
    ctx.tools = InMemoryToolRegistry()
    ctx.tools.register(build_search_memory_tool_definition())
    ctx.tools.register(build_read_memory_source_tool_definition())
    processor = OwnerReadConversationProcessor(users=V2UserStore(atomic.queue.db_path),
        budgets=SQLiteTaskBudgetStore(str(tmp_path / 'unused.sqlite')), authorize_claim=authorize_trusted_owner_claim,
        save_checkpoint=lambda *_: None, atomic_checkpoints=atomic.store,
        now_ms=lambda: initial.budget.started_at_ms)
    result = processor.process(task, ctx)
    assert result.status == ProcessingStatus.WAITING, result.error
    assert result.snapshot.metadata['draft'] == 'Saturday.'
    assert len(provider.requests) == 2


def test_recovery_refuses_changed_source_and_inflight_action(setup):
    atomic, _, ctx, conversations, owner = setup
    saved = settled_read(setup)
    conversations.record(owner_user_id=owner.user_id, role='user', content='More apples.',
                         source='test', source_message_id='source-two')
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=saved.version)
    pending = atomic.reserve(expected_version=saved.version, reservation_id='model-2', kind='model',
        amounts=dict(actions=0, model_calls=1, input_tokens=100, output_tokens=20), now_ms=saved.budget.started_at_ms)
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=pending.version)


def test_recovery_refuses_tampered_observation_digest(setup):
    atomic, _, ctx, _, _ = setup
    saved = settled_read(setup)
    with atomic.store.store._connect() as conn:
        row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                           (atomic.task.task_id,)).fetchone()
        raw = json.loads(row['task_state_json'])
        raw['metadata']['owner_read_observations']['owner-read:one']['observation_sha256'] = '0' * 64
        conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?',
                     (json.dumps(raw), atomic.task.task_id))
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=saved.version)


def test_recovery_refuses_revoked_owner_and_missing_receipt(setup):
    atomic, initial, ctx, _, owner = setup
    ctx.claim_receipt = None
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=initial.version)
    ctx.claim_receipt = atomic.receipt
    V2UserStore(atomic.queue.db_path).update_user(owner.user_id, is_active=False)
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=initial.version)


@pytest.mark.parametrize('input_tokens', [None, 101])
def test_recovery_refuses_unknown_or_overrun_model_usage(setup, input_tokens):
    atomic, initial, ctx, _, _ = setup
    pending = atomic.reserve(expected_version=initial.version, reservation_id='model-1', kind='model',
        amounts=dict(actions=0, model_calls=1, input_tokens=100, output_tokens=20), now_ms=initial.budget.started_at_ms)
    settled = atomic.settle_model(expected_version=pending.version, reservation_id='model-1', outcome='success',
                                  input_tokens=input_tokens, output_tokens=10)
    settled.state.invalid_decision_retries = 1
    settled.state.decision_format_pending = True
    settled = atomic.save_state(settled.state, consume_decision=True)
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=settled.version)


def test_recovery_refuses_uncommitted_model_decision(setup):
    atomic, initial, ctx, _, _ = setup
    pending = atomic.reserve(expected_version=initial.version, reservation_id='model-1', kind='model',
        amounts=dict(actions=0, model_calls=1, input_tokens=100, output_tokens=20), now_ms=initial.budget.started_at_ms)
    settled = atomic.settle_model(expected_version=pending.version, reservation_id='model-1', outcome='success',
                                  input_tokens=10, output_tokens=10)
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=settled.version)


def test_recovery_rechecks_owner_after_source_io(setup, monkeypatch):
    from alphonse.agent_v2.core.intelligence.simple_loop import owner_read_recovery

    atomic, _, ctx, _, owner = setup
    saved = settled_read(setup)
    original = owner_read_recovery.revalidate_conversation_observation

    def revoke_after_check(*args, **kwargs):
        valid = original(*args, **kwargs)
        V2UserStore(atomic.queue.db_path).update_user(owner.user_id, is_active=False)
        return valid

    monkeypatch.setattr(owner_read_recovery, 'revalidate_conversation_observation', revoke_after_check)
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=saved.version)


def test_recovery_does_not_reread_source_after_owner_revoked_during_read(setup, monkeypatch):
    from alphonse.agent_v2.core.intelligence.simple_loop import owner_read_recovery

    atomic, _, ctx, _, owner = setup
    saved = settled_read(setup)
    original = owner_read_recovery.execute_search_memory
    source_revalidations = []

    def revoke_during_read(*args, **kwargs):
        observation = original(*args, **kwargs)
        V2UserStore(atomic.queue.db_path).update_user(owner.user_id, is_active=False)
        return observation

    def track_revalidation(*args, **kwargs):
        source_revalidations.append(True)
        return True

    monkeypatch.setattr(owner_read_recovery, 'execute_search_memory', revoke_during_read)
    monkeypatch.setattr(owner_read_recovery, 'revalidate_conversation_observation', track_revalidation)
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=saved.version)
    assert source_revalidations == []


def test_recovery_does_not_reread_after_checkpoint_changes_during_read(setup, monkeypatch):
    from alphonse.agent_v2.core.intelligence.simple_loop import owner_read_recovery

    atomic, _, ctx, _, _ = setup
    saved = settled_read(setup)
    original = owner_read_recovery.execute_search_memory
    source_revalidations = []

    def change_checkpoint_during_read(*args, **kwargs):
        observation = original(*args, **kwargs)
        with atomic.store.store._connect() as conn:
            conn.execute('UPDATE v2_task_checkpoints SET version=version+1 WHERE task_id=?',
                         (atomic.task.task_id,))
        return observation

    monkeypatch.setattr(owner_read_recovery, 'execute_search_memory', change_checkpoint_during_read)
    monkeypatch.setattr(owner_read_recovery, 'revalidate_conversation_observation',
                        lambda *args, **kwargs: source_revalidations.append(True) or True)
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=saved.version)
    assert source_revalidations == []


def test_real_owner_processor_resumes_after_reconstructing_source(setup, tmp_path):
    atomic, _, ctx, _, _ = setup
    saved = settled_read(setup)
    task = atomic.task
    task._checkpoint_version = saved.version

    class Provider:
        supports_image_input = False
        def __init__(self):
            self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            return InferenceResult(json_value={'kind': 'answer', 'tool_id': None, 'arguments': None,
                                             'text': 'Synthetic apples.'}, raw_response='{}',
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    provider = Provider()
    ctx.inference = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline'))
    ctx.tools = InMemoryToolRegistry()
    ctx.tools.register(build_search_memory_tool_definition())
    ctx.tools.register(build_read_memory_source_tool_definition())
    processor = OwnerReadConversationProcessor(users=V2UserStore(atomic.queue.db_path),
        budgets=SQLiteTaskBudgetStore(str(tmp_path / 'unused.sqlite')), authorize_claim=authorize_trusted_owner_claim,
        save_checkpoint=lambda *_: None, atomic_checkpoints=atomic.store)
    result = processor.process(task, ctx)
    assert result.status == ProcessingStatus.WAITING, result.error
    assert result.snapshot.metadata['draft'] == 'Synthetic apples.'
    assert len(provider.requests) == 1 and 'Synthetic apples.' in provider.requests[0].prompt


def test_recovery_reconstructs_paginated_snapshot_and_rejects_changes(setup):
    atomic, initial, ctx, conversations, owner = setup
    for i in range(31):
        conversations.record(owner_user_id=owner.user_id, role='user', content=f'Synthetic apples {i}',
                             source='test', source_message_id=f'page-{i}')
    current = initial
    for i, offset in enumerate((0, 30)):
        args = {'query': 'apples', 'source_types': ['conversation'], 'limit': 30}
        if offset:
            args['offset'] = offset
        decision = {'kind': 'tool', 'tool_id': SEARCH_MEMORY_TOOL_ID, 'arguments': args}
        model = atomic.reserve(expected_version=current.version, reservation_id=f'page-model-{i}', kind='model',
            amounts=dict(actions=0, model_calls=1, input_tokens=100, output_tokens=20), now_ms=initial.budget.started_at_ms)
        selected = atomic.settle_model(expected_version=model.version, reservation_id=f'page-model-{i}', outcome='success',
            input_tokens=10, output_tokens=10, decision=decision, projection_fingerprint='a' * 64)
        pending = atomic.reserve(expected_version=selected.version, reservation_id=f'page-read-{i}', kind='tool',
            amounts=dict(actions=1, model_calls=0, input_tokens=0, output_tokens=0), now_ms=initial.budget.started_at_ms,
            tool_id=SEARCH_MEMORY_TOOL_ID, arguments=args, selected_decision=decision, projection_fingerprint='a' * 64)
        observation = execute_search_memory(args, context=ctx.tool_execution_context(atomic.task))
        current = atomic.settle_read(expected_version=pending.version, reservation_id=f'page-read-{i}',
            tool_id=SEARCH_MEMORY_TOOL_ID, arguments=args, evidence_ref=f'owner-read:page-{i}', observation=observation)
    saved, bindings = restore_owner_read_observations(atomic, ctx, expected_version=current.version)
    assert saved.state.evidence_sources == ['owner-read:page-0', 'owner-read:page-1']
    assert bindings['owner-read:page-0']['observation']['snapshot'] == bindings['owner-read:page-1']['observation']['snapshot']
    assert bindings['owner-read:page-1']['observation']['next_offset'] is None
    conversations.record(owner_user_id=owner.user_id, role='user', content='Another synthetic apple.',
                         source='test', source_message_id='changed-after-pages')
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=current.version)


def test_search_pages_reject_orphans_gaps_and_changed_snapshots(setup):
    from copy import deepcopy
    from alphonse.agent_v2.core.tools.registry.native.memory import conversation_search_sequences
    atomic, _, ctx, conversations, owner = setup
    for i in range(31):
        conversations.record(owner_user_id=owner.user_id, role='user', content=f'Synthetic apples {i}',
                             source='test', source_message_id=f'invalid-page-{i}')
    args0 = {'query': 'apples', 'source_types': ['conversation'], 'limit': 30}
    args1 = {**args0, 'offset': 30}
    page0 = execute_search_memory(args0, context=ctx.tool_execution_context(atomic.task))
    page1 = execute_search_memory(args1, context=ctx.tool_execution_context(atomic.task))
    def item(args, page): return (SEARCH_MEMORY_TOOL_ID, args, page)
    with pytest.raises(ValueError):
        conversation_search_sequences(owner.user_id, {'second': item(args1, page1)})
    wrong = deepcopy(page1)
    wrong['snapshot'] = 'changed'
    with pytest.raises(ValueError):
        conversation_search_sequences(owner.user_id, {'first': item(args0, page0), 'second': item(args1, wrong)})
    gapargs = {**args0, 'offset': 31}
    gappage = execute_search_memory(gapargs, context=ctx.tool_execution_context(atomic.task))
    with pytest.raises(ValueError):
        conversation_search_sequences(owner.user_id, {'first': item(args0, page0), 'gap': item(gapargs, gappage)})
