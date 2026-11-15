"""Disposable owner candidate tests for the shared-claim checkpoint mode."""
import json
import os
import threading
import tempfile
from types import SimpleNamespace
from datetime import datetime, timezone

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.owner_atomic_checkpoint import OwnerAtomicCheckpoint, observation_digest
from alphonse.agent_v2.core.intelligence.v3.owner_read_processor import OwnerReadConversationProcessor, authorize_trusted_owner_claim
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import CheckpointConflict, ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetExhausted, BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / 'owner.db'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    SQLiteOutboundStore(path)
    questions = SQLiteQuestionStore(path)
    message_id = queue.enqueue_trusted_local_owner(prompt='Find the apples note.', peer_uid=os.getuid())
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    store = ExperimentalCheckpointStore(questions)
    atomic = OwnerAtomicCheckpoint(queue, store, task, acquired.claim_receipt, clock=lambda: datetime.now(timezone.utc))
    return tmp_path, users, owner, queue, acquired, task, store, atomic


def make_budget():
    vector = lambda a, m, i, o: {'actions': a, 'model_calls': m, 'input_tokens': i, 'output_tokens': o}
    return TaskBudget(BudgetLimits(vector(3, 3, 1000, 300), vector(0, 1, 100, 50), 100000, 0), started_at_ms=100)



def reserve_selected_tool(atomic, start, tool_id, arguments):
    projection = 'a' * 64
    decision = {'kind': 'tool', 'tool_id': tool_id, 'arguments': arguments}
    reserved = atomic.reserve(expected_version=start.version, reservation_id='model-prerequisite', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=100)
    settled = atomic.settle_model(expected_version=reserved.version, reservation_id='model-prerequisite',
        outcome='success', input_tokens=1, output_tokens=1, decision=decision, projection_fingerprint=projection)
    return atomic.reserve(expected_version=settled.version, reservation_id='read-1', kind='tool',
        amounts={'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}, now_ms=101,
        tool_id=tool_id, arguments=arguments, selected_decision=decision, projection_fingerprint=projection)

def test_reservation_and_observation_are_atomic_and_reloadable(setup):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = reserve_selected_tool(atomic, start, 'native.read_memory_source',
                                     {'source_id': 'conversation:fixture'})
    checkpoint, observations = atomic.load(expected_version=reserved.version)
    assert checkpoint.state.inflight == 'read-1' and checkpoint.budget.usage()['in_flight'] == ['read-1']
    assert observations == {}
    with pytest.raises(CheckpointConflict):
        atomic.settle_read(expected_version=reserved.version, reservation_id='read-1', tool_id='native.read_memory_source',
            arguments={'source_id': 'conversation:other'}, evidence_ref='e1', observation={'text': 'x'})
    settled = atomic.settle_read(expected_version=reserved.version, reservation_id='read-1', tool_id='native.read_memory_source',
        arguments={'source_id': 'conversation:fixture'}, evidence_ref='e1', observation={'text': 'x'})
    restored, observations = atomic.load(expected_version=settled.version)
    assert restored.state.inflight == '' and restored.state.evidence_sources == ['e1']
    assert not restored.budget.usage()['in_flight']
    assert observations['e1'] == {'actor': task.user, 'tool_id': 'native.read_memory_source',
                                 'arguments': {'source_id': 'conversation:fixture'},
                                 'observation_sha256': observation_digest({'text': 'x'})}


def test_replacement_claim_cannot_settle_prior_reservation(setup):
    _, _, _, queue, acquired, task, store, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = atomic.reserve(expected_version=start.version, reservation_id='m1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    # Simulate a recovery coordinator putting a lost lease back on the queue.
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET status='pending', lease_owner='', lease_expires_at='' WHERE message_id=?", (task.message_id,))
    replacement = queue.claim_next(lease_owner='replacement')
    assert replacement and replacement.claim_receipt.claim_generation != acquired.claim_receipt.claim_generation
    new_worker = OwnerAtomicCheckpoint(queue, store, task, replacement.claim_receipt, clock=lambda: datetime.now(timezone.utc))
    with pytest.raises(CheckpointConflict, match='owner_atomic_reservation_mismatch'):
        new_worker.settle_model(expected_version=reserved.version, reservation_id='m1', outcome='success', input_tokens=2, output_tokens=1)
    loaded, _ = new_worker.load(expected_version=reserved.version)
    assert loaded.state.inflight == 'm1'


def test_overrun_cannot_start_another_action(setup):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = atomic.reserve(expected_version=start.version, reservation_id='m1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    settled = atomic.settle_model(expected_version=reserved.version, reservation_id='m1', outcome='success',
                                  input_tokens=101, output_tokens=1)
    with pytest.raises(CheckpointConflict, match='owner_atomic_model_reservation_overrun'):
        atomic.reserve(expected_version=settled.version, reservation_id='read', kind='tool',
            amounts={'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}, now_ms=102,
            tool_id='native.read_memory_source', arguments={'source_id': 'conversation:fixture'})


def test_crash_after_model_settlement_does_not_replay_lost_decision(setup):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = atomic.reserve(expected_version=start.version, reservation_id='m1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    settled = atomic.settle_model(expected_version=reserved.version, reservation_id='m1', outcome='success',
                                  input_tokens=10, output_tokens=5)
    with pytest.raises(CheckpointConflict, match='owner_atomic_pending_decision_unresolved'):
        atomic.load(expected_version=settled.version)
    with pytest.raises(CheckpointConflict, match='owner_atomic_pending_decision_unresolved'):
        atomic.reserve(expected_version=settled.version, reservation_id='m2', kind='model',
            amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=102)


def test_dispatch_check_requires_current_checkpoint_and_exact_reserved_action(setup):
    _, _, _, queue, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = reserve_selected_tool(atomic, start, 'native.read_memory_source',
                                     {'source_id': 'conversation:fixture'})
    assert atomic.check_claim('read-1', 'tool', tool_id='native.read_memory_source', arguments={'source_id': 'conversation:fixture'})
    assert not atomic.check_claim('read-1', 'tool', tool_id='native.read_memory_source', arguments={'source_id': 'conversation:other'})
    with queue._connect() as conn:
        conn.execute("UPDATE v2_task_checkpoints SET status='cancelled', version=? WHERE task_id=?", (reserved.version + 1, task.task_id))
    assert not atomic.check_claim('read-1', 'tool', tool_id='native.read_memory_source', arguments={'source_id': 'conversation:fixture'})


def test_rejected_reservation_retains_time_observation_and_can_save_blocker(setup):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    with pytest.raises(BudgetExhausted):
        atomic.reserve(expected_version=start.version, reservation_id='m1', kind='model',
            amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=100100)
    assert atomic.version == start.version + 1
    state = CandidateState(task.task_id, task.user, 'goal', status='blocked', blocker='budget exhausted')
    result = atomic.save_state(state)
    loaded, _ = atomic.load(expected_version=result.version)
    assert loaded.state.status == 'blocked' and loaded.budget.last_observed_at_ms == 100100


class Provider:
    supports_image_input = False
    def __init__(self, source_id):
        self.source_id, self.requests = source_id, []
    def generate_json(self, request):
        self.requests.append(request)
        actions = [
            {'kind': 'tool', 'tool_id': 'native.search_memory', 'arguments': {'query': 'apples', 'source_types': ['conversation']}, 'text': None},
            {'kind': 'tool', 'tool_id': 'native.read_memory_source', 'arguments': {'source_id': self.source_id, 'neighbors': 0}, 'text': None},
            {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic apples.'},
        ]
        result = actions[len(self.requests) - 1]
        return InferenceResult(json_value=result, raw_response=json.dumps(result), usage={'input_tokens': 100, 'output_tokens': 20})


@pytest.mark.parametrize('mode', ['acknowledged', 'source_revoked', 'owner_deactivated', 'deactivated_ack', 'expired_ack', 'reclaimed', 'history_failure', 'legacy_handoff'])
def test_real_owner_processor_to_peer_verified_one_time_pickup(setup, mode, tmp_path):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
    _, users, owner, queue, acquired, task, store, _ = setup
    conversation = SQLiteConversationStore(queue.db_path)
    event = conversation.record(owner_user_id=owner.user_id, role='user', content='Synthetic apples.', source='test', source_message_id='one')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, tools=registry,
        inference=InferenceRouter(provider=Provider('conversation:' + event.event_id), default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=conversation, cancellation_checker=lambda: False)
    daemon = V2Daemon.__new__(V2Daemon)
    daemon.runtime = SimpleNamespace(queue=queue)
    daemon._owner_checkpoint_lock = threading.Lock()
    daemon._owner_checkpoints = {}
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'unused.db')),
        authorize_claim=authorize_trusted_owner_claim, save_checkpoint=lambda *_: pytest.fail('legacy checkpoint called'),
        atomic_checkpoints=store, selected_draft_sink=daemon.register_owner_checkpoint)
    result = processor.process(task, context)
    assert result.status == ProcessingStatus.WAITING, result.error
    if mode == 'source_revoked':
        assert conversation.delete(event_id=event.event_id, owner_user_id=owner.user_id)
    if mode == 'owner_deactivated':
        users.update_user(owner.user_id, is_active=False)
    with tempfile.TemporaryDirectory(prefix='owner-pick-', dir='/private/tmp') as socket_dir:
        server = V2DaemonServer(daemon, socket_path=os.path.join(socket_dir, 'p.sock'), experimental_owner_ingress=True)
        server.start()
        try:
            client = V2DaemonClient(server.socket_path)
            with pytest.raises(RuntimeError, match='owner_desktop_pickup_payload_denied'):
                client.request('experimental_owner_take_draft', message_id=task.message_id, user=owner.user_id)
            if mode in {'source_revoked', 'owner_deactivated'}:
                with pytest.raises(RuntimeError, match='owner_recovery_unavailable'):
                    client.request('experimental_owner_take_draft', message_id=task.message_id)
            else:
                picked = client.request('experimental_owner_take_draft', message_id=task.message_id)
                assert picked['message_id'] == task.message_id and picked['message'] == 'Synthetic apples.'
                assert picked['delivery_status'] == 'started_unconfirmed'
                token = picked['acknowledgement_token']
                with pytest.raises(RuntimeError, match='owner_desktop_pickup_unavailable'):
                    client.request('experimental_owner_take_draft', message_id=task.message_id)
                with pytest.raises(RuntimeError, match='owner_desktop_ack_unavailable'):
                    client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token='f' * 32)
                if mode == 'deactivated_ack':
                    users.update_user(owner.user_id, is_active=False)
                    with pytest.raises(RuntimeError, match='owner_claim_authorization_unavailable'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                elif mode == 'expired_ack':
                    with queue._connect() as conn:
                        conn.execute("UPDATE v2_inbound_messages SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE message_id=?", (task.message_id,))
                    with pytest.raises(RuntimeError, match='claim_expired'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                elif mode == 'reclaimed':
                    # Deliberately unsafe SQL fixture simulates an old worker
                    # replacing the generation under the same lease owner.
                    with queue._connect() as conn:
                        conn.execute("UPDATE v2_inbound_messages SET claim_generation=? WHERE message_id=?",
                                     ('a' * 32, task.message_id))
                    with pytest.raises(RuntimeError, match='claim_not_current'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                    assert [(item.role, item.content) for item in conversation.list(owner_user_id=owner.user_id)] == [('user', 'Synthetic apples.')]
                elif mode == 'history_failure':
                    original = conversation.record
                    def fail_assistant(**kwargs):
                        if kwargs.get('role') == 'assistant':
                            raise ValueError('fixture_history_failure')
                        return original(**kwargs)
                    conversation.record = fail_assistant
                    with pytest.raises(RuntimeError, match='fixture_history_failure'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                    assert [(item.role, item.content) for item in conversation.list(owner_user_id=owner.user_id)] == [('user', 'Synthetic apples.')]
                elif mode == 'legacy_handoff':
                    with queue._connect() as conn:
                        conn.execute('INSERT INTO v2_scheduled_handoffs(message_id,task_id,owner_id) VALUES (?,?,?)',
                                     (task.message_id, task.task_id, task.user))
                    with pytest.raises(RuntimeError, match='owner_desktop_ack_unavailable'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                    assert [(item.role, item.content) for item in conversation.list(owner_user_id=owner.user_id)] == [('user', 'Synthetic apples.')]
                else:
                    assert client.request('experimental_owner_ack_draft', message_id=task.message_id,
                                          acknowledgement_token=token) == {'status': 'client_acknowledged'}
                    assert client.request('experimental_owner_ack_draft', message_id=task.message_id,
                                          acknowledgement_token=token) == {'status': 'client_acknowledged'}
                    # Idempotence uses the durable receipt, not an in-memory
                    # processor handoff, and is revoked with owner access.
                    assert task.message_id not in daemon._owner_checkpoints
                    restarted = V2Daemon.__new__(V2Daemon)
                    restarted.runtime = SimpleNamespace(queue=queue)
                    restarted._owner_checkpoint_lock = threading.Lock()
                    restarted._owner_checkpoints = {}
                    restarted.acknowledge_owner_desktop_draft(task.message_id, token)
                    events = conversation.list(owner_user_id=owner.user_id)
                    assert [(event.role, event.content) for event in events] == [
                        ('user', 'Synthetic apples.'), ('user', 'Find the apples note.'),
                        ('assistant', 'Synthetic apples.')]
                    users.update_user(owner.user_id, is_active=False)
                    with pytest.raises(RuntimeError, match='owner_desktop_ack_unavailable'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                    # User data cleanup must also remove the retained owner
                    # ingress proof after the queue row was finalized.
                    assert daemon._delete_user_inbound_data(owner.user_id) == 0
                    with queue._connect() as cleanup_conn:
                        assert cleanup_conn.execute('SELECT 1 FROM v2_trusted_owner_ingress WHERE message_id=?',
                                                    (task.message_id,)).fetchone() is None
                with queue._connect() as conn:
                    raw = json.loads(store._row(conn, task.task_id)['task_state_json'])
                assert raw['metadata']['owner_read_desktop_pickup_started']['channel'] == 'desktop'
                assert raw['metadata']['owner_read_desktop_pickup_started']['status'] == (
                    'started_unconfirmed' if mode in {'expired_ack', 'deactivated_ack', 'reclaimed', 'history_failure', 'legacy_handoff'} else 'client_acknowledged')
                with queue._connect() as conn:
                    still_queued = conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (task.message_id,)).fetchone()
                assert (still_queued is None) == (mode == 'acknowledged')
        finally:
            server.stop()

def test_real_owner_processor_uses_atomic_checkpoint_and_persists_reads(setup):
    tmp_path, users, owner, queue, acquired, task, store, _ = setup
    conversation = SQLiteConversationStore(tmp_path / 'conversation.db')
    event = conversation.record(owner_user_id=owner.user_id, role='user', content='Synthetic apples. Unique source 9fcb.', source='test', source_message_id='one')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    provider = Provider('conversation:' + event.event_id)
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=conversation, cancellation_checker=lambda: False)
    delivered_to_server = []
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'unused.db')),
        authorize_claim=authorize_trusted_owner_claim, save_checkpoint=lambda *_: pytest.fail('legacy checkpoint called'),
        atomic_checkpoints=store, selected_draft_sink=lambda checkpoint, context: delivered_to_server.append((checkpoint, context)))
    result = processor.process(task, context)
    assert result.status == ProcessingStatus.WAITING, result.error
    assert result.snapshot.metadata['draft'] == 'Synthetic apples.'
    assert len(delivered_to_server) == 1
    assert delivered_to_server[0][0].receipt == acquired.claim_receipt
    assert delivered_to_server[0][1] is context
    assert delivered_to_server[0][0].version > 0
    with queue._connect() as conn:
        row = store._row(conn, task.task_id)
        raw = json.loads(row['task_state_json'])
    state = raw['metadata']['single_loop_checkpoint']['state']
    observations = raw['metadata']['owner_read_observations']
    assert state['status'] == 'completed' and len(state['evidence_sources']) == 2
    assert set(observations) == set(state['evidence_sources'])
    assert all('observation' not in binding and len(binding['observation_sha256']) == 64
               for binding in observations.values())
    # The captured source text is not copied into durable evidence bindings.
    assert 'Synthetic apples.' not in json.dumps(observations)
    assert 'Unique source 9fcb.' not in row['task_state_json']
    assert raw['metadata']['single_loop_checkpoint']['budget']['reservations']
    assert 'owner_read_pending_action' not in raw['metadata']


def test_claim_loss_during_model_call_preserves_reservation_for_recovery(setup):
    tmp_path, users, _, queue, acquired, task, store, _ = setup
    class LostClaimProvider:
        supports_image_input = False
        def generate_json(self, request):
            with queue._connect() as conn:
                conn.execute("UPDATE v2_inbound_messages SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE message_id=?", (task.message_id,))
            decision = {'kind': 'answer', 'text': 'Synthetic.', 'tool_id': None, 'arguments': None}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 10, 'output_tokens': 5})
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, tools=registry,
        inference=InferenceRouter(provider=LostClaimProvider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=SQLiteConversationStore(tmp_path / 'conversation.db'), cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'unused.db')),
        authorize_claim=authorize_trusted_owner_claim, save_checkpoint=lambda *_: pytest.fail('legacy checkpoint called'),
        atomic_checkpoints=store)
    result = processor.process(task, context)
    assert result.status == ProcessingStatus.FAILED
    with queue._connect() as conn:
        raw = json.loads(store._row(conn, task.task_id)['task_state_json'])
    state = raw['metadata']['single_loop_checkpoint']['state']
    pending = raw['metadata']['owner_read_pending_action']
    assert state['inflight'] == pending['reservation_id'] and pending['kind'] == 'model'
    assert raw['metadata']['single_loop_checkpoint']['budget']['reservations'][state['inflight']]['status'] == 'reserved'


def test_claim_loss_immediately_before_dispatch_never_calls_provider(setup, monkeypatch):
    tmp_path, users, _, queue, acquired, task, store, _ = setup
    provider = Provider('conversation:unused')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=SQLiteConversationStore(tmp_path / 'conversation.db'), cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'unused.db')),
        authorize_claim=authorize_trusted_owner_claim, save_checkpoint=lambda *_: pytest.fail('legacy checkpoint called'),
        atomic_checkpoints=store)
    monkeypatch.setattr(OwnerAtomicCheckpoint, 'check_claim', lambda *_: False)
    result = processor.process(task, context)
    assert result.status == ProcessingStatus.FAILED and provider.requests == []
    with queue._connect() as conn:
        raw = json.loads(store._row(conn, task.task_id)['task_state_json'])
    assert raw['metadata']['single_loop_checkpoint']['state']['inflight']


def test_terminal_draft_is_bound_to_exact_selected_decision_and_projection(setup):
    import hashlib
    _, _, _, queue, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = atomic.reserve(expected_version=start.version, reservation_id='model-selection', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    decision = {'kind': 'answer', 'text': 'selected fixture', 'usage': {'input_tokens': 3, 'output_tokens': 2}}
    projection = 'a' * 64
    atomic.settle_model(expected_version=reserved.version, reservation_id='model-selection', outcome='success',
                        input_tokens=3, output_tokens=2, decision=decision, projection_fingerprint=projection)
    state = CandidateState(task.task_id, task.user, 'goal', status='completed', response='selected fixture')
    with pytest.raises(CheckpointConflict, match='selected_decision_mismatch'):
        atomic.save_state(state, consume_decision=True, selected_decision={**decision, 'text': 'changed'},
                          projection_fingerprint=projection)
    with pytest.raises(CheckpointConflict, match='selected_decision_mismatch'):
        atomic.save_state(state, consume_decision=True, selected_decision=decision,
                          projection_fingerprint='b' * 64)
    result = atomic.save_state(state, consume_decision=True, selected_decision=decision,
                               projection_fingerprint=projection)
    with queue._connect() as conn:
        raw = json.loads(atomic.store._row(conn, task.task_id)['task_state_json'])
    binding = raw['metadata']['owner_read_selected_draft']
    assert binding['answer_sha256'] == hashlib.sha256(b'selected fixture').hexdigest()
    assert binding['projection_sha256'] == projection and binding['model_reservation_id'] == 'model-selection'
    assert binding['evidence_refs'] == [] and 'owner_read_pending_decision' not in raw['metadata']
    assert result.state.response == 'selected fixture'


def test_selected_tool_arguments_are_bound_to_model_decision(setup):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = atomic.reserve(expected_version=start.version, reservation_id='model-selection', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    decision = {'kind': 'tool', 'tool_id': 'native.read_memory_source',
                'arguments': {'source_id': 'conversation:fixture'}}
    projection = 'a' * 64
    settled = atomic.settle_model(expected_version=reserved.version, reservation_id='model-selection',
        outcome='success', input_tokens=3, output_tokens=2, decision=decision, projection_fingerprint=projection)
    tool_args = dict(expected_version=settled.version, reservation_id='read', kind='tool',
        amounts={'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}, now_ms=102,
        tool_id='native.read_memory_source', selected_decision=decision, projection_fingerprint=projection)
    with pytest.raises(CheckpointConflict, match='pending_decision_unresolved'):
        atomic.reserve(**tool_args, arguments={'source_id': 'conversation:other'})
    result = atomic.reserve(**tool_args, arguments=decision['arguments'])
    assert result.state.inflight == 'read'


def test_lost_inflight_clock_can_be_explicitly_abandoned(setup):
    _, _, _, queue, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserve_selected_tool(atomic, start, 'native.owner_clock', {})
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE message_id=?", (task.message_id,))
    assert queue.abandon_trusted_owner_message(task.message_id) == {
        'status': 'abandoned', 'delivery_status': 'no_pickup_recorded'}
