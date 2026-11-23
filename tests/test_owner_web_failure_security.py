"""Web failure observations retain authorization and never replay a read."""
from datetime import datetime, timezone
import json
import os

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable, restore_owner_read_observations
from alphonse.agent_v2.core.intelligence.simple_loop.owner_web import web_failure_observation
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, TaskBudget
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.core.tools.registry.native.web import build_web_search_tool_definition
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.web_tools_settings import WebToolsSettings


@pytest.fixture
def failed_web_checkpoint(tmp_path):
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversations = SQLiteConversationStore(path)
    message = queue.enqueue_trusted_local_owner(prompt='Find a synthetic report.', peer_uid=os.getuid())
    acquired = queue.claim_next(lease_seconds=60)
    task = TaskState.from_message(acquired.message, message_id=message)
    task.task_id = message
    calls = []
    tools = InMemoryToolRegistry()
    tool = build_web_search_tool_definition(WebToolsSettings(enabled=True, searxng_base_url='https://example.test'))
    tools.register(ToolDefinition(tool.descriptor, lambda args: calls.append(args), tool.argument_schema))
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, tools=tools,
                              conversation_store=conversations, cancellation_checker=lambda: False)
    now = datetime.now(timezone.utc)
    atomic = OwnerAtomicCheckpoint(queue, ExperimentalCheckpointStore(SQLiteQuestionStore(path)),
                                   task, acquired.claim_receipt, clock=lambda: now)
    limits = dict(actions=5, model_calls=8, input_tokens=1000, output_tokens=1000)
    zero = dict(actions=0, model_calls=0, input_tokens=0, output_tokens=0)
    start = atomic.initialize(CandidateState(message, owner.user_id, 'goal'),
        TaskBudget(BudgetLimits(limits, zero, 100000, 0), started_at_ms=int(now.timestamp() * 1000)))
    arguments = {'query': 'synthetic report'}
    selected = {'kind': 'tool', 'tool_id': 'native.web_search', 'arguments': arguments}
    fingerprint = 'a' * 64
    model = atomic.reserve(expected_version=start.version, reservation_id='model-1', kind='model',
        amounts=dict(actions=0, model_calls=1, input_tokens=100, output_tokens=20), now_ms=start.budget.started_at_ms)
    decision = atomic.settle_model(expected_version=model.version, reservation_id='model-1', outcome='success',
        input_tokens=10, output_tokens=10, decision=selected, projection_fingerprint=fingerprint)
    reserved = atomic.reserve(expected_version=decision.version, reservation_id='read-1', kind='tool',
        amounts=dict(actions=1, model_calls=0, input_tokens=0, output_tokens=0),
        now_ms=start.budget.started_at_ms, tool_id='native.web_search', arguments=arguments,
        selected_decision=selected, projection_fingerprint=fingerprint)
    outcome = {'status': 'failed', 'error': {'code': 'searxng_http_error', 'message': 'private-url'},
        'result': {'output': None, 'exception': {'code': 'searxng_http_error', 'message': 'private-url',
                                                'retryable': True, 'status_code': 503}}}
    observation = web_failure_observation('native.web_search', outcome)
    assert observation is not None
    saved = atomic.settle_read(expected_version=reserved.version, reservation_id='read-1',
        tool_id='native.web_search', arguments=arguments, evidence_ref='owner-read:failure',
        observation=observation)
    return atomic, context, saved, calls


def test_failed_web_observation_restores_without_external_retry(failed_web_checkpoint):
    atomic, context, saved, calls = failed_web_checkpoint
    checkpoint, bindings = restore_owner_read_observations(atomic, context, expected_version=saved.version)
    assert checkpoint.state.evidence_sources == ['owner-read:failure']
    assert bindings['owner-read:failure']['observation']['error_code'] == 'searxng_http_error'
    assert checkpoint.budget.to_dict()['reservations']['read-1']['outcome'] == 'failure'
    assert calls == []
    with atomic.store.store._connect() as conn:
        raw = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                           (atomic.task.task_id,)).fetchone()['task_state_json']
    assert 'private-url' not in raw


def test_interrupted_later_web_read_is_not_replayed(failed_web_checkpoint):
    atomic, context, saved, calls = failed_web_checkpoint
    arguments = {'query': 'another synthetic source'}
    decision = {'kind': 'tool', 'tool_id': 'native.web_search', 'arguments': arguments}
    model = atomic.reserve(expected_version=saved.version, reservation_id='model-2', kind='model',
        amounts=dict(actions=0, model_calls=1, input_tokens=100, output_tokens=20), now_ms=saved.budget.started_at_ms)
    selected = atomic.settle_model(expected_version=model.version, reservation_id='model-2',
        outcome='success', input_tokens=10, output_tokens=10, decision=decision, projection_fingerprint='b' * 64)
    pending = atomic.reserve(expected_version=selected.version, reservation_id='read-2', kind='tool',
        amounts=dict(actions=1, model_calls=0, input_tokens=0, output_tokens=0),
        now_ms=saved.budget.started_at_ms, tool_id='native.web_search', arguments=arguments,
        selected_decision=decision, projection_fingerprint='b' * 64)
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, context, expected_version=pending.version)
    assert calls == []


@pytest.mark.parametrize('restriction', ['cancel', 'authority_revoked', 'tool_disabled', 'tampered'])
def test_failed_web_observation_does_not_survive_revocation_or_tampering(failed_web_checkpoint, restriction):
    atomic, context, saved, calls = failed_web_checkpoint
    if restriction == 'cancel':
        context.cancellation_checker = lambda: True
    elif restriction == 'authority_revoked':
        context._owner_authority_required = True
        context._owner_authority_check = lambda: False
    elif restriction == 'tool_disabled':
        descriptor = context.tools.get('native.web_search')
        context.tools.register(ToolDefinition(descriptor, lambda _: None, descriptor.argument_schema, enabled=False))
    else:
        with atomic.store.store._connect() as conn:
            row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                               (atomic.task.task_id,)).fetchone()
            raw = json.loads(row['task_state_json'])
            raw['metadata']['owner_read_observations']['owner-read:failure']['web_observation']['error_code'] = 'searxng_timeout'
            conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?',
                         (json.dumps(raw), atomic.task.task_id))
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, context, expected_version=saved.version)
    assert calls == []
