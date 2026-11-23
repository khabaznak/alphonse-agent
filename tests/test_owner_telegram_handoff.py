"""Telegram handoff uses a real synthetic claim and checkpoint, never Desktop pickup."""
from datetime import datetime, timezone
import json

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_delivery import ensure_schema
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_handoff import (
    stage_telegram_reply, dispatch_telegram_reply, finalize_telegram_reply, verify_finalized_telegram_reply,
)
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import enqueue_telegram_owner
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, TaskBudget
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def ready(tmp_path):
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture', users_root=tmp_path / 'users')
    users.bind_address(user_id=owner.user_id, integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
    conversation = SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    questions = SQLiteQuestionStore(path)
    with queue._connect() as conn:
        conn.execute('''CREATE TABLE v2_integrations (integration_id TEXT PRIMARY KEY, provider_key TEXT, enabled INTEGER,
            config_json TEXT, credential_generation INTEGER, revocation_pending INTEGER, credential_deletion_pending INTEGER,
            updated_at TEXT NOT NULL)''')
        conn.execute('INSERT INTO v2_integrations VALUES (?,?,?,?,?,?,?,?)', ('telegram-test', 'telegram', 1, '{}', 1, 0, 0, 'revision-1'))
        ensure_schema(conn)
    update = {'update_id': 42, 'message': {'message_id': 9, 'text': 'hello',
              'from': {'id': 123}, 'chat': {'id': 123, 'type': 'private'}}}
    message_id = enqueue_telegram_owner(queue, integration_id='telegram-test', update=update)
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    store = ExperimentalCheckpointStore(questions)
    atomic = OwnerAtomicCheckpoint(queue, store, task, acquired.claim_receipt, clock=lambda: datetime.now(timezone.utc))
    vector = lambda a, m, i, o: {'actions': a, 'model_calls': m, 'input_tokens': i, 'output_tokens': o}
    budget = TaskBudget(BudgetLimits(vector(3, 3, 1000, 300), vector(0, 1, 100, 50), 100000, 0), started_at_ms=100)
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), budget)
    reserved = atomic.reserve(expected_version=start.version, reservation_id='model-selection', kind='model',
        amounts=vector(0, 1, 100, 50), now_ms=101)
    decision = {'kind': 'answer', 'text': 'Hello from Alphonse', 'usage': {'input_tokens': 3, 'output_tokens': 2}}
    projection = 'a' * 64
    atomic.settle_model(expected_version=reserved.version, reservation_id='model-selection', outcome='success',
                        input_tokens=3, output_tokens=2, decision=decision, projection_fingerprint=projection)
    atomic.save_state(CandidateState(task.task_id, task.user, 'goal', status='completed', response=decision['text']),
                      consume_decision=True, selected_decision=decision, projection_fingerprint=projection)
    atomic.mark_ready_handoff()
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt,
        conversation_store=conversation, cancellation_checker=lambda: False)
    return queue, atomic, context


def test_exact_selected_draft_to_verified_telegram_recipient(ready):
    queue, atomic, context = ready
    assert stage_telegram_reply(atomic, context) == atomic.task.message_id
    sent = []
    assert dispatch_telegram_reply(atomic, context, send=lambda **kw: sent.append(kw) or '71') == 'provider_accepted'
    assert sent == [{'integration_id': 'telegram-test', 'chat_id': '123', 'text': 'Hello from Alphonse'}]
    with queue._connect() as conn:
        raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                      (atomic.task.task_id,)).fetchone()[0])
        assert 'owner_read_desktop_pickup_started' not in raw['metadata']


def test_revocation_after_staging_blocks_sending(ready):
    queue, atomic, context = ready
    stage_telegram_reply(atomic, context)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_user_addresses SET is_active=0')
    sent = []
    with pytest.raises(Exception):
        dispatch_telegram_reply(atomic, context, send=lambda **kw: sent.append(kw))
    assert not sent


def test_uncertain_provider_never_gets_automatic_resend(ready):
    _, atomic, context = ready
    stage_telegram_reply(atomic, context)
    sent = []
    def timeout(**kwargs):
        sent.append(kwargs)
        raise TimeoutError()
    assert dispatch_telegram_reply(atomic, context, send=timeout) == 'started_unconfirmed'
    assert dispatch_telegram_reply(atomic, context, send=timeout) == 'started_unconfirmed'
    assert len(sent) == 1


@pytest.mark.parametrize('change', ['draft', 'evidence'])
def test_altered_selected_draft_or_source_bindings_never_dispatch(ready, change):
    queue, atomic, context = ready
    stage_telegram_reply(atomic, context)
    with queue._connect() as conn:
        row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                           (atomic.task.task_id,)).fetchone()
        raw = json.loads(row[0])
        if change == 'draft':
            raw['metadata']['owner_read_selected_draft']['answer_sha256'] = '0' * 64
        else:
            raw['metadata']['owner_read_observations']['forged'] = {'tool_id': 'native.read_memory_source'}
        conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?',
                     (json.dumps(raw), atomic.task.task_id))
    sent = []
    with pytest.raises(Exception):
        dispatch_telegram_reply(atomic, context, send=lambda **kw: sent.append(kw))
    assert not sent


def test_provider_accepted_reply_finalizes_once_and_is_verifiable_after_restart(ready):
    queue, atomic, context = ready
    stage_telegram_reply(atomic, context)
    assert dispatch_telegram_reply(atomic, context, send=lambda **_: '71') == 'provider_accepted'
    finalize_telegram_reply(atomic, context)
    assert verify_finalized_telegram_reply(queue, atomic.task.message_id)
    restarted_queue = SQLiteMessageQueue(queue.db_path, experimental_claim_fencing=True)
    assert verify_finalized_telegram_reply(restarted_queue, atomic.task.message_id)
    finalize_telegram_reply(atomic, context)
    with queue._connect() as conn:
        assert conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (atomic.task.message_id,)).fetchone() is None
        assert conn.execute('SELECT count(*) FROM v2_conversation_events WHERE source=? AND role=?', ('telegram', 'assistant')).fetchone()[0] == 1


def test_unconfirmed_send_is_not_finalized_or_retried(ready):
    queue, atomic, context = ready
    stage_telegram_reply(atomic, context)
    assert dispatch_telegram_reply(atomic, context, send=lambda **_: '') == 'started_unconfirmed'
    with pytest.raises(Exception):
        finalize_telegram_reply(atomic, context)
    assert not verify_finalized_telegram_reply(queue, atomic.task.message_id)
    with queue._connect() as conn:
        assert conn.execute('SELECT status FROM v2_owner_telegram_replies WHERE message_id=?',
                            (atomic.task.message_id,)).fetchone()[0] == 'started_unconfirmed'


def test_finalized_receipt_invalidates_if_origin_revoked(ready):
    queue, atomic, context = ready
    stage_telegram_reply(atomic, context)
    dispatch_telegram_reply(atomic, context, send=lambda **_: '71')
    finalize_telegram_reply(atomic, context)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_user_addresses SET is_active=0')
    assert not verify_finalized_telegram_reply(queue, atomic.task.message_id)


def test_telegram_claim_cannot_use_desktop_draft_pickup(ready):
    _, atomic, context = ready
    with pytest.raises(Exception):
        atomic.take_desktop_draft(context)


def test_accepted_but_unfinalized_reply_survives_restart_without_resending(ready):
    queue, atomic, context = ready
    stage_telegram_reply(atomic, context)
    assert dispatch_telegram_reply(atomic, context, send=lambda **_: '71') == 'provider_accepted'
    restarted_queue = SQLiteMessageQueue(queue.db_path, experimental_claim_fencing=True)
    assert not verify_finalized_telegram_reply(restarted_queue, atomic.task.message_id)
    with restarted_queue._connect() as conn:
        assert conn.execute('SELECT status FROM v2_owner_telegram_replies WHERE message_id=?',
                            (atomic.task.message_id,)).fetchone()[0] == 'provider_accepted'
        assert conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?',
                            (atomic.task.message_id,)).fetchone() is not None
