"""Draft bounded-steer ingress tests; no IPC exposure yet."""
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, TaskBudget
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.owner_steering import submit, remove_steering, list_steering
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def reserved(tmp_path):
    path = tmp_path / 'steer.db'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, lease_owner='fixture', experimental_claim_fencing=True)
    SQLiteConversationStore(path)
    questions = SQLiteQuestionStore(path)
    message_id = queue.enqueue_trusted_local_owner(prompt='Find a note', peer_uid=os.getuid())
    acquired = queue.claim_next(lease_owner='fixture')
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    atomic = OwnerAtomicCheckpoint(queue, ExperimentalCheckpointStore(questions), task,
        acquired.claim_receipt, clock=lambda: datetime.now(timezone.utc))
    vector = lambda a, m, i, o: {'actions': a, 'model_calls': m, 'input_tokens': i, 'output_tokens': o}
    budget = TaskBudget(BudgetLimits(vector(3, 3, 1000, 300), vector(0, 1, 100, 50), 100000, 0), started_at_ms=100)
    atomic.initialize(CandidateState(message_id, owner.user_id, 'goal'), budget)
    return queue, owner, message_id, atomic


def send(fixture, text='Use the blue note', request_id='a' * 32):
    queue, owner, message_id, _ = fixture
    return submit(queue, owner=owner.user_id, peer_uid=os.getuid(), daemon_id='fixture',
                  message_id=message_id, request_id=request_id, content=text)


def test_acceptance_only_during_exact_live_model_reservation(reserved):
    queue, _, _, atomic = reserved
    assert send(reserved)['status'] == 'not_accepted'
    atomic.reserve(expected_version=atomic.version, reservation_id='model-test', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    assert send(reserved)['status'] == 'accepted'
    assert send(reserved)['status'] == 'accepted'
    with pytest.raises(ValueError):
        send(reserved, text='Different text')
    with queue._connect() as conn:
        assert conn.execute('SELECT count(*) FROM v2_owner_steering').fetchone()[0] == 1


def test_steer_invalidates_outstanding_decision_and_settles_budget(reserved):
    queue, _, message_id, atomic = reserved
    atomic.reserve(expected_version=atomic.version, reservation_id='model-1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    assert send(reserved)['status'] == 'accepted'
    result = atomic.settle_model(expected_version=atomic.version, reservation_id='model-1', outcome='success',
        input_tokens=5, output_tokens=2, decision={'kind': 'answer', 'text': 'stale'}, projection_fingerprint='a' * 64)
    assert atomic.steering_decision_stale is True
    assert result.budget.usage()['in_flight'] == []
    assert atomic.steering_context() == ['Use the blue note']
    with queue._connect() as conn:
        import json
        raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (message_id,)).fetchone()[0])
        assert 'owner_read_pending_decision' not in raw['metadata']
    assert send(reserved, request_id='b' * 32)['status'] == 'not_accepted'
    atomic.reserve(expected_version=atomic.version, reservation_id='model-2', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=102,
        projection_fingerprint='b' * 64)
    atomic.settle_model(expected_version=atomic.version, reservation_id='model-2', outcome='success',
        input_tokens=5, output_tokens=2, decision={'kind': 'answer', 'text': 'fresh'}, projection_fingerprint='b' * 64)
    assert atomic.steering_decision_stale is False
    with queue._connect() as conn:
        assert conn.execute('SELECT status FROM v2_owner_steering').fetchone()[0] == 'accepted'
    with pytest.raises(ValueError):
        atomic.mark_steering_included({'kind': 'answer', 'text': 'fresh'}, 'b' * 64, lambda: True)
    with pytest.raises(ValueError):
        atomic.mark_steering_included({'kind': 'answer', 'text': 'different'}, 'b' * 64, lambda: False)
    atomic.mark_steering_included({'kind': 'answer', 'text': 'fresh'}, 'b' * 64, lambda: False)
    with queue._connect() as conn:
        assert conn.execute('SELECT status FROM v2_owner_steering').fetchone()[0] == 'included_in_planning'


def test_model_settlement_wins_race_and_refuses_late_steer(reserved):
    _, _, _, atomic = reserved
    atomic.reserve(expected_version=atomic.version, reservation_id='model-done', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    atomic.settle_model(expected_version=atomic.version, reservation_id='model-done', outcome='success',
        input_tokens=5, output_tokens=2, decision={'kind': 'answer', 'text': 'finished'}, projection_fingerprint='a' * 64)
    assert send(reserved)['status'] == 'not_accepted'


def test_steer_fails_closed_after_identity_revocation(reserved):
    queue, owner, _, atomic = reserved
    atomic.reserve(expected_version=atomic.version, reservation_id='model-1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner.user_id,))
    with pytest.raises(PermissionError):
        send(reserved)


def test_explicit_removal_waits_for_request_and_erases_plaintext(reserved):
    queue, owner, message_id, atomic = reserved
    atomic.reserve(expected_version=atomic.version, reservation_id='model-1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    assert send(reserved)['status'] == 'accepted'
    with pytest.raises(ValueError, match='owner_steer_request_unresolved'):
        remove_steering(queue, owner=owner.user_id, peer_uid=os.getuid(), steering_id='a' * 32)
    with queue._connect() as conn:
        conn.execute('DELETE FROM v2_inbound_messages WHERE message_id=?', (message_id,))
    assert remove_steering(queue, owner=owner.user_id, peer_uid=os.getuid(), steering_id='a' * 32)['status'] == 'removed'
    assert list_steering(queue, owner=owner.user_id, peer_uid=os.getuid())['items'] == []
    with queue._connect() as conn:
        assert conn.execute('SELECT content FROM v2_owner_steering').fetchone()[0] == ''
    assert send(reserved)['status'] == 'removed'


def test_removed_steer_tombstone_does_not_block_turn_deletion(reserved):
    queue, owner, message_id, atomic = reserved
    atomic.reserve(expected_version=atomic.version, reservation_id='model-1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    send(reserved)
    conversation = SQLiteConversationStore(queue.db_path)
    event = conversation.event_for_source_message_id(owner_user_id=owner.user_id, source_message_id='inbound:' + message_id)
    with pytest.raises(ValueError, match='owner_conversation_deletion_requires_reconciliation'):
        conversation.delete(event_id=event.event_id, owner_user_id=owner.user_id)
    with queue._connect() as conn:
        conn.execute('DELETE FROM v2_inbound_messages WHERE message_id=?', (message_id,))
        conn.execute('DELETE FROM v2_task_checkpoints WHERE task_id=?', (message_id,))
    remove_steering(queue, owner=owner.user_id, peer_uid=os.getuid(), steering_id='a' * 32)
    assert conversation.delete(event_id=event.event_id, owner_user_id=owner.user_id) is True


def test_steer_and_draft_reconciliation_are_atomic(reserved):
    from alphonse.agent_v2.owner_composer_drafts import OwnerComposerDrafts
    queue, owner, message_id, atomic = reserved
    atomic.reserve(expected_version=atomic.version, reservation_id='model-1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    drafts = OwnerComposerDrafts(queue, owner.user_id, os.getuid())
    drafts.save('a' * 32, message_id, 'Use the blue note')
    barrier = threading.Barrier(2)
    def submit_race():
        barrier.wait()
        try:
            return send(reserved)['status']
        except ValueError:
            return 'closed'
    def reconcile_race():
        barrier.wait()
        return drafts.reconcile('a' * 32)['resolution']
    with ThreadPoolExecutor(max_workers=2) as pool:
        submission = pool.submit(submit_race)
        reconciliation = pool.submit(reconcile_race)
        outcomes = (submission.result(), reconciliation.result())
    assert outcomes in {('accepted', 'submitted'), ('closed', 'not_submitted')}


def test_steer_and_queue_cannot_both_claim_same_attempt(reserved):
    from alphonse.agent_v2.owner_followups import OwnerFollowups
    queue, owner, message_id, atomic = reserved
    atomic.reserve(expected_version=atomic.version, reservation_id='model-1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    followups = OwnerFollowups(queue, owner.user_id, os.getuid())
    barrier = threading.Barrier(2)
    def steer():
        barrier.wait()
        try:
            return send(reserved)['status']
        except ValueError:
            return 'closed'
    def enqueue():
        barrier.wait()
        try:
            return followups.add(message_id, 'Use the blue note', 'a' * 32)['status']
        except ValueError:
            return 'closed'
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(steer), pool.submit(enqueue)
        outcomes = (first.result(), second.result())
    assert outcomes in {('accepted', 'closed'), ('closed', 'queued')}
