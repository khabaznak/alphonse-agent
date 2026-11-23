"""Scheduled Telegram replies use the genuine origin and live scheduled claim."""
from datetime import timezone
import pytest

from test_owner_telegram_handoff import ready
from test_owner_telegram_schedule_origin import telegram_schedule
from test_owner_schedule_origin import fixture as desktop_schedule
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_origin import stage_due_occurrence_locked
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_delivery import TelegramReplyUnavailable
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_scheduled_delivery import (
    is_telegram_scheduled_origin, stage_scheduled_telegram_reply, dispatch_scheduled_telegram_reply,
)
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, TaskBudget
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.questions import SQLiteQuestionStore


@pytest.fixture
def scheduled_ready(telegram_schedule):
    queue, record, due = telegram_schedule
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        mid = stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)
    queue._owner_schedule_clock = lambda: due
    acquired = queue.claim_next(selector=MessageSelector(message_id=mid), lease_owner='synthetic-schedule', lease_seconds=600)
    task = TaskState.from_message(acquired.message, message_id=mid)
    task.task_id = mid
    atomic = OwnerAtomicCheckpoint(queue, ExperimentalCheckpointStore(SQLiteQuestionStore(queue.db_path)),
                                   task, acquired.claim_receipt, clock=lambda: due.astimezone(timezone.utc))
    vec = lambda a, m, i, o: {'actions': a, 'model_calls': m, 'input_tokens': i, 'output_tokens': o}
    budget = TaskBudget(BudgetLimits(vec(3, 3, 1000, 300), vec(0, 1, 100, 50), 100000, 0), started_at_ms=100)
    initial = atomic.initialize(CandidateState(mid, task.user, 'goal'), budget)
    reservation = atomic.reserve(expected_version=initial.version, reservation_id='decision', kind='model',
                                 amounts=vec(0, 1, 100, 50), now_ms=101)
    decision = {'kind': 'answer', 'text': 'Synthetic result', 'usage': {'input_tokens': 3, 'output_tokens': 2}}
    projection = 'a' * 64
    atomic.settle_model(expected_version=reservation.version, reservation_id='decision', outcome='success',
                        input_tokens=3, output_tokens=2, decision=decision, projection_fingerprint=projection)
    atomic.save_state(CandidateState(mid, task.user, 'goal', status='completed', response=decision['text']),
                      consume_decision=True, selected_decision=decision, projection_fingerprint=projection)
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt,
        conversation_store=SQLiteConversationStore(queue.db_path), cancellation_checker=lambda: False)
    return queue, atomic, context


def test_scheduled_result_sent_to_original_private_owner_at_most_once(scheduled_ready):
    queue, atomic, context = scheduled_ready
    assert is_telegram_scheduled_origin(atomic) is True
    assert stage_scheduled_telegram_reply(atomic, context) == atomic.task.message_id
    sent = []
    send = lambda **values: sent.append(values) or '72'
    assert dispatch_scheduled_telegram_reply(atomic, context, send=send) == 'provider_accepted'
    assert dispatch_scheduled_telegram_reply(atomic, context, send=send) == 'provider_accepted'
    assert sent == [{'integration_id': 'telegram-test', 'chat_id': '123', 'text': 'Synthetic result'}]
    with queue._connect() as conn:
        assert conn.execute('SELECT status FROM v2_owner_telegram_replies WHERE message_id=?',
                            (atomic.task.message_id,)).fetchone()['status'] == 'provider_accepted'


def test_uncertain_scheduled_send_is_never_repeated(scheduled_ready):
    queue, atomic, context = scheduled_ready
    stage_scheduled_telegram_reply(atomic, context)
    sent = []
    def timeout(**values):
        sent.append(values)
        raise TimeoutError()
    assert dispatch_scheduled_telegram_reply(atomic, context, send=timeout) == 'started_unconfirmed'
    assert dispatch_scheduled_telegram_reply(atomic, context, send=timeout) == 'started_unconfirmed'
    assert len(sent) == 1
    with queue._connect() as conn:
        assert conn.execute('SELECT status FROM v2_owner_telegram_replies WHERE message_id=?',
                            (atomic.task.message_id,)).fetchone()['status'] == 'started_unconfirmed'


def test_missing_occurrence_is_unknown_not_non_telegram(scheduled_ready):
    queue, atomic, _ = scheduled_ready
    with queue._connect() as conn:
        conn.execute('DELETE FROM v2_owner_schedule_occurrences WHERE message_id=?', (atomic.task.message_id,))
    assert is_telegram_scheduled_origin(atomic) is None


@pytest.mark.parametrize('drop_telegram_table', [False, True])
def test_verified_existing_desktop_schedule_stays_desktop(desktop_schedule, drop_telegram_table):
    queue, record, due = desktop_schedule
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        mid = stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)
        if drop_telegram_table:
            conn.execute('DROP TABLE v2_owner_schedule_telegram_origins')
    queue._owner_schedule_clock = lambda: due
    acquired = queue.claim_next(selector=MessageSelector(message_id=mid), lease_owner='desktop-schedule', lease_seconds=600)
    task = TaskState.from_message(acquired.message, message_id=mid)
    task.task_id = mid
    atomic = OwnerAtomicCheckpoint(queue, ExperimentalCheckpointStore(SQLiteQuestionStore(queue.db_path)),
                                   task, acquired.claim_receipt, clock=lambda: due)
    assert is_telegram_scheduled_origin(atomic) is False
    with queue._connect() as conn:
        conn.execute("UPDATE v2_trusted_owner_ingress SET payload_sha256='tampered'")
    assert is_telegram_scheduled_origin(atomic) is None


def test_missing_telegram_origin_table_does_not_misclassify_telegram(scheduled_ready):
    queue, atomic, _ = scheduled_ready
    with queue._connect() as conn:
        conn.execute('DROP TABLE v2_owner_schedule_telegram_origins')
    assert is_telegram_scheduled_origin(atomic) is None


@pytest.mark.parametrize('statement', ["UPDATE v2_user_addresses SET is_active=0",
    "UPDATE v2_scheduled_tasks SET status='cancelled'",
    "UPDATE v2_owner_telegram_replies SET chat_id='999' WHERE text='Synthetic result'"])
def test_revoked_cancelled_or_changed_destination_cannot_send(scheduled_ready, statement):
    queue, atomic, context = scheduled_ready
    stage_scheduled_telegram_reply(atomic, context)
    with queue._connect() as conn:
        conn.execute(statement)
    assert is_telegram_scheduled_origin(atomic) is True
    sent = []
    with pytest.raises((TelegramReplyUnavailable, OwnerRecoveryUnavailable)):
        dispatch_scheduled_telegram_reply(atomic, context, send=lambda **values: sent.append(values) or '72')
    assert sent == []
