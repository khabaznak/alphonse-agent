"""A Telegram-origin schedule stays bound to its authenticated owner and chat."""
from datetime import datetime, timedelta, timezone

import pytest

from test_owner_telegram_handoff import ready
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_origin import (
    create_schedule_origin_locked, stage_due_occurrence_locked,
    verified_schedule_origin_locked, verified_telegram_schedule_destination_locked,
)
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_handoff import (
    stage_telegram_reply, dispatch_telegram_reply, finalize_telegram_reply,
)
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore


@pytest.fixture
def telegram_schedule(ready):
    queue, atomic, context = ready
    now = datetime.now(timezone.utc)
    due = now + timedelta(minutes=5)
    schedules = ScheduledTaskStore(queue.db_path)
    with atomic._transaction() as conn:
        record = schedules.create_task(owner_user_id=atomic.task.user, name='Synthetic task',
            prompt=atomic.task.goal, schedule_kind='once', run_at=due.isoformat(), delivery_mode='pdca',
            idempotency_key='owner-schedule:' + atomic.task.message_id, now=now, connection=conn)
        create_schedule_origin_locked(conn, atomic.task, record)
        destination = verified_telegram_schedule_destination_locked(conn, record.scheduled_task_id)
        assert destination['owner_id'] == atomic.task.user
        assert destination['chat_id'] == '123'
    stage_telegram_reply(atomic, context)
    assert dispatch_telegram_reply(atomic, context, send=lambda **_: '71') == 'provider_accepted'
    finalize_telegram_reply(atomic, context)
    return queue, record, due


def test_telegram_schedule_remains_bound_after_reply_and_stages_once(telegram_schedule):
    queue, record, due = telegram_schedule
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        origin = verified_schedule_origin_locked(conn, record.scheduled_task_id)
        assert origin is not None
        assert verified_telegram_schedule_destination_locked(conn, record.scheduled_task_id)['chat_id'] == '123'
        occurrence = stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)
        assert stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due) == occurrence
        prompt = conn.execute('SELECT prompt FROM v2_inbound_messages WHERE message_id=?', (occurrence,)).fetchone()[0]
        assert prompt.endswith('Original request:\n' + record.prompt)
        assert 'delivery is handled separately and is not guaranteed' in prompt


def test_unconfirmed_confirmation_does_not_cancel_scheduled_work(ready):
    queue, atomic, context = ready
    now = datetime.now(timezone.utc)
    due = now + timedelta(minutes=5)
    store = ScheduledTaskStore(queue.db_path)
    with atomic._transaction() as conn:
        record = store.create_task(owner_user_id=atomic.task.user, name='Synthetic task',
            prompt=atomic.task.goal, schedule_kind='once', run_at=due.isoformat(), delivery_mode='pdca',
            idempotency_key='owner-schedule:' + atomic.task.message_id, now=now, connection=conn)
        create_schedule_origin_locked(conn, atomic.task, record)
    stage_telegram_reply(atomic, context)
    assert dispatch_telegram_reply(atomic, context, send=lambda **_: '') == 'started_unconfirmed'
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        assert conn.execute('SELECT status FROM v2_owner_telegram_replies').fetchone()[0] == 'started_unconfirmed'
        assert verified_schedule_origin_locked(conn, record.scheduled_task_id) is not None
        assert stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)


def test_revoked_channel_schedule_is_shown_uncertain_without_changing_it(telegram_schedule):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_recovery import list_schedule_attention
    queue, record, _ = telegram_schedule
    with queue._connect() as conn:
        conn.execute('UPDATE v2_user_addresses SET is_active=0')
    assert list_schedule_attention(queue, record.owner_user_id)[record.scheduled_task_id] == 'uncertain'
    with queue._connect() as conn:
        assert conn.execute('SELECT status FROM v2_scheduled_tasks WHERE scheduled_task_id=?',
                            (record.scheduled_task_id,)).fetchone()[0] == 'active'


@pytest.mark.parametrize('statement', ["UPDATE v2_user_addresses SET is_active=0",
    "UPDATE v2_integrations SET credential_generation=2",
    "UPDATE v2_scheduled_tasks SET prompt='modified'",
    "UPDATE v2_owner_telegram_ingress SET chat_id='999'"])
def test_telegram_schedule_denies_revoked_or_tampered_origin(telegram_schedule, statement):
    queue, record, due = telegram_schedule
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.execute(statement)
        assert verified_schedule_origin_locked(conn, record.scheduled_task_id) is None
        assert verified_telegram_schedule_destination_locked(conn, record.scheduled_task_id) is None
        with pytest.raises(ValueError):
            stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)
