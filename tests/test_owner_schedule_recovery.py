import os
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_origin import (
    create_schedule_origin_locked, stage_due_occurrence_locked,
)
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_recovery import (
    list_schedule_attention, record_uncertain_schedule,
)
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedules import execute_owner_schedules
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.users import V2UserStore


def _scheduled(tmp_path):
    path = tmp_path / 'synthetic.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    schedules = ScheduledTaskStore(str(path))
    goal = 'In 5 minutes, check the pantry note.'
    source = queue.enqueue_trusted_local_owner(prompt=goal, peer_uid=os.getuid())
    task = SimpleNamespace(message_id=source, user=owner.user_id, goal=goal)
    now = datetime.now(timezone.utc)
    due = now + timedelta(minutes=5)
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        record = schedules.create_task(owner_user_id=owner.user_id, name='Check pantry', prompt=goal,
            schedule_kind='once', run_at=due.isoformat(), delivery_mode='pdca',
            idempotency_key='owner-schedule:' + source, now=now, connection=conn)
        create_schedule_origin_locked(conn, task, record)
        message_id = stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)
    queue._owner_schedule_clock = lambda: due
    claim = queue.claim_next(selector=MessageSelector(message_id=message_id), lease_owner='synthetic', lease_seconds=180)
    running = TaskState.from_message(claim.message, message_id=message_id)
    running.task_id = message_id
    return queue, owner.user_id, record.scheduled_task_id, message_id, claim, running


def test_failed_worker_marker_is_durable_uncertain_and_never_replays(tmp_path):
    queue, owner, schedule_id, message_id, claim, task = _scheduled(tmp_path)
    assert list_schedule_attention(queue, owner)[schedule_id] == 'processing'
    record_uncertain_schedule(queue, claim.claim_receipt, task)
    assert list_schedule_attention(queue, owner)[schedule_id] == 'uncertain'
    assert list_schedule_attention(queue, 'other') == {}
    assert queue.status_for(message_id) == 'processing'
    with queue._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_owner_schedule_occurrences').fetchone()[0] == 1
        assert conn.execute('SELECT state FROM v2_owner_schedule_occurrences').fetchone()[0] == 'queued'


def test_crashed_worker_expired_lease_is_uncertain_without_marker(tmp_path):
    queue, owner, schedule_id, message_id, _, _ = _scheduled(tmp_path)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_inbound_messages SET lease_expires_at=? WHERE message_id=?',
                     ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(), message_id))
    assert list_schedule_attention(queue, owner)[schedule_id] == 'uncertain'
    assert queue.status_for(message_id) == 'processing'
    assert queue.claim_next(selector=MessageSelector(message_id=message_id), lease_owner='new-worker') is None


def test_owner_schedule_listing_exposes_uncertain_without_replaying(tmp_path):
    queue, owner, schedule_id, message_id, claim, task = _scheduled(tmp_path)
    record_uncertain_schedule(queue, claim.claim_receipt, task)
    store = ScheduledTaskStore(queue.db_path)
    context = SimpleNamespace(task=SimpleNamespace(user=owner), messages=queue, schedule_store=store,
                              owner_schedule_store=store, cancellation_checker=lambda: False)
    schedules = execute_owner_schedules({}, context=context)['schedules']
    row = next(item for item in schedules if item['id'] == schedule_id)
    assert row['execution_state'] == 'uncertain'
    assert queue.status_for(message_id) == 'processing'
    context.task.user = 'someone-else'
    assert execute_owner_schedules({}, context=context)['schedules'] == []


def test_failed_owner_schedule_remains_visible(tmp_path):
    queue, owner, schedule_id, _, _, _ = _scheduled(tmp_path)
    with queue._connect() as conn:
        conn.execute("UPDATE v2_scheduled_tasks SET status='failed' WHERE scheduled_task_id=?", (schedule_id,))
        conn.execute("UPDATE v2_owner_schedule_occurrences SET state='failed' WHERE schedule_id=?", (schedule_id,))
    store = ScheduledTaskStore(queue.db_path)
    context = SimpleNamespace(task=SimpleNamespace(user=owner), messages=queue, schedule_store=store,
                              owner_schedule_store=store, cancellation_checker=lambda: False)
    row = next(item for item in execute_owner_schedules({}, context=context)['schedules'] if item['id'] == schedule_id)
    assert row['status'] == row['execution_state'] == 'failed'
