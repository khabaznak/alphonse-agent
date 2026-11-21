"""Scheduled work has its own provenance, never a forged direct-owner ingress."""
import os
import json
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_origin import (
    create_schedule_origin_locked, stage_due_occurrence_locked, verified_occurrence_locked,
)
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def fixture(tmp_path):
    path = tmp_path / 'synthetic.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    schedules = ScheduledTaskStore(str(path))
    goal = 'In 5 minutes, check the pantry note and tell me the number of cans.'
    message_id = queue.enqueue_trusted_local_owner(prompt=goal, peer_uid=os.getuid())
    task = SimpleNamespace(message_id=message_id, user=owner.user_id, goal=goal)
    now = datetime.now(timezone.utc)
    due = now + timedelta(minutes=5)
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        record = schedules.create_task(owner_user_id=owner.user_id, name='Check pantry', prompt=goal,
            schedule_kind='once', run_at=due.isoformat(), delivery_mode='pdca',
            idempotency_key='owner-schedule:' + message_id, now=now, connection=conn)
        create_schedule_origin_locked(conn, task, record)
    return queue, record, due


def test_due_handoff_is_distinct_atomic_and_idempotent(fixture):
    queue, record, due = fixture
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        with pytest.raises(ValueError, match='not_due'):
            stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due - timedelta(microseconds=1))
        message_id = stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)
        assert stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due) == message_id
        assert verified_occurrence_locked(conn, message_id, now=due)
        assert conn.execute('SELECT 1 FROM v2_trusted_owner_ingress WHERE message_id=?', (message_id,)).fetchone() is None
        assert conn.execute('SELECT COUNT(*) FROM v2_owner_schedule_occurrences').fetchone()[0] == 1


@pytest.mark.parametrize('changed', ['prompt', 'schedule', 'source', 'queued', 'cancelled'])
def test_tampering_or_cancellation_invalidates_occurrence(fixture, changed):
    queue, record, due = fixture
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        message_id = stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)
        assert verified_occurrence_locked(conn, message_id, now=due)
        if changed == 'prompt':
            conn.execute("UPDATE v2_scheduled_tasks SET prompt='different' WHERE scheduled_task_id=?", (record.scheduled_task_id,))
        elif changed == 'schedule':
            conn.execute("UPDATE v2_scheduled_tasks SET schedule_json='{}' WHERE scheduled_task_id=?", (record.scheduled_task_id,))
        elif changed == 'source':
            conn.execute("UPDATE v2_trusted_owner_ingress SET payload_sha256='forged' WHERE message_id=(SELECT request_message_id FROM v2_owner_schedule_origins)")
        elif changed == 'queued':
            conn.execute("UPDATE v2_inbound_messages SET prompt='forged' WHERE message_id=?", (message_id,))
        else:
            conn.execute("UPDATE v2_scheduled_tasks SET status='cancelled' WHERE scheduled_task_id=?", (record.scheduled_task_id,))
        assert not verified_occurrence_locked(conn, message_id, now=due)


@pytest.mark.parametrize('changed', ['metadata', 'cancelled', 'claim', 'none'])
def test_scheduled_claim_rechecks_authority_and_exact_payload(fixture, changed):
    queue, record, due = fixture
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        message_id = stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)
    queue._owner_schedule_clock = lambda: due
    acquired = queue.claim_next(selector=MessageSelector(message_id=message_id), lease_owner='synthetic', lease_seconds=180)
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    assert queue.verify_trusted_scheduled_owner_claim(acquired.claim_receipt, task)
    assert not queue.verify_trusted_local_owner_claim(acquired.claim_receipt, task)
    with queue._connect() as conn:
        if changed == 'metadata':
            conn.execute("UPDATE v2_inbound_messages SET metadata_json='{}' WHERE message_id=?", (message_id,))
        elif changed == 'cancelled':
            conn.execute("UPDATE v2_scheduled_tasks SET status='cancelled' WHERE scheduled_task_id=?", (record.scheduled_task_id,))
        elif changed == 'claim':
            conn.execute("UPDATE v2_inbound_messages SET claim_generation=? WHERE message_id=?", ('a' * 32, message_id))
    assert queue.verify_trusted_scheduled_owner_claim(acquired.claim_receipt, task) is (changed == 'none')


def test_revocation_during_scheduled_claim_rolls_back_effect(fixture):
    from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable
    queue, record, due = fixture
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        message_id = stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)
    queue._owner_schedule_clock = lambda: due
    acquired = queue.claim_next(selector=MessageSelector(message_id=message_id), lease_owner='synthetic', lease_seconds=180)
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    with pytest.raises(ClaimFenceUnavailable):
        with queue.trusted_scheduled_owner_claim_transaction(acquired.claim_receipt, task) as conn:
            conn.execute('CREATE TABLE synthetic_effect (value TEXT)')
            conn.execute("INSERT INTO synthetic_effect VALUES ('must roll back')")
            conn.execute("UPDATE v2_scheduled_tasks SET status='cancelled' WHERE scheduled_task_id=?", (record.scheduled_task_id,))
    with queue._connect() as conn:
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name='synthetic_effect'").fetchone() is None


def test_owner_cancel_before_due_prevents_staging_and_checks_actor(fixture):
    from alphonse.agent_v2.daemon import V2Daemon
    queue, record, due = fixture
    daemon = V2Daemon.__new__(V2Daemon)
    daemon.experimental_owner_only = True
    daemon.runtime = SimpleNamespace(queue=queue)
    with pytest.raises(ValueError, match='cancel_unavailable'):
        daemon.cancel_owner_schedule(actor_user_id='other-user', scheduled_task_id=record.scheduled_task_id)
    result = daemon.cancel_owner_schedule(actor_user_id=record.owner_user_id, scheduled_task_id=record.scheduled_task_id)
    assert result == {'scheduled_task_id': record.scheduled_task_id, 'status': 'cancelled',
                      'execution_may_have_started': False}
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        with pytest.raises(ValueError, match='origin_unavailable'):
            stage_due_occurrence_locked(conn, record.scheduled_task_id, now=due)


def test_owner_view_includes_new_and_legacy_schedules_once(fixture, tmp_path):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedules import execute_owner_schedules
    queue, record, due = fixture
    owner_store = ScheduledTaskStore(queue.db_path)
    legacy = ScheduledTaskStore(str(tmp_path / 'legacy.sqlite'))
    older = legacy.create_task(owner_user_id=record.owner_user_id, name='Legacy', prompt='Legacy task',
                               schedule_kind='once', run_at=(due + timedelta(days=1)).isoformat())
    context = SimpleNamespace(task=SimpleNamespace(user=record.owner_user_id), schedule_store=legacy,
                              owner_schedule_store=owner_store, cancellation_checker=lambda: False)
    result = execute_owner_schedules({}, context=context)
    assert {item['id'] for item in result['schedules']} == {record.scheduled_task_id, older.scheduled_task_id}
    new = next(item for item in result['schedules'] if item['id'] == record.scheduled_task_id)
    assert new['execution'] == 'owner_worker'
    context.schedule_store = owner_store
    result = execute_owner_schedules({}, context=context)
    assert [item['id'] for item in result['schedules']] == [record.scheduled_task_id]


def test_owner_cancel_ipc_requires_peer_and_disallows_actor_payload(fixture, monkeypatch):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.ipc import V2DaemonServer, PROTOCOL_VERSION
    queue, record, _ = fixture
    daemon = V2Daemon.__new__(V2Daemon)
    daemon.experimental_owner_only = True
    daemon.runtime = SimpleNamespace(queue=queue, user_store=V2UserStore(queue.db_path))
    server = V2DaemonServer(daemon, experimental_owner_ingress=True)
    request = {'version': PROTOCOL_VERSION, 'params': {'scheduled_task_id': record.scheduled_task_id}}
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: (_ for _ in ()).throw(PermissionError('denied')))
    with pytest.raises(PermissionError):
        server._owner_cancel_schedule(None, request)
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: os.getuid())
    with pytest.raises(ValueError, match='payload_denied'):
        server._owner_cancel_schedule(None, {'version': PROTOCOL_VERSION,
            'params': {'scheduled_task_id': record.scheduled_task_id, 'actor_user_id': 'forged'}})
    assert server._owner_cancel_schedule(None, request)['status'] == 'cancelled'


def test_daemon_consumes_due_occurrence_and_records_result(fixture, tmp_path, monkeypatch):
    from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
    from alphonse.agent_v2.core.inference.router import InferenceRouter
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host

    queue, record, due = fixture
    path = queue.db_path
    users = V2UserStore(path)
    conversations = SQLiteConversationStore(path)
    with queue._connect() as conn:
        # Simulate the already acknowledged original owner request. Provenance
        # persists after its queue row is removed.
        conn.execute('DELETE FROM v2_inbound_messages WHERE message_id=(SELECT request_message_id FROM v2_owner_schedule_origins)')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            value = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic result.'}
            return InferenceResult(json_value=value, raw_response=json.dumps(value), usage={'input_tokens': 1, 'output_tokens': 1})
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversations, outbox=SQLiteOutboundStore(path), tools=registry,
        inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    queue._owner_schedule_clock = lambda: due
    daemon = V2Daemon(runtime, daemon_id='scheduled-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            with queue._connect() as conn:
                occurrence = conn.execute('SELECT * FROM v2_owner_schedule_occurrences WHERE schedule_id=?',
                                          (record.scheduled_task_id,)).fetchone()
                if occurrence is not None and occurrence['state'] == 'completed':
                    break
            time.sleep(.02)
        assert occurrence is not None and occurrence['state'] == 'completed'
        events = conversations.list(owner_user_id=record.owner_user_id)
        assert any(event.content == 'Synthetic result.' and event.source == 'scheduled' for event in events)
        assert daemon.scheduler._thread is None
    finally:
        daemon.stop()
