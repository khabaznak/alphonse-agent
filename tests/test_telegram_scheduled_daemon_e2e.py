"""A Telegram-created scheduled task's due result reaches the same chat once.

The original request and schedule are real local fixture records; model and
Telegram provider are fake. There is no external network or household data.
"""
from datetime import datetime, timedelta, timezone
import json
import time

import pytest

from test_owner_telegram_handoff import ready
from test_owner_telegram_schedule_origin import telegram_schedule
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_origin import _digest, stage_due_occurrence_locked
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize('ambiguous', [False, True])
def test_real_scheduled_worker_sends_result_once_to_origin_chat(telegram_schedule, tmp_path, monkeypatch, ambiguous):
    queue, record, _ = telegram_schedule
    users = V2UserStore(queue.db_path)
    users.set_workspace_root(tmp_path / 'workspace')
    # Move only the disposable schedule's due time into the past and update
    # the fixture's immutable origin binding, then stage a real occurrence.
    due = datetime.now(timezone.utc) - timedelta(seconds=1)
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('SELECT schedule_json FROM v2_scheduled_tasks WHERE scheduled_task_id=?',
                           (record.scheduled_task_id,)).fetchone()
        schedule = json.loads(row['schedule_json']); schedule['run_at'] = due.isoformat()
        conn.execute('UPDATE v2_scheduled_tasks SET next_run_at=?,schedule_json=? WHERE scheduled_task_id=?',
                     (due.isoformat(), json.dumps(schedule, sort_keys=True), record.scheduled_task_id))
        current = conn.execute('SELECT * FROM v2_scheduled_tasks WHERE scheduled_task_id=?',
                               (record.scheduled_task_id,)).fetchone()
        conn.execute('UPDATE v2_owner_schedule_origins SET schedule_sha256=? WHERE scheduled_task_id=?',
                     (_digest(current), record.scheduled_task_id))
        message_id = stage_due_occurrence_locked(conn, record.scheduled_task_id, now=datetime.now(timezone.utc))

    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic due result'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    class Listener:
        def __init__(self): self.sent = []
        def send_reply(self, **values):
            self.sent.append(values)
            if ambiguous: raise TimeoutError('synthetic timeout')
            return 'synthetic-provider-reply'
        def stop(self, *, timeout=12): return True

    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition()); tools.register(build_read_memory_source_tool_definition())
    provider, listener = Provider(), Listener()
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(queue.db_path),
        schedule_store=ScheduledTaskStore(queue.db_path), conversation_store=SQLiteConversationStore(queue.db_path),
        outbox=SQLiteOutboundStore(queue.db_path), tools=tools,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='synthetic-scheduled-telegram', experimental_owner_only=True)
    daemon._owner_telegram_listener = listener
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None); monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        deadline = time.monotonic() + 8
        result_state = occurrence_state = None
        while time.monotonic() < deadline:
            with queue._connect() as conn:
                row = conn.execute('SELECT status FROM v2_owner_telegram_replies WHERE message_id=?', (message_id,)).fetchone()
                result_state = row['status'] if row else None
                occurrence = conn.execute('SELECT state FROM v2_owner_schedule_occurrences WHERE message_id=?', (message_id,)).fetchone()
                occurrence_state = occurrence['state'] if occurrence else None
            if result_state in {'provider_accepted', 'started_unconfirmed'} and occurrence_state == 'completed':
                break
            time.sleep(.03)
        assert result_state == ('started_unconfirmed' if ambiguous else 'provider_accepted')
        assert occurrence_state == 'completed'
        time.sleep(.3)
        assert listener.sent == [{'integration_id': 'telegram-test', 'chat_id': '123', 'text': 'Synthetic due result'}]
        assert provider.calls == 1
        with queue._connect() as conn:
            assert conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone() is None
            event = conn.execute('SELECT content FROM v2_conversation_events WHERE source_message_id=?',
                                 ('outbound:scheduled:' + message_id,)).fetchone()
            assert event['content'] == 'Synthetic due result'
        if ambiguous:
            # Restarting storage must retain the unknown send; completion of
            # the work must not imply that Telegram received the reply.
            from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
            restarted = SQLiteMessageQueue(queue.db_path, experimental_claim_fencing=True)
            with restarted._connect() as conn:
                assert conn.execute('SELECT status FROM v2_owner_telegram_replies WHERE message_id=?',
                                    (message_id,)).fetchone()['status'] == 'started_unconfirmed'
            items = daemon.owner_recovery_list()
            current = [item for item in items if item['message_id'] == message_id]
            assert len(current) == 1
            assert current[0]['status'] == 'telegram_delivery_unconfirmed'
            assert current[0]['can_abandon'] is False
            assert current[0]['prompt'] == 'Telegram response'
            with queue._connect() as conn:
                conn.execute("UPDATE v2_owner_telegram_replies SET owner_id='someone-else' WHERE message_id=?", (message_id,))
            assert not any(item['message_id'] == message_id for item in daemon.owner_recovery_list())
    finally:
        daemon.stop()
