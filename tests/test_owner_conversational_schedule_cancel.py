"""Owner-worker cancellation requires an observed, still-current schedule."""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_origin import create_schedule_origin_locked
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
from alphonse.agent_v2.core.tools.registry.native.scheduled_task import build_scheduled_task_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize('mode', ['normal', 'reminder_wording', 'recover', 'unobserved', 'stale', 'running', 'inference_failure', 'running_failure', 'repeat'])
def test_cancellation_owner_worker_observes_and_commits_once(tmp_path, monkeypatch, mode):
    path = tmp_path / 'fixture.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    users.set_timezone('UTC', owner.user_id)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    tools = InMemoryToolRegistry()
    for definition in (build_search_memory_tool_definition(), build_read_memory_source_tool_definition(),
                       build_scheduled_task_tool_definition()):
        tools.register(definition)
    goal = ('Cancel the task you set to remind me about the pantry.'
            if mode == 'reminder_wording' else 'Cancel the pantry check.')
    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            answer_at = 4 if mode == 'recover' else 3
            if self.calls == answer_at and mode in {'normal', 'reminder_wording', 'recover', 'running', 'inference_failure', 'running_failure'}:
                if mode in {'inference_failure', 'running_failure'}:
                    raise RuntimeError('synthetic_inference_failure')
                # The scripted model omits the running-work caveat. The
                # independently verified effect must still reach the client.
                answer = 'Listo, cancelé la tarea.'
                value = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': answer}
                return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                       usage={'input_tokens': 100, 'output_tokens': 20})
            if (self.calls == 1 and mode not in {'unobserved', 'recover'}) or (mode == 'recover' and self.calls == 2):
                value = {'kind': 'tool', 'tool_id': 'native.owner_schedules', 'arguments': {}, 'text': None}
            else:
                if mode == 'stale':
                    with queue._connect() as conn:
                        conn.execute("UPDATE v2_scheduled_tasks SET name='changed elsewhere' WHERE scheduled_task_id=?", (record.scheduled_task_id,))
                value = {'kind': 'tool', 'tool_id': 'native.scheduled_task', 'text': None,
                    'arguments': {'operation': 'cancel', 'scheduled_task_id': record.scheduled_task_id,
                                  'request_quote': goal}}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversation, outbox=SQLiteOutboundStore(path), tools=tools,
        inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    original = 'In 5 minutes, check the pantry.'
    original_id = queue.enqueue_trusted_local_owner(prompt=original, peer_uid=os.getuid())
    original_task = SimpleNamespace(message_id=original_id, user=owner.user_id, goal=original)
    now = datetime.now(timezone.utc)
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        record = runtime.owner_schedule_store.create_task(owner_user_id=owner.user_id, name='Pantry', prompt=original,
            schedule_kind='once', run_at=(now + timedelta(minutes=5)).isoformat(), delivery_mode='pdca',
            idempotency_key='owner-schedule:' + original_id, now=now, connection=conn)
        create_schedule_origin_locked(conn, original_task, record)
        conn.execute('DELETE FROM v2_inbound_messages WHERE message_id=?', (original_id,))
        if mode in {'running', 'running_failure'}:
            conn.execute('''INSERT INTO v2_owner_schedule_occurrences
                (schedule_id,owner_id,message_id,due_at,payload_sha256,state) VALUES (?,?,?,?,?,'queued')''',
                (record.scheduled_task_id, owner.user_id, 'a' * 32, record.next_run_at, 'b' * 64))
    daemon = V2Daemon(runtime, daemon_id='fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        message_id = queue.enqueue_trusted_local_owner(prompt=goal, peer_uid=os.getuid())
        deadline = time.monotonic() + 6
        while time.monotonic() < deadline and daemon.owner_message_status(message_id) in {'pending', 'processing'}:
            time.sleep(.01)
        stored = runtime.owner_schedule_store.get_task(record.scheduled_task_id)
        if mode in {'unobserved', 'stale'}:
            assert daemon.owner_message_status(message_id) != 'ready'
            assert stored.status == 'active'
        elif mode in {'inference_failure', 'running_failure', 'repeat'}:
            assert daemon.owner_message_status(message_id) != 'ready'
            assert stored.status == 'cancelled'
            items = daemon.owner_recovery_list()
            item = next(item for item in items if item['message_id'] == message_id)
            assert item['status'] == 'effect_committed' and item['can_abandon'] is False
            assert item['committed_effect'] == {'kind': 'scheduled_task_cancelled',
                'schedule_id': record.scheduled_task_id, 'name': 'Pantry', 'execution_may_have_started': mode == 'running_failure'}
            restarted = V2Daemon(runtime, daemon_id='restarted', experimental_owner_only=True)
            recovered = next(item for item in restarted.owner_recovery_list() if item['message_id'] == message_id)
            assert recovered['status'] == 'effect_committed'
            assert recovered['committed_effect'] == item['committed_effect']
            with queue._connect() as conn:
                assert conn.execute('SELECT COUNT(*) FROM v2_owner_schedule_cancellations WHERE request_id=?', (message_id,)).fetchone()[0] == 1
            with pytest.raises(ValueError, match='effect_committed'):
                queue.abandon_trusted_owner_message(message_id, clock=lambda: now + timedelta(hours=1))
        else:
            assert daemon.owner_message_status(message_id) == 'ready'
            assert stored.status == 'cancelled'
            draft = daemon.take_owner_desktop_draft(message_id)
            assert 'Listo, cancelé la tarea.' in draft['message']
            assert draft['committed_effect'] == {'kind': 'scheduled_task_cancelled',
                'schedule_id': record.scheduled_task_id, 'name': 'Pantry', 'execution_may_have_started': mode == 'running'}
            daemon.acknowledge_owner_desktop_draft(message_id, draft['acknowledgement_token'])
            history = daemon.desktop_conversation_history(user=owner.user_id)
            outbound = next(item for item in history if item['id'] == 'owner:' + message_id)
            assert outbound['committed_effect'] == draft['committed_effect']
    finally:
        daemon.stop()
