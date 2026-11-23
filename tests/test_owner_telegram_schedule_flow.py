"""Synthetic Telegram schedule and cancellation through the real owner worker."""
from datetime import datetime, timedelta, timezone
import json
import time
import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.core.tools.registry.native.scheduled_task import build_scheduled_task_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize('initial', [
    'In five minutes, count the beans in my pantry note.',
    'Remind me to water the plants in five minutes.',
    'Recuérdame regar las plantas en cinco minutos.',
])
@pytest.mark.parametrize('mode', ['cancel', 'due'])
def test_telegram_schedule_lifecycle_in_owner_worker(tmp_path, monkeypatch, initial, mode):
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    users.bind_address(user_id=owner.user_id, integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
    store = SQLiteIntegrationStore(path)
    store.upsert(integration_id='telegram-test', provider_key='telegram', display_name='Telegram',
                 enabled=True, config={'allowed_chat_ids': ['123']})
    monkeypatch.setattr(store, 'resolve_credentials', lambda _: {'bot_token': 'synthetic'})
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    tools = InMemoryToolRegistry()
    for definition in (build_search_memory_tool_definition(), build_read_memory_source_tool_definition(), build_scheduled_task_tool_definition()):
        tools.register(definition)
    cancel = 'Cancel the pantry task.'

    class Provider:
        supports_image_input = False
        calls = 0
        schedule_id = None
        def generate_json(self, request):
            self.calls += 1
            if self.calls == 1:
                action = {'kind': 'tool', 'tool_id': 'native.scheduled_task', 'text': None,
                          'arguments': {'request_quote': initial, 'run_at': (datetime.now(timezone.utc) + timedelta(minutes=5)).isoformat()}}
            elif self.calls == 2:
                action = ({'kind': 'tool', 'tool_id': 'native.owner_schedules', 'text': None, 'arguments': {}}
                          if mode == 'cancel' else
                          {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Reminder: water the plants.'})
            elif self.calls == 3:
                action = {'kind': 'tool', 'tool_id': 'native.scheduled_task', 'text': None,
                          'arguments': {'operation': 'cancel', 'request_quote': cancel, 'scheduled_task_id': self.schedule_id}}
            else:
                action = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'The task was cancelled.'}
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    class Client:
        def __init__(self): self.updates, self.sent = [], []
        def get_updates(self, *, offset, timeout):
            return [event for event in self.updates if offset is None or event['update_id'] >= offset]
        def send_message(self, *, chat_id, text):
            self.sent.append((chat_id, text))
            return str(200 + len(self.sent))
        def add(self, update_id, text):
            self.updates.append({'update_id': update_id, 'message': {'date': 1001, 'message_id': update_id,
                'text': text, 'from': {'id': 123}, 'chat': {'id': 123, 'type': 'private'}}})

    from alphonse.agent_v2.core.intelligence.simple_loop import owner_telegram_listener as listener_module
    real_listener = listener_module.TelegramOwnerListener
    client = Client()
    monkeypatch.setattr(listener_module, 'TelegramOwnerListener', lambda q, s, i: real_listener(
        q, s, i, client_factory=lambda **_: client, clock=lambda: datetime.fromtimestamp(1000, timezone.utc), poll_interval=.1))
    provider = Provider()
    runtime = build_runtime_host(user_store=users, messages=queue, integration_store=store,
        question_store=SQLiteQuestionStore(path), conversation_store=conversation, outbox=SQLiteOutboundStore(path),
        tools=tools, inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='telegram-schedule-fixture', experimental_owner_only=True, owner_telegram_enabled=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        client.add(1, initial)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and len(client.sent) < 1:
            time.sleep(.03)
        assert len(client.sent) == 1
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            with queue._connect() as conn:
                source = conn.execute('SELECT status FROM v2_owner_telegram_replies').fetchone()
            if source and source['status'] == 'finalized': break
            time.sleep(.03)
        assert source and source['status'] == 'finalized'
        records = runtime.owner_schedule_store.list_tasks(owner_user_id=owner.user_id)
        assert len(records) == 1 and records[0].status == 'active'
        provider.schedule_id = records[0].scheduled_task_id
        if mode == 'cancel':
            client.add(2, cancel)
        else:
            queue._owner_schedule_clock = lambda: datetime.now(timezone.utc) + timedelta(minutes=6)
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and len(client.sent) < 2:
            time.sleep(.03)
        assert len(client.sent) == 2
        if mode == 'cancel':
            assert runtime.owner_schedule_store.get_task(provider.schedule_id).status == 'cancelled'
            assert 'cancelled' in client.sent[1][1].lower()
        else:
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and runtime.owner_schedule_store.get_task(provider.schedule_id).status == 'active':
                time.sleep(.03)
            assert runtime.owner_schedule_store.get_task(provider.schedule_id).status == 'completed'
            assert 'reminder' in client.sent[1][1].lower()
    finally:
        daemon.stop()
