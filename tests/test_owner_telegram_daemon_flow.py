"""Synthetic Telegram polling through the actual owner worker and reply handoff."""
from datetime import datetime, timezone
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
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize('uncertain', [False, True])
@pytest.mark.parametrize('bash_enabled', [False, True])
def test_telegram_update_through_real_owner_worker_is_delivered_at_most_once(tmp_path, monkeypatch, uncertain, bash_enabled):
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    users.bind_address(user_id=owner.user_id, integration_id='telegram-test',
                       provider_key='telegram', provider_user_id='123')
    store = SQLiteIntegrationStore(path)
    store.upsert(integration_id='telegram-test', provider_key='telegram', display_name='Telegram',
                 enabled=True, config={'allowed_chat_ids': ['123']})
    monkeypatch.setattr(store, 'resolve_credentials', lambda _: {'bot_token': 'synthetic'})
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition())
    tools.register(build_read_memory_source_tool_definition())
    tools.register(build_bash_tool_definition())

    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            assert 'Synthetic incoming message' in request.prompt
            assert ('native.bash' in request.prompt) is bash_enabled
            if self.calls == 1 and bash_enabled:
                value = {'kind': 'tool', 'tool_id': 'native.bash', 'arguments': {'command': 'printf synthetic-check'}, 'text': None}
            else:
                if bash_enabled:
                    assert 'synthetic-check' in request.prompt
                value = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic reply'}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    class Client:
        def __init__(self): self.sent = []
        def get_updates(self, *, offset, timeout):
            # Repeated provider responses test deduplication and offset handling.
            def update(update_id, date, text):
                return {'update_id': update_id, 'message': {'date': date, 'message_id': update_id,
                    'text': text, 'from': {'id': 123}, 'chat': {'id': 123, 'type': 'private'}}}
            return [update(40, 999, 'Old message must be held'), update(41, 1001, 'Synthetic incoming message')]
        def send_message(self, *, chat_id, text):
            self.sent.append((chat_id, text))
            if uncertain: raise TimeoutError('synthetic timeout')
            return 'synthetic-provider-reply'

    from alphonse.agent_v2.core.intelligence.simple_loop import owner_telegram_listener as module
    real_listener = module.TelegramOwnerListener
    fake_client = Client()
    monkeypatch.setattr(module, 'TelegramOwnerListener', lambda q, s, i: real_listener(
        q, s, i, client_factory=lambda **_: fake_client,
        clock=lambda: datetime.fromtimestamp(1000, timezone.utc), poll_interval=.1))
    provider = Provider()
    runtime = build_runtime_host(user_store=users, messages=queue, integration_store=store,
        question_store=SQLiteQuestionStore(path), conversation_store=conversation,
        outbox=SQLiteOutboundStore(path), tools=tools,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    if bash_enabled:
        runtime.processor.owner_read_candidate.enable_owner_host_bash(trusted_local_development=True)
    daemon = V2Daemon(runtime, daemon_id='synthetic-telegram', experimental_owner_only=True,
                      owner_telegram_enabled=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        expected = 'started_unconfirmed' if uncertain else 'finalized'
        deadline = time.monotonic() + 8
        reply_status = None
        while time.monotonic() < deadline:
            with queue._connect() as conn:
                row = conn.execute('SELECT status FROM v2_owner_telegram_replies').fetchone()
                reply_status = row['status'] if row else None
            if reply_status == expected: break
            time.sleep(.03)
        assert reply_status == expected
        time.sleep(.3)
        assert fake_client.sent == [('123', 'Synthetic reply')]
        assert provider.calls == (2 if bash_enabled else 1)
        with queue._connect() as conn:
            assert conn.execute('SELECT count(*) FROM v2_owner_telegram_ingress').fetchone()[0] == 1
            assert conn.execute('SELECT count(*) FROM v2_owner_telegram_held_updates').fetchone()[0] == 1
            assert conn.execute('SELECT count(*) FROM v2_owner_telegram_replies').fetchone()[0] == 1
            assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_trusted_owner_ingress'").fetchone()
    finally:
        daemon.stop()
