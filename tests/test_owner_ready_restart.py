"""Owner may recover an unpicked answer under the original live claim.

The test uses real file locking and sockets, but only disposable stores and a
scripted provider. It does not claim expired-claim takeover or effect recovery.
"""
import json
import multiprocessing
import os
import shutil
import tempfile
import time
from pathlib import Path

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import (
    build_read_memory_source_tool_definition, build_search_memory_tool_definition,
)
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def _crash_fixture_daemon(root: str, socket_dir: str, signal):
    """A standalone daemon that can be killed after durable handoff."""
    from alphonse.agent_v2 import daemon as module
    module.default_socket_path = lambda: Path(socket_dir) / 'daemon.sock'
    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            signal.put('model_call')
            value = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    db = Path(root) / 'fixture.sqlite'
    users = V2UserStore(db)
    users.onboard(display_name='Synthetic owner', users_root=Path(root) / 'users')
    queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    router = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline'))
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
        conversation_store=SQLiteConversationStore(db), outbox=SQLiteOutboundStore(db), tools=registry,
        inference=router, coding_skill_service=object(), experimental_owner_read=True,
        owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='crashed', experimental_owner_only=True)
    daemon.ipc = V2DaemonServer(daemon, str(Path(socket_dir) / 'one.sock'))
    daemon.start()
    signal.put('started')
    while True:
        time.sleep(1)


def test_ready_answer_survives_real_daemon_process_death(tmp_path, monkeypatch):
    from alphonse.agent_v2 import daemon as module
    socket_dir = Path(tempfile.mkdtemp(prefix='owner-crash-', dir='/private/tmp'))
    monkeypatch.setattr(module, 'default_socket_path', lambda: socket_dir / 'daemon.sock')
    ctx = multiprocessing.get_context('spawn')
    signal = ctx.Queue()
    process = ctx.Process(target=_crash_fixture_daemon, args=(str(tmp_path), str(socket_dir), signal))
    process.start()
    replacement = None
    try:
        assert signal.get(timeout=20) == 'started'
        client = V2DaemonClient(str(socket_dir / 'one.sock'))
        message_id = client.request('experimental_owner_ingress', prompt='Say hello.')['message_id']
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = client.request('experimental_owner_status', message_id=message_id)['status']
            if status != 'pending':
                break
            time.sleep(.02)
        assert status == 'ready'
        assert signal.get(timeout=2) == 'model_call'
        process.kill()
        process.join(timeout=10)
        assert not process.is_alive()

        db = tmp_path / 'fixture.sqlite'
        users = V2UserStore(db)
        queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
        conversation = SQLiteConversationStore(db)
        owner = users.admin_user()
        # The accepted request survives process death before Desktop picks up
        # the answer; a prepared draft is not yet a delivered response.
        assert [(row.role, row.content) for row in conversation.list(owner_user_id=owner.user_id)] == [
            ('user', 'Say hello.')]
        registry = InMemoryToolRegistry()
        registry.register(build_search_memory_tool_definition())
        registry.register(build_read_memory_source_tool_definition())
        class NeverProvider:
            supports_image_input = False
            def generate_json(self, request):
                pytest.fail('a recovered ready answer must not repeat inference')
        runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
            conversation_store=conversation, outbox=SQLiteOutboundStore(db), tools=registry,
            inference=InferenceRouter(provider=NeverProvider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
            coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        replacement = V2Daemon(runtime, daemon_id='replacement', experimental_owner_only=True)
        replacement.ipc = V2DaemonServer(replacement, str(socket_dir / 'two.sock'))
        replacement.start()
        resumed = V2DaemonClient(replacement.ipc.socket_path)
        assert resumed.request('experimental_owner_status', message_id=message_id)['status'] == 'ready'
        draft = resumed.request('experimental_owner_take_draft', message_id=message_id)
        assert draft['message'] == 'Synthetic answer.'
        assert resumed.request('experimental_owner_ack_draft', message_id=message_id,
            acknowledgement_token=draft['acknowledgement_token'])['status'] == 'client_acknowledged'
        owner = users.admin_user()
        assert [(row.role, row.content) for row in conversation.list(owner_user_id=owner.user_id)] == [
            ('user', 'Say hello.'), ('assistant', 'Synthetic answer.')]
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        if replacement is not None:
            replacement.stop()
        signal.close()
        shutil.rmtree(socket_dir)


@pytest.mark.parametrize('mode', ['ready', 'no_handoff', 'picked_up', 'expired', 'cancelled', 'source_changed', 'replaced', 'owner_revoked', 'unknown_effect', 'outbox_effect', 'mismatched_goal', 'mismatched_metadata'])
def test_exclusive_restart_restores_only_confirmed_unpicked_draft(tmp_path, monkeypatch, mode):
    from alphonse.agent_v2 import daemon as module
    socket_dir = Path(tempfile.mkdtemp(prefix='owner-restart-', dir='/private/tmp'))
    monkeypatch.setattr(module, 'default_socket_path', lambda: socket_dir / 'daemon.sock')
    holder = {}
    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            if mode == 'source_changed' and self.calls <= 2:
                value = ({'kind': 'tool', 'tool_id': 'native.search_memory',
                          'arguments': {'query': 'apples', 'source_types': ['conversation']}, 'text': None}
                         if self.calls == 1 else {'kind': 'tool', 'tool_id': 'native.read_memory_source',
                          'arguments': {'source_id': holder['source_id'], 'neighbors': 0}, 'text': None})
                return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                       usage={'input_tokens': 100, 'output_tokens': 20})
            if mode == 'cancelled':
                holder['daemon'].trigger_killswitch(actor_user_id=holder['owner'])
            value = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    db = tmp_path / 'fixture.sqlite'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    holder['owner'] = owner.user_id
    queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(db)
    if mode == 'source_changed':
        source = conversation.record(owner_user_id=owner.user_id, role='user', content='Synthetic apples.',
                                     source='test', source_message_id='source-fixture')
        holder['source_id'] = 'conversation:' + source.event_id
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline'))
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
        conversation_store=conversation, outbox=SQLiteOutboundStore(db), tools=registry, inference=router,
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    first = V2Daemon(runtime, daemon_id='old', experimental_owner_only=True)
    holder['daemon'] = first
    first.ipc = V2DaemonServer(first, str(socket_dir / 'one.sock'))
    first.start()
    second = None
    try:
        client = V2DaemonClient(first.ipc.socket_path)
        message_id = client.request('experimental_owner_ingress', prompt='Say hello.')['message_id']
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = client.request('experimental_owner_status', message_id=message_id)['status']
            if status != 'pending':
                break
            time.sleep(.02)
        assert status == ('failed' if mode == 'cancelled' else 'ready')
        expected_calls = 3 if mode == 'source_changed' else 1
        assert provider.calls == expected_calls

        # The real flock must reject another owner daemon while this one lives.
        second = V2Daemon(runtime, daemon_id='new', experimental_owner_only=True)
        second.ipc = V2DaemonServer(second, str(socket_dir / 'two.sock'))
        with pytest.raises(RuntimeError, match='daemon_lock_held|daemon_already_running'):
            second.start()

        if mode == 'picked_up':
            client.request('experimental_owner_take_draft', message_id=message_id)
        if mode == 'source_changed':
            assert conversation.delete(event_id=source.event_id, owner_user_id=owner.user_id)
        if mode == 'owner_revoked':
            users.update_user(owner.user_id, is_active=False)
        if mode in {'no_handoff', 'expired', 'replaced', 'unknown_effect', 'outbox_effect', 'mismatched_goal', 'mismatched_metadata'}:
            with queue._connect() as conn:
                if mode == 'expired':
                    conn.execute("UPDATE v2_inbound_messages SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE message_id=?", (message_id,))
                elif mode == 'replaced':
                    # Simulate an unsafe old writer replacing the generation.
                    conn.execute("UPDATE v2_inbound_messages SET claim_generation=? WHERE message_id=?", ('a' * 32, message_id))
                elif mode == 'outbox_effect':
                    conn.execute("""INSERT INTO v2_outbox(outbox_message_id,integration_id,provider_key,
                        channel_target,message,kind,task_id,created_at,updated_at)
                        VALUES ('effect','desktop','desktop','fixture','fixture','response',?,'now','now')""", (message_id,))
                else:
                    row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (message_id,)).fetchone()
                    saved = json.loads(row['task_state_json'])
                    if mode == 'unknown_effect':
                        saved['metadata']['unknown_external_effect'] = {'status': 'uncertain'}
                    elif mode == 'mismatched_goal':
                        saved['goal'] = 'Different request.'
                    elif mode == 'mismatched_metadata':
                        saved['metadata']['intelligence_schema_version'] = True
                    else:
                        assert saved['metadata'].pop('owner_read_ready_handoff')
                    conn.execute('UPDATE v2_task_checkpoints SET task_state_json=?,version=version+1 WHERE task_id=?',
                                 (json.dumps(saved), message_id))
        first.stop()
        second.start()
        replacement = V2DaemonClient(second.ipc.socket_path)
        status = replacement.request('experimental_owner_status', message_id=message_id)['status']
        if mode == 'source_changed':
            assert status == 'ready'
            with pytest.raises(RuntimeError, match='owner_recovery_unavailable'):
                replacement.request('experimental_owner_take_draft', message_id=message_id)
            assert provider.calls == expected_calls
            return
        if mode != 'ready':
            assert status == 'unavailable'
            with pytest.raises(RuntimeError, match='owner_desktop_pickup_unavailable'):
                replacement.request('experimental_owner_take_draft', message_id=message_id)
            assert provider.calls == expected_calls
            return
        assert status == 'ready'
        draft = replacement.request('experimental_owner_take_draft', message_id=message_id)
        assert draft['message'] == 'Synthetic answer.'
        assert replacement.request('experimental_owner_ack_draft', message_id=message_id,
            acknowledgement_token=draft['acknowledgement_token'])['status'] == 'client_acknowledged'
        assert [(event.role, event.content) for event in conversation.list(owner_user_id=owner.user_id)] == [
            ('user', 'Say hello.'), ('assistant', 'Synthetic answer.')]
        assert provider.calls == 1
    finally:
        first.stop()
        if second is not None:
            second.stop()
        shutil.rmtree(socket_dir)
