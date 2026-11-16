"""Accepted owner text survives a failed worker without inventing a reply."""
import tempfile
import time
from pathlib import Path

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import ModelProfile
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


class FailingProvider:
    supports_image_input = False

    def __init__(self):
        self.calls = 0

    def generate_json(self, request):
        self.calls += 1
        raise RuntimeError('synthetic provider failure')


def test_failed_worker_preserves_accepted_request_across_restart(tmp_path, monkeypatch):
    database = tmp_path / 'owner.sqlite'
    users = V2UserStore(database)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    provider = FailingProvider()

    def make_daemon(socket_path, daemon_id):
        queue = SQLiteMessageQueue(database, experimental_claim_fencing=True)
        conversations = SQLiteConversationStore(database)
        tools = InMemoryToolRegistry()
        tools.register(build_search_memory_tool_definition())
        tools.register(build_read_memory_source_tool_definition())
        runtime = build_runtime_host(user_store=users, messages=queue,
            question_store=SQLiteQuestionStore(database), conversation_store=conversations,
            outbox=SQLiteOutboundStore(database), tools=tools,
            inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
            coding_skill_service=object(), experimental_owner_read=True,
            owner_selected_draft_sink=lambda *_: None)
        daemon = V2Daemon(runtime, daemon_id=daemon_id, experimental_owner_only=True)
        monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
        monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
        daemon.ipc = V2DaemonServer(daemon, socket_path)
        return daemon, conversations

    with tempfile.TemporaryDirectory(prefix='owner-failure-', dir='/private/tmp') as socket_dir:
        first, conversation = make_daemon(Path(socket_dir) / 'first.sock', 'fixture-first')
        first.start()
        try:
            client = V2DaemonClient(first.ipc.socket_path)
            receipt = client.request('experimental_owner_ingress', prompt='A synthetic accepted request.')
            message_id = receipt['message_id']
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                status = client.request('experimental_owner_status', message_id=message_id)['status']
                if status != 'pending':
                    break
                time.sleep(.02)
            assert status == 'failed'
            assert provider.calls >= 1
            events = conversation.list(owner_user_id=owner.user_id)
            assert [(event.role, event.content, event.source_message_id) for event in events] == [
                ('user', 'A synthetic accepted request.', f'inbound:{message_id}')]
            assert events[0].created_at == receipt['accepted_at']
        finally:
            first.stop()

        calls_before_restart = provider.calls
        second, restarted_conversation = make_daemon(Path(socket_dir) / 'second.sock', 'fixture-second')
        second.start()
        try:
            # Reopening the real worker does not retry an uncertain prior claim
            # or append an answer. The canonical request remains readable.
            client = V2DaemonClient(second.ipc.socket_path)
            assert client.ping()['status'] == 'ready'
            assert client.request('experimental_owner_status', message_id=message_id)['status'] == 'unavailable'
            with second.runtime.queue._connect() as conn:
                assert conn.execute('SELECT status FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()[0] == 'processing'
            events = restarted_conversation.list(owner_user_id=owner.user_id)
            assert [(event.role, event.content, event.source_message_id) for event in events] == [
                ('user', 'A synthetic accepted request.', f'inbound:{message_id}')]
            assert provider.calls == calls_before_restart
        finally:
            second.stop()
