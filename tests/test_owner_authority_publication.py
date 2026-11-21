"""Cancellation checks against a socket-backed synthetic authority service."""
import json
import os
import socket
import threading
import time

import pytest

from alphonse.agent_v2.authority_service.transport import AuthorityClient, AuthoritySocketService, EndpointIdentity
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop.owner_authority_bridge import OwnerAuthorityBridge
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def client(service, role):
    def connect():
        local, remote = socket.socketpair()
        local.settimeout(3)
        threading.Thread(target=service.serve_connection, args=(remote,),
                         kwargs={'endpoint_role': role}, daemon=True).start()
        return local
    return AuthorityClient(connect)


@pytest.mark.parametrize('cancel_at', ['after_return', 'after_ready', 'during_pickup'])
def test_service_revocation_blocks_answer_or_pickup(tmp_path, monkeypatch, cancel_at):
    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            if self.calls == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.bash',
                            'arguments': {'command': 'printf synthetic > prospect.txt'}, 'text': None}
            else:
                if cancel_at == 'after_return':
                    desktop.cancel(request_id=authority_request)
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                            'text': 'The synthetic note was written.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    owner = users.onboard(display_name='Synthetic owner')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    registry = InMemoryToolRegistry()
    for definition in (build_search_memory_tool_definition(), build_read_memory_source_tool_definition(),
                       build_bash_tool_definition()):
        registry.register(definition)
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=SQLiteConversationStore(path), outbox=SQLiteOutboundStore(path), tools=registry,
        inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    service = AuthoritySocketService(tmp_path / 'authority.sqlite', endpoints={
        'desktop': EndpointIdentity('desktop', owner.user_id, 'synthetic-desktop'),
        'worker': EndpointIdentity('worker', owner.user_id, 'synthetic-worker'),
    }, verify_peer=lambda *_: True)
    desktop, worker = client(service, 'desktop'), client(service, 'worker')
    desktop.grant_bash(expected_generation=desktop.bash_capability()['generation'])
    goal, message_id = 'Create a synthetic note.', 'a' * 32
    authority_request = desktop.submit(goal=goal, idempotency_key=message_id)['request_id']
    candidate = runtime.processor.owner_read_candidate
    candidate.enable_owner_host_bash()
    candidate.attach_experimental_owner_authority(OwnerAuthorityBridge(worker, lambda _: authority_request))
    daemon = V2Daemon(runtime, daemon_id='authority-publication-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        assert queue.enqueue_trusted_local_owner(prompt=goal, peer_uid=os.getuid(),
            _authority_binding=(message_id, authority_request)) == message_id
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = daemon.owner_message_status(message_id)
            if status not in {'pending', 'processing'}:
                break
            time.sleep(.01)
        assert (users.ordinary_workspace(owner.user_id) / 'prospect.txt').read_text() == 'synthetic'
        if cancel_at == 'after_return':
            assert status == 'failed'
        elif cancel_at == 'after_ready':
            assert status == 'ready'
            assert desktop.cancel(request_id=authority_request)['execution_may_have_started'] is True
            with pytest.raises(ValueError):
                daemon.take_owner_desktop_draft(message_id)
        else:
            assert status == 'ready'
            checkpoint, context = daemon._owner_checkpoints[message_id]
            original_check = context._owner_authority_check
            calls = []
            def revoke_after_pickup_commit():
                calls.append(1)
                if len(calls) == 3:
                    desktop.cancel(request_id=authority_request)
                return original_check()
            context._owner_authority_check = revoke_after_pickup_commit
            with pytest.raises(ValueError):
                daemon.take_owner_desktop_draft(message_id)
            assert len(calls) == 3
            # The local pickup marker has committed. It must remain uncertain,
            # with no automatic repeat after the service revokes the claim.
            with checkpoint.queue._connect() as conn:
                row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                   (message_id,)).fetchone()
            marker = json.loads(row['task_state_json'])['metadata']['owner_read_desktop_pickup_started']
            assert marker['status'] == 'started_unconfirmed'
            with pytest.raises(ValueError):
                daemon.take_owner_desktop_draft(message_id)
    finally:
        daemon.stop()
