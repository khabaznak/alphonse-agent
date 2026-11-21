"""Owner host-Bash worker wired to a synthetic, socket-backed ledger.

The fake peer verifier and in-process endpoint are test fixtures, not a
protected OS service, Desktop identity, or human-authority boundary.
"""
import json
import os
import socket
import sqlite3
import threading
import time
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.authority_service.transport import AuthorityClient, AuthoritySocketService, EndpointIdentity
from alphonse.agent_v2.authority_service.experimental_ingress import queue_request_binding
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop.owner_authority_bridge import OwnerAuthorityBridge, OwnerAuthorityUnavailable
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def _client(service, role):
    def connect():
        client, server = socket.socketpair()
        client.settimeout(3)
        threading.Thread(target=service.serve_connection, args=(server,),
                         kwargs={'endpoint_role': role}, daemon=True).start()
        return client
    return AuthorityClient(connect)


def test_started_reservation_cannot_restart_or_be_reclaimed_after_reconstruction(tmp_path):
    service = AuthoritySocketService(tmp_path / 'ledger.sqlite', endpoints={
        'desktop': EndpointIdentity('desktop', 'synthetic-owner', 'desktop'),
        'worker': EndpointIdentity('worker', 'synthetic-owner', 'worker'),
    }, verify_peer=lambda *_: True)
    desktop, worker = _client(service, 'desktop'), _client(service, 'worker')
    desktop.grant_bash(expected_generation=desktop.bash_capability()['generation'])
    task = SimpleNamespace(user='synthetic-owner', task_id='b' * 32, goal='Synthetic request')
    request_id = desktop.submit(goal=task.goal, idempotency_key='b' * 32)['request_id']
    bridge = OwnerAuthorityBridge(worker, lambda _: request_id)
    bridge.begin(task)
    reservation = 'tool-' + 'a' * 32
    bridge.start(task, 'native.bash', {'command': 'printf synthetic'}, reservation_id=reservation)
    with pytest.raises(OwnerAuthorityUnavailable):
        bridge.start(task, 'native.bash', {'command': 'printf synthetic'}, reservation_id=reservation)
    with pytest.raises(OwnerAuthorityUnavailable):
        OwnerAuthorityBridge(worker, lambda _: request_id).begin(task)


@pytest.mark.parametrize('mode', ['allowed', 'provisioned_bridge', 'cancel_before_start', 'cancel_running', 'revoke_before_start',
                                  'revoke_regrant_running', 'missing_binding',
                                  'actor_mismatch', 'lost_start_reply', 'no_bridge',
                                  'grant_disabled', 'capability_query_unavailable'])
def test_real_owner_worker_uses_preexisting_service_request_and_fences_dispatch(tmp_path, monkeypatch, mode):
    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            if mode in {'grant_disabled', 'capability_query_unavailable'}:
                assert 'native.bash' not in request.prompt
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                            'text': 'Hello, how can I help?'}
            elif self.calls == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.bash',
                            'arguments': {'command': ("printf started > started.txt; sleep 5; printf done > prospect.txt"
                                                      if mode in {'cancel_running', 'revoke_regrant_running'}
                                                      else "printf synthetic > prospect.txt")}, 'text': None}
            elif self.calls == 2:
                decision = {'kind': 'tool', 'tool_id': 'native.bash',
                            'arguments': {'command': 'cat prospect.txt'}, 'text': None}
            else:
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                            'text': 'The note contains synthetic.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    owner = users.onboard(display_name='Synthetic owner')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    registry = InMemoryToolRegistry()
    for definition in (build_search_memory_tool_definition(), build_read_memory_source_tool_definition(),
                       build_bash_tool_definition()):
        registry.register(definition)
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversation, outbox=SQLiteOutboundStore(path), tools=registry,
        inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)

    authority_path = tmp_path / 'authority.sqlite'
    service_actor = 'other-owner' if mode == 'actor_mismatch' else owner.user_id
    service = AuthoritySocketService(authority_path, endpoints={
        'desktop': EndpointIdentity('desktop', service_actor, 'synthetic-desktop'),
        'worker': EndpointIdentity('worker', service_actor, 'synthetic-worker'),
    }, verify_peer=lambda *_: True)
    desktop, worker = _client(service, 'desktop'), _client(service, 'worker')
    if mode != 'grant_disabled':
        desktop.grant_bash(expected_generation=desktop.bash_capability()['generation'])
    goal = 'Create a synthetic prospect note and check it.'
    authority_request = desktop.submit(goal=goal, idempotency_key='a' * 32)['request_id']
    if mode == 'capability_query_unavailable':
        class UnavailableCapabilityQuery:
            def __getattr__(self, name):
                return getattr(worker, name)
            def check_bash_capability(self, **kwargs):
                raise OSError('synthetic discovery failure')
        worker_client = UnavailableCapabilityQuery()
    elif mode in {'cancel_before_start', 'revoke_before_start'}:
        class CancelBeforeStart:
            def __getattr__(self, name):
                return getattr(worker, name)
            def start_action(self, **kwargs):
                if mode == 'cancel_before_start':
                    desktop.cancel(request_id=authority_request)
                else:
                    desktop.revoke_bash(expected_generation=desktop.bash_capability()['generation'])
                return worker.start_action(**kwargs)
        worker_client = CancelBeforeStart()
    elif mode == 'lost_start_reply':
        class LostStartReply:
            def __getattr__(self, name):
                return getattr(worker, name)
            def start_action(self, **kwargs):
                worker.start_action(**kwargs)
                raise OSError('synthetic lost response')
        worker_client = LostStartReply()
    else:
        worker_client = worker
    candidate = runtime.processor.owner_read_candidate
    assert candidate._owner_host_bash_enabled is False
    candidate.enable_owner_host_bash()
    if mode == 'provisioned_bridge':
        # Exercise the production connection assembly against the same actual
        # owner worker and disposable ledger. The protected file and kernel
        # identity checks are faked here; this is not a deployment proof.
        from alphonse.agent_v2.authority_service.macos import worker_endpoint
        source = tmp_path / 'worker-config.json'
        source.write_text(json.dumps({'version': 1, 'service_uid': 501,
                                      'owner_uid': 502, 'socket_path': '/private/tmp/fixture.sock'}))
        with monkeypatch.context() as provision:
            provision.setattr(worker_endpoint.sys, 'platform', 'darwin')
            provision.setattr(worker_endpoint.os, 'getuid', lambda: 502)
            provision.setattr(worker_endpoint.os, 'geteuid', lambda: 502)
            provision.setattr(worker_endpoint, '_open_checked', lambda *a, **kw: os.open(source, os.O_RDONLY))
            provision.setattr(worker_endpoint, 'macos_authority_client', lambda *a, **kw: worker_client)
            candidate.attach_experimental_owner_authority(worker_endpoint.provisioned_worker_bridge(queue))
    elif mode != 'no_bridge':
        candidate.attach_experimental_owner_authority(OwnerAuthorityBridge(
            worker_client, lambda task: 'f' * 32 if mode == 'missing_binding' else queue_request_binding(queue, task)))
    daemon = V2Daemon(runtime, daemon_id='authority-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        message_id = queue.enqueue_trusted_local_owner(prompt=goal, peer_uid=os.getuid(),
            _authority_binding=('a' * 32, authority_request))
        if mode in {'cancel_running', 'revoke_regrant_running'}:
            # Wait for the actual process to begin, then revoke the service
            # claim while it sleeps. No user files are involved.
            started_file = users.ordinary_workspace(owner.user_id) / 'started.txt'
            started_deadline = time.monotonic() + 5
            while time.monotonic() < started_deadline and not started_file.exists():
                time.sleep(.01)
            assert started_file.read_text() == 'started'
            if mode == 'cancel_running':
                cancelled = desktop.cancel(request_id=authority_request)
                assert cancelled['execution_may_have_started'] is True
            else:
                desktop.revoke_bash(expected_generation=desktop.bash_capability()['generation'])
                desktop.grant_bash(expected_generation=desktop.bash_capability()['generation'])
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            status = daemon.owner_message_status(message_id)
            if status not in {'pending', 'processing'}:
                break
            time.sleep(.01)
        assert status == ('ready' if mode in {'allowed', 'provisioned_bridge', 'grant_disabled', 'capability_query_unavailable'} else 'failed')
        output = users.ordinary_workspace(owner.user_id) / 'prospect.txt'
        with sqlite3.connect(authority_path) as conn:
            actions = conn.execute('SELECT state FROM actions ORDER BY rowid').fetchall()
        if mode in {'grant_disabled', 'capability_query_unavailable'}:
            assert not output.exists()
            assert not actions
            assert daemon.take_owner_desktop_draft(message_id)['message'] == 'Hello, how can I help?'
        elif mode not in {'allowed', 'provisioned_bridge'}:
            assert not output.exists()
            assert actions == ([('reserved',)] if mode in {'cancel_before_start', 'revoke_before_start'}
                               else [('started',)] if mode in {'cancel_running', 'revoke_regrant_running', 'lost_start_reply'} else [])
        else:
            assert output.read_text() == 'synthetic'
            assert actions == [('return_reported',), ('return_reported',)]
            draft = daemon.take_owner_desktop_draft(message_id)
            assert draft['message'] == 'The note contains synthetic.'
            daemon.acknowledge_owner_desktop_draft(message_id, draft['acknowledgement_token'])
    finally:
        daemon.stop()
