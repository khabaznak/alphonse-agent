"""Disposable socket/queue integration; fake identity is not a security boundary."""
import os
import socket
import tempfile
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.authority_service.experimental_ingress import (
    ExperimentalIngressUnavailable, queue_request_binding, submit_owner_request,
)
from alphonse.agent_v2.authority_service.prototype import AuthorityDenied
from alphonse.agent_v2.authority_service.transport import AuthorityClient, AuthoritySocketService, EndpointIdentity
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_authority_bridge import OwnerAuthorityBridge, OwnerAuthorityUnavailable
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.users import V2UserStore


def service_client(service, role):
    def connect():
        client, server = socket.socketpair()
        threading.Thread(target=service.serve_connection, args=(server,),
                         kwargs={'endpoint_role': role}, daemon=True).start()
        return client
    return AuthorityClient(connect)


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    SQLiteConversationStore(path)
    service = AuthoritySocketService(tmp_path / 'authority.sqlite', endpoints={
        'desktop': EndpointIdentity('desktop', owner.user_id, 'synthetic-desktop'),
        'worker': EndpointIdentity('worker', owner.user_id, 'synthetic-worker'),
    }, verify_peer=lambda *_: True)
    with tempfile.TemporaryDirectory(prefix='owner-authority-', dir='/private/tmp') as directory:
        socket_path = Path(directory) / 'owner.sock'
        server = V2DaemonServer(SimpleNamespace(runtime=SimpleNamespace(queue=queue)), socket_path,
                                experimental_owner_ingress=True, experimental_authority_ingress=True)
        server.start()
        try:
            yield queue, service_client(service, 'desktop'), service_client(service, 'worker'), V2DaemonClient(socket_path), owner
        finally:
            server.stop()


def test_same_goal_different_message_cannot_steal_request_via_tampered_queue_binding(setup):
    queue, desktop, worker, client, owner = setup
    first = submit_owner_request(desktop, client, prompt='  inspect  ', message_id='a' * 32)
    second = submit_owner_request(desktop, client, prompt='inspect', message_id='b' * 32)
    with queue._connect() as conn:
        rows = conn.execute('SELECT message_id,request_id FROM v2_authority_ingress_bindings ORDER BY message_id').fetchall()
        first_request, second_request = rows[0]['request_id'], rows[1]['request_id']
        conn.execute('DELETE FROM v2_authority_ingress_bindings WHERE message_id=?', (second['message_id'],))
        conn.execute('UPDATE v2_authority_ingress_bindings SET request_id=? WHERE message_id=?',
                     (second_request, first['message_id']))
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=first['message_id'])
    task.task_id = first['message_id']
    assert task.user == owner.user_id and task.goal == 'inspect'
    assert queue_request_binding(queue, task) == second_request  # local metadata can be tampered with
    bridge = OwnerAuthorityBridge(worker, lambda active: queue_request_binding(queue, active))
    with pytest.raises(OwnerAuthorityUnavailable):
        bridge.begin(task)  # protected service rejects different origin ID
    assert desktop.inspect_request(request_id=first_request, message_id=first['message_id'], goal='inspect')['status'] == 'pending'
    assert desktop.inspect_request(request_id=second_request, message_id=second['message_id'], goal='inspect')['status'] == 'pending'


def test_lost_submit_reply_does_not_enqueue_and_can_be_inspected(setup):
    queue, desktop, _, client, owner = setup
    class LostResponse:
        def submit(self, **kwargs):
            desktop.submit(**kwargs)
            raise OSError('synthetic response lost')
    with pytest.raises(ExperimentalIngressUnavailable):
        submit_owner_request(LostResponse(), client, prompt='prepare', message_id='c' * 32)
    assert queue.size() == 0
    found = desktop.lookup_request(message_id='c' * 32, goal='prepare')
    assert found['found'] is True and found['actor'] == owner.user_id and found['status'] == 'pending'
    with pytest.raises(AuthorityDenied):
        desktop.lookup_request(message_id='c' * 32, goal='different')
    receipt = submit_owner_request(desktop, client, prompt='prepare', message_id='c' * 32)
    assert receipt['message_id'] == 'c' * 32 and queue.size() == 1


def test_lost_queue_reply_and_retry_never_duplicate_owner_message(setup):
    queue, desktop, _, client, _ = setup
    class LostResponse:
        def request(self, *args, **kwargs):
            client.request(*args, **kwargs)
            raise OSError('synthetic response lost')
    with pytest.raises(ExperimentalIngressUnavailable):
        submit_owner_request(desktop, LostResponse(), prompt='prepare', message_id='d' * 32)
    assert queue.size() == 1
    # Repeating a request is not silently treated as a second enqueue or success.
    with pytest.raises(ExperimentalIngressUnavailable):
        submit_owner_request(desktop, client, prompt='prepare', message_id='d' * 32)
    assert queue.size() == 1
