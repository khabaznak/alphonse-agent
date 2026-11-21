"""Cross-store observations never authorize replay or mutate either store."""
from types import SimpleNamespace
import os
import socket
import sqlite3
import threading

from alphonse.agent_v2.authority_service.prototype import AuthenticatedPeer, AuthorityService
from alphonse.agent_v2.authority_service.reconciliation import inspect_owner_request
from alphonse.agent_v2.authority_service.transport import AuthorityClient, AuthoritySocketService, EndpointIdentity
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.users import V2UserStore


def fixture(tmp_path):
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    SQLiteConversationStore(path)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    # A prior unrelated ingress initializes the actual origin schema, without
    # creating the request under inspection.
    queue.enqueue_trusted_local_owner(prompt='Unrelated synthetic request', peer_uid=os.getuid())
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS v2_authority_ingress_bindings
                        (message_id TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE) STRICT''')
    desktop, worker = object(), object()
    def authenticate(peer):
        if peer is desktop:
            return AuthenticatedPeer('desktop', owner.user_id, 'desktop-fixture')
        if peer is worker:
            return AuthenticatedPeer('worker', owner.user_id, 'worker-fixture')
        raise ValueError()
    service = AuthorityService(tmp_path / 'authority.sqlite', authenticate=authenticate)
    client = SimpleNamespace(inspect_request=lambda **kwargs: service.inspect_request(desktop, **kwargs),
                             lookup_request=lambda **kwargs: service.lookup_request(desktop, **kwargs))
    return path, owner.user_id, queue, service, client, desktop, worker


def observe(path, owner, client, message_id, request_id, goal):
    return inspect_owner_request(desktop_client=client, queue_db=path, actor=owner,
                                 message_id=message_id, request_id=request_id, goal=goal)


def test_crash_after_service_submit_never_reenqueues_or_replays(tmp_path):
    path, owner, queue, service, client, desktop, _ = fixture(tmp_path)
    goal, message_id = 'Synthetic owner task', 'a' * 32
    request_id = service.submit(desktop, goal=goal, idempotency_key=message_id)['request_id']
    result = observe(path, owner, client, message_id, request_id, goal)
    assert result == {'authority': 'pending', 'queue': 'absent', 'started_actions': 0,
                      'recovery': 'local_row_absent', 'may_auto_replay': False}
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()[0] == 0


def test_lost_submit_reply_can_resolve_request_from_protected_ledger(tmp_path):
    path, owner, queue, service, client, desktop, _ = fixture(tmp_path)
    goal, message_id = 'Synthetic owner task', 'a' * 32
    request_id = service.submit(desktop, goal=goal, idempotency_key=message_id)['request_id']
    # The caller retained the message ID, but not the service response.
    result = inspect_owner_request(desktop_client=client, queue_db=path, actor=owner,
                                   message_id=message_id, request_id=None, goal=goal)
    assert result['resolved_request_id'] == request_id
    assert result['authority'] == 'pending' and result['queue'] == 'absent'
    assert result['may_auto_replay'] is False
    denied = inspect_owner_request(desktop_client=client, queue_db=path, actor=owner,
                                   message_id=message_id, request_id=None, goal='wrong goal')
    assert denied['authority'] == 'unavailable_or_mismatch' and denied['queue'] == 'not_checked'
    missing = inspect_owner_request(desktop_client=client, queue_db=path, actor=owner,
                                    message_id='e' * 32, request_id=None, goal=goal)
    assert missing['authority'] == 'not_found' and missing['may_auto_replay'] is False


def test_linked_queue_and_cancellation_are_observed_without_grant(tmp_path):
    path, owner, queue, service, client, desktop, _ = fixture(tmp_path)
    goal = 'Synthetic owner task'
    message_id = queue.enqueue_trusted_local_owner(prompt=goal, peer_uid=os.getuid())
    request_id = service.submit(desktop, goal=goal, idempotency_key=message_id)['request_id']
    with sqlite3.connect(path) as conn:
        conn.execute('INSERT INTO v2_authority_ingress_bindings VALUES (?,?)', (message_id, request_id))
    result = observe(path, owner, client, message_id, request_id, goal)
    assert result['queue'] == 'pending' and result['recovery'] == 'awaiting_authorized_claim'
    assert result['may_auto_replay'] is False
    service.cancel(desktop, request_id=request_id)
    result = observe(path, owner, client, message_id, request_id, goal)
    assert result['authority'] == 'cancelled' and result['recovery'] == 'cancelled'
    assert result['may_auto_replay'] is False


def test_started_action_is_uncertain_even_if_local_row_disappears(tmp_path):
    path, owner, queue, service, client, desktop, worker = fixture(tmp_path)
    service.grant_bash(desktop, expected_generation=service.bash_capability(desktop)['generation'])
    goal, message_id = 'Synthetic owner task', 'b' * 32
    request_id = service.submit(desktop, goal=goal, idempotency_key=message_id)['request_id']
    claim = service.claim(worker, request_id=request_id, message_id=message_id, goal=goal, expected_actor=owner)
    action = {'tool_id': 'native.bash', 'arguments': {'command': 'printf synthetic'}}
    action_id = service.reserve_action(worker, claim=claim, action=action, idempotency_key='c' * 32)['action_id']
    service.start_action(worker, claim=claim, action_id=action_id, action=action)
    result = observe(path, owner, client, message_id, request_id, goal)
    assert result['authority'] == 'claimed' and result['queue'] == 'absent'
    assert result['started_actions'] == 1 and result['recovery'] == 'execution_uncertain'
    assert result['may_auto_replay'] is False


def test_local_tampering_and_protected_mismatch_do_not_authorize(tmp_path):
    path, owner, queue, service, client, desktop, _ = fixture(tmp_path)
    goal = 'Synthetic owner task'
    message_id = queue.enqueue_trusted_local_owner(prompt=goal, peer_uid=os.getuid())
    request_id = service.submit(desktop, goal=goal, idempotency_key=message_id)['request_id']
    with sqlite3.connect(path) as conn:
        conn.execute('INSERT INTO v2_authority_ingress_bindings VALUES (?,?)', (message_id, request_id))
        conn.execute('UPDATE v2_inbound_messages SET prompt=? WHERE message_id=?', ('altered', message_id))
    assert observe(path, owner, client, message_id, request_id, goal)['queue'] == 'mismatch'
    result = observe(path, owner, client, message_id, request_id, 'wrong goal')
    assert result['authority'] == 'unavailable_or_mismatch' and result['queue'] == 'not_checked'
    assert result['may_auto_replay'] is False
    result = observe(path, owner, client, 'd' * 32, request_id, goal)
    assert result['authority'] == 'unavailable_or_mismatch' and result['queue'] == 'not_checked'


def test_failed_queue_row_is_not_reported_as_waiting_to_run(tmp_path):
    path, owner, queue, service, client, desktop, _ = fixture(tmp_path)
    goal = 'Synthetic owner task'
    message_id = queue.enqueue_trusted_local_owner(prompt=goal, peer_uid=os.getuid())
    request_id = service.submit(desktop, goal=goal, idempotency_key=message_id)['request_id']
    with sqlite3.connect(path) as conn:
        conn.execute('INSERT INTO v2_authority_ingress_bindings VALUES (?,?)', (message_id, request_id))
        conn.execute("UPDATE v2_inbound_messages SET status='failed' WHERE message_id=?", (message_id,))
    result = observe(path, owner, client, message_id, request_id, goal)
    assert result['queue'] == 'failed' and result['recovery'] == 'local_state_requires_review'
    assert result['may_auto_replay'] is False


def test_socket_ledger_and_real_queue_link_after_lost_reply(tmp_path):
    path, owner, queue, _, _, _, _ = fixture(tmp_path)
    service = AuthoritySocketService(tmp_path / 'socket-ledger.sqlite', endpoints={
        'desktop': EndpointIdentity('desktop', owner, 'synthetic-desktop'),
        'worker': EndpointIdentity('worker', owner, 'synthetic-worker'),
    }, verify_peer=lambda *_: True)
    def connect():
        client, server = socket.socketpair()
        threading.Thread(target=service.serve_connection, args=(server,),
                         kwargs={'endpoint_role': 'desktop'}, daemon=True).start()
        return client
    client = AuthorityClient(connect)
    goal = 'Synthetic owner task'
    message_id = queue.enqueue_trusted_local_owner(prompt=goal, peer_uid=os.getuid())
    request_id = client.submit(goal=goal, idempotency_key=message_id)['request_id']
    with sqlite3.connect(path) as conn:
        conn.execute('INSERT INTO v2_authority_ingress_bindings VALUES (?,?)', (message_id, request_id))
    result = inspect_owner_request(desktop_client=client, queue_db=path, actor=owner,
                                   message_id=message_id, request_id=None, goal=goal)
    assert result['resolved_request_id'] == request_id and result['queue'] == 'pending'
    assert result['recovery'] == 'awaiting_authorized_claim' and result['may_auto_replay'] is False
