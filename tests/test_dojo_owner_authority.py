from types import SimpleNamespace
import socket
import threading

import pytest

from alphonse.agent_v2.authority_service.prototype import AuthorityDenied, AuthenticatedPeer, AuthorityService
from alphonse.agent_v2.evaluation.dojo_owner_authority import DojoOwnerAuthority


@pytest.fixture
def setup(tmp_path):
    desktop, worker = object(), object()
    peers = {desktop: AuthenticatedPeer('desktop', 'owner', 'desktop'),
             worker: AuthenticatedPeer('worker', 'owner', 'worker')}
    service = AuthorityService(tmp_path / 'authority.sqlite', authenticate=peers.__getitem__)
    service.grant_bash(desktop, expected_generation=service.bash_capability(desktop)['generation'])
    events = []
    authority = DojoOwnerAuthority(service, desktop, worker, emit=events.append)
    return authority, service, desktop, events


def test_owner_claim_and_actions_are_bound_and_returns_unverified(setup):
    authority, service, _, events = setup
    task = SimpleNamespace(user='owner', task_id='message', goal='Find a shopping list')
    authority.begin(task)
    assert authority.active(task)
    action = authority.start(task, 'native.bash', {'command': 'ls'}, reservation_id='tool-' + '1' * 32)
    authority.returned(task, action, {'status': 'completed', 'exit_code': 0, 'truncated': False})
    with service._connect() as db:
        assert db.execute('SELECT state FROM actions').fetchone()['state'] == 'return_reported'
    assert events[-1] == {'event': 'authority_return_reported', 'effect': 'unverified'}
    with pytest.raises(AuthorityDenied):
        authority.start(SimpleNamespace(user='owner', task_id='message', goal='A different goal'),
                        'native.bash', {'command': 'ls'}, reservation_id='tool-' + '1' * 32)


def test_cancellation_revokes_claim_and_blocks_action(setup):
    authority, service, desktop, _ = setup
    task = SimpleNamespace(user='owner', task_id='message', goal='Find a shopping list')
    authority.begin(task)
    request_id = authority._claim(task)['request_id']
    service.cancel(desktop, request_id=request_id)
    assert not authority.active(task)
    with pytest.raises(AuthorityDenied):
        authority.start(task, 'native.bash', {'command': 'ls'}, reservation_id='tool-' + '1' * 32)


def test_synthetic_actor_mismatch_cannot_claim(setup):
    authority, _, _, events = setup
    with pytest.raises(AuthorityDenied):
        authority.begin(SimpleNamespace(user='someone_else', task_id='message', goal='Read files'))
    assert events[-1]['event'] == 'authority_denied'


def test_socket_dojo_uses_preexisting_service_request_and_records_action(tmp_path):
    from alphonse.agent_v2.authority_service.transport import AuthorityClient, AuthoritySocketService, EndpointIdentity
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_authority_bridge import OwnerAuthorityBridge

    service = AuthoritySocketService(tmp_path / 'authority.sqlite', endpoints={
        'desktop': EndpointIdentity('desktop', 'owner', 'desktop'),
        'worker': EndpointIdentity('worker', 'owner', 'worker'),
    }, verify_peer=lambda *_: True)
    def client(role):
        def connect():
            left, right = socket.socketpair()
            left.settimeout(3)
            threading.Thread(target=service.serve_connection, args=(right,),
                             kwargs={'endpoint_role': role}, daemon=True).start()
            return left
        return AuthorityClient(connect)
    desktop, worker = client('desktop'), client('worker')
    desktop.grant_bash(expected_generation=desktop.bash_capability()['generation'])
    task = SimpleNamespace(user='owner', task_id='a' * 32, goal='Synthetic file request')
    request_id = desktop.submit(goal=task.goal, idempotency_key=task.task_id)['request_id']
    events = []
    adapter = DojoOwnerAuthority.from_worker_bridge(OwnerAuthorityBridge(worker, lambda _: request_id),
                                                     emit=events.append)
    adapter.begin(task)
    action = adapter.start(task, 'native.bash', {'command': 'printf synthetic'}, reservation_id='tool-' + 'b' * 32)
    adapter.returned(task, action, {'status': 'completed', 'exit_code': 0, 'truncated': False})
    assert [item['event'] for item in events] == ['authority_request_claimed', 'authority_action_started',
                                                   'authority_return_reported']
    with service._ledger._connect() as conn:
        assert conn.execute('SELECT state FROM actions').fetchone()['state'] == 'return_reported'


def test_restart_with_same_checkpoint_reservation_cannot_repeat_started_action(setup):
    authority, service, _, _ = setup
    task = SimpleNamespace(user='owner', task_id='message', goal='Create a file')
    reservation = 'tool-' + '2' * 32
    authority.begin(task)
    claim = authority._claim(task)
    authority.start(task, 'native.bash', {'command': 'touch /tmp/file'}, reservation_id=reservation)

    # Reopen the service ledger and model recovery with the same authenticated
    # claim and checkpoint reservation. This is not production claim recovery.
    restarted_service = AuthorityService(service.path, authenticate=service.authenticate)
    restarted = DojoOwnerAuthority(restarted_service, authority.desktop, authority.worker)
    restarted._claims[(task.user, task.task_id, task.goal)] = claim
    with pytest.raises(AuthorityDenied, match='authority_action_unavailable'):
        restarted.start(task, 'native.bash', {'command': 'touch /tmp/file'}, reservation_id=reservation)
    with restarted_service._connect() as db:
        assert [row['state'] for row in db.execute('SELECT state FROM actions')] == ['started']

    # The same reservation cannot be rebound to different arguments.
    with pytest.raises(AuthorityDenied, match='authority_idempotency_conflict'):
        restarted.start(task, 'native.bash', {'command': 'rm /tmp/file'}, reservation_id=reservation)


def test_distinct_checkpoint_reservations_allow_distinct_actions(setup):
    authority, service, _, _ = setup
    task = SimpleNamespace(user='owner', task_id='message', goal='Inspect files')
    authority.begin(task)
    first = authority.start(task, 'native.bash', {'command': 'ls'}, reservation_id='tool-' + '3' * 32)
    second = authority.start(task, 'native.bash', {'command': 'ls'}, reservation_id='tool-' + '4' * 32)
    assert first != second
    with service._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM actions').fetchone()[0] == 2


def test_missing_or_invalid_checkpoint_reservation_fails_closed(setup):
    authority, service, _, _ = setup
    task = SimpleNamespace(user='owner', task_id='message', goal='Inspect files')
    authority.begin(task)
    with pytest.raises(AuthorityDenied, match='dojo_authority_reservation_invalid'):
        authority.start(task, 'native.bash', {'command': 'ls'}, reservation_id='')
    with service._connect() as db:
        assert db.execute('SELECT COUNT(*) FROM actions').fetchone()[0] == 0


def test_lost_start_reply_does_not_dispatch_again(setup, monkeypatch):
    authority, service, _, _ = setup
    task = SimpleNamespace(user='owner', task_id='message', goal='Create a file')
    reservation = 'tool-' + '5' * 32
    authority.begin(task)
    start = service.start_action

    def lose_reply(*args, **kwargs):
        start(*args, **kwargs)  # Ledger commits, but caller does not receive the reply.
        raise RuntimeError('connection_lost')

    monkeypatch.setattr(service, 'start_action', lose_reply)
    with pytest.raises(RuntimeError, match='connection_lost'):
        authority.start(task, 'native.bash', {'command': 'touch /tmp/file'}, reservation_id=reservation)
    monkeypatch.setattr(service, 'start_action', start)
    with pytest.raises(AuthorityDenied, match='authority_action_unavailable'):
        authority.start(task, 'native.bash', {'command': 'touch /tmp/file'}, reservation_id=reservation)
    with service._connect() as db:
        assert [row['state'] for row in db.execute('SELECT state FROM actions')] == ['started']


def test_failure_before_start_can_resume_same_reserved_action(setup, monkeypatch):
    authority, service, _, _ = setup
    task = SimpleNamespace(user='owner', task_id='message', goal='Create a file')
    reservation = 'tool-' + '6' * 32
    authority.begin(task)
    start = service.start_action

    def fail_before_start(*_args, **_kwargs):
        raise RuntimeError('connection_lost')

    monkeypatch.setattr(service, 'start_action', fail_before_start)
    with pytest.raises(RuntimeError, match='connection_lost'):
        authority.start(task, 'native.bash', {'command': 'touch /tmp/file'}, reservation_id=reservation)
    monkeypatch.setattr(service, 'start_action', start)
    authority.start(task, 'native.bash', {'command': 'touch /tmp/file'}, reservation_id=reservation)
    with service._connect() as db:
        assert [row['state'] for row in db.execute('SELECT state FROM actions')] == ['started']
