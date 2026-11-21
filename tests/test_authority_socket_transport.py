"""Socket-to-ledger tests use fake peer verification, not OS isolation."""
import json
import socket
import struct
import threading
import tempfile
from pathlib import Path
import time

import pytest

from alphonse.agent_v2.authority_service.prototype import AuthorityDenied
from alphonse.agent_v2.authority_service import transport
from alphonse.agent_v2.authority_service.transport import (
    AuthorityClient, AuthoritySocketService, EndpointIdentity, _read_frame, _write_frame,
)


@pytest.fixture
def service(tmp_path):
    endpoints = {'desktop': EndpointIdentity('desktop', 'synthetic-owner', 'desktop-app'),
                 'worker': EndpointIdentity('worker', 'synthetic-owner', 'owner-worker')}
    return AuthoritySocketService(tmp_path / 'ledger.sqlite', endpoints=endpoints,
                                  verify_peer=lambda _conn, _role: True,
                                  verify_listener=lambda _conn: True)


def client_for(service, role):
    threads = []
    def connect():
        client, server = socket.socketpair()
        thread = threading.Thread(target=service.serve_connection, args=(server,),
                                  kwargs={'endpoint_role': role})
        thread.start()
        threads.append(thread)
        return client
    return AuthorityClient(connect), threads


def test_framed_ledger_flow_binds_role_and_records_unverified_effect(service):
    desktop, desktop_threads = client_for(service, 'desktop')
    worker, worker_threads = client_for(service, 'worker')
    assert desktop.grant_bash(expected_generation=desktop.bash_capability()['generation']) == {'enabled': True, 'generation': 1}
    request = desktop.submit(goal='Create a note', idempotency_key='a' * 32)
    with pytest.raises(AuthorityDenied):
        worker.submit(goal='Impersonate desktop', idempotency_key='b' * 32)
    with pytest.raises(AuthorityDenied):
        worker.claim(request_id=request['request_id'], message_id='b' * 32, goal='Create a note', expected_actor='synthetic-owner')
    assert desktop.inspect_request(request_id=request['request_id'], message_id='a' * 32,
                                   goal='Create a note')['actor'] == 'synthetic-owner'
    receipt = worker.claim(request_id=request['request_id'], message_id='a' * 32, goal='Create a note', expected_actor='synthetic-owner')
    assert receipt['actor'] == 'synthetic-owner'
    claim = receipt['claim']
    assert worker.check_claim(claim=claim) is True
    action = {'tool_id': 'native.bash', 'arguments': {'command': 'printf test'}}
    reserved = worker.reserve_action(claim=claim, action=action, idempotency_key='c' * 32)
    assert worker.start_action(claim=claim, action_id=reserved['action_id'], action=action)['effect'] == 'unverified'
    with pytest.raises(AuthorityDenied):
        worker.start_action(claim=claim, action_id=reserved['action_id'], action=action)
    assert worker.record_process_return(claim=claim, action_id=reserved['action_id'],
                                        process_result={'exit_code': 0})['effect'] == 'unverified'
    assert desktop.cancel(request_id=request['request_id'])['execution_may_have_started'] is True
    with pytest.raises(AuthorityDenied):
        worker.check_claim(claim=claim)
    for thread in desktop_threads + worker_threads:
        thread.join(timeout=3)
        assert not thread.is_alive()


def test_bash_capability_is_owner_scoped_persistent_and_revocation_fences_old_claim(service):
    desktop, _ = client_for(service, 'desktop')
    worker, _ = client_for(service, 'worker')
    assert desktop.bash_capability() == {'enabled': False, 'generation': 0}
    with pytest.raises(AuthorityDenied):
        worker.grant_bash(expected_generation=0)
    goal = 'Synthetic task'
    request = desktop.submit(goal=goal, idempotency_key='1' * 32)
    claim = worker.claim(request_id=request['request_id'], message_id='1' * 32,
                         goal=goal, expected_actor='synthetic-owner')['claim']
    assert worker.check_claim(claim=claim) is True
    action = {'tool_id': 'native.bash', 'arguments': {'command': 'printf test'}}
    with pytest.raises(AuthorityDenied):
        worker.reserve_action(claim=claim, action=action, idempotency_key='2' * 32)
    assert desktop.grant_bash(expected_generation=desktop.bash_capability()['generation']) == {'enabled': True, 'generation': 1}
    assert desktop.grant_bash(expected_generation=desktop.bash_capability()['generation']) == {'enabled': True, 'generation': 1}
    reserved = worker.reserve_action(claim=claim, action=action, idempotency_key='2' * 32)
    assert desktop.revoke_bash(expected_generation=desktop.bash_capability()['generation']) == {'enabled': False, 'generation': 2}
    with pytest.raises(AuthorityDenied):
        worker.start_action(claim=claim, action_id=reserved['action_id'], action=action)
    assert desktop.grant_bash(expected_generation=desktop.bash_capability()['generation']) == {'enabled': True, 'generation': 3}
    with pytest.raises(AuthorityDenied):
        worker.check_claim(claim=claim)
    with pytest.raises(AuthorityDenied):
        worker.start_action(claim=claim, action_id=reserved['action_id'], action=action)
    # Reconstructing the service keeps the persisted grant and fenced claim.
    restored = AuthoritySocketService(service._ledger.path, endpoints=service._endpoints,
                                      verify_peer=lambda *_: True)
    restored_desktop, _ = client_for(restored, 'desktop')
    restored_worker, _ = client_for(restored, 'worker')
    assert restored_desktop.bash_capability() == {'enabled': True, 'generation': 3}
    with pytest.raises(AuthorityDenied):
        restored_worker.check_claim(claim=claim)


def test_bash_mutations_require_generation_and_fence_pending_grant(service):
    desktop, _ = client_for(service, 'desktop')
    with pytest.raises(AuthorityDenied):
        desktop._call('grant_bash')
    with pytest.raises(AuthorityDenied):
        desktop._call('revoke_bash')
    assert desktop.revoke_bash(expected_generation=0) == {'enabled': False, 'generation': 1}
    with pytest.raises(AuthorityDenied):
        desktop.grant_bash(expected_generation=0)
    assert desktop.bash_capability() == {'enabled': False, 'generation': 1}


@pytest.mark.parametrize('verification', [lambda *_: False, lambda *_: (_ for _ in ()).throw(RuntimeError())])
def test_unverified_peer_receives_no_reply_and_never_reaches_ledger(tmp_path, verification):
    endpoints = {'desktop': EndpointIdentity('desktop', 'owner', 'd'),
                 'worker': EndpointIdentity('worker', 'owner', 'w')}
    service = AuthoritySocketService(tmp_path / 'state.sqlite', endpoints=endpoints, verify_peer=verification)
    client, server = socket.socketpair()
    thread = threading.Thread(target=service.serve_connection, args=(server,), kwargs={'endpoint_role': 'desktop'})
    thread.start()
    client.settimeout(3)
    assert client.recv(10) == b''
    client.close()
    thread.join(timeout=3)
    with service._ledger._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM requests').fetchone()[0] == 0


def test_untrusted_identity_fields_and_malformed_frames_are_rejected(service):
    for payload in [
        b'{"op":"submit","arguments":{"goal":"Hi","idempotency_key":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","actor":"other"}}',
        b'{"op":"submit","op":"cancel","arguments":{}}',
        b'{"op":"claim","arguments":{"request_id":"x","goal":"x","lease_seconds":NaN}}',
    ]:
        client, server = socket.socketpair()
        thread = threading.Thread(target=service.serve_connection, args=(server,),
                                  kwargs={'endpoint_role': 'desktop'})
        thread.start()
        client.sendall(struct.pack('!I', len(payload)) + payload)
        response = _read_frame(client)
        assert response == {'ok': False, 'error': 'authority_request_denied'}
        client.close()
        thread.join(timeout=3)
    with service._ledger._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM requests').fetchone()[0] == 0


def test_oversize_frame_is_rejected_before_reading_body(service):
    client, server = socket.socketpair()
    thread = threading.Thread(target=service.serve_connection, args=(server,), kwargs={'endpoint_role': 'worker'})
    thread.start()
    client.sendall(struct.pack('!I', 131073))
    assert _read_frame(client) == {'ok': False, 'error': 'authority_request_denied'}
    client.close()
    thread.join(timeout=3)


def test_endpoint_role_and_actor_are_preconfigured(tmp_path):
    with pytest.raises(ValueError):
        AuthoritySocketService(tmp_path / 'bad.sqlite', endpoints={
            'desktop': EndpointIdentity('desktop', 'alice', 'same'),
            'worker': EndpointIdentity('worker', 'bob', 'same'),
        }, verify_peer=lambda *_: True)


@pytest.mark.parametrize('check', [None, lambda _: False, lambda _: (_ for _ in ()).throw(OSError())])
def test_listener_loop_refuses_missing_or_failed_kernel_check(service, check):
    service._verify_listener = check
    first, second = socket.socketpair()
    try:
        with pytest.raises(ValueError, match='authority_listener_configuration_invalid'):
            service.serve_listeners({'desktop': first, 'worker': second}, stop=threading.Event())
    finally:
        first.close()
        second.close()


def test_prebound_listeners_route_roles_and_are_not_removed(service):
    with tempfile.TemporaryDirectory(prefix='authority-protocol-', dir='/private/tmp') as directory:
        root = Path(directory)
        listeners = {}
        for role in ('desktop', 'worker'):
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(root / (role + '.sock')))
            listener.listen(4)
            listeners[role] = listener
        stop = threading.Event()
        server = threading.Thread(target=service.serve_listeners, args=(listeners,), kwargs={'stop': stop})
        server.start()
        def connect(role):
            peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            peer.settimeout(3)
            peer.connect(str(root / (role + '.sock')))
            return peer
        try:
            desktop = AuthorityClient(lambda: connect('desktop'))
            worker = AuthorityClient(lambda: connect('worker'))
            request = desktop.submit(goal='inspect', idempotency_key='d' * 32)
            receipt = worker.claim(request_id=request['request_id'], message_id='d' * 32, goal='inspect', expected_actor='synthetic-owner')
            assert receipt['actor'] == 'synthetic-owner'
            assert worker.check_claim(claim=receipt['claim']) is True
            with pytest.raises(AuthorityDenied):
                desktop.check_claim(claim=receipt['claim'])
        finally:
            stop.set()
            server.join(timeout=4)
            for listener in listeners.values():
                listener.close()
        assert not server.is_alive()
        assert (root / 'desktop.sock').exists() and (root / 'worker.sock').exists()


def test_trickled_request_cannot_keep_connection_open_indefinitely(service, monkeypatch):
    monkeypatch.setattr(transport, '_CONNECTION_DEADLINE_SECONDS', .25)
    client, server = socket.socketpair()
    thread = threading.Thread(target=service.serve_connection, args=(server,),
                              kwargs={'endpoint_role': 'desktop'})
    thread.start()
    started = time.monotonic()
    try:
        client.sendall(struct.pack('!I', 100))
        for _ in range(10):
            try:
                client.sendall(b' ')
            except (BrokenPipeError, ConnectionResetError):
                break
            time.sleep(.08)
        thread.join(timeout=1)
        assert not thread.is_alive()
        assert time.monotonic() - started < 1.5
    finally:
        client.close()
        thread.join(timeout=1)
    with service._ledger._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM requests').fetchone()[0] == 0


def test_excess_connections_are_closed_without_reading(service):
    with tempfile.TemporaryDirectory(prefix='authority-protocol-', dir='/private/tmp') as directory:
        root = Path(directory)
        listeners = {}
        for role in ('desktop', 'worker'):
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(root / (role + '.sock')))
            listener.listen(4)
            listeners[role] = listener
        entered = threading.Event()
        service._verify_peer = lambda *_: (entered.set() or True)
        stop = threading.Event()
        server = threading.Thread(target=service.serve_listeners, args=(listeners,),
                                  kwargs={'stop': stop, 'max_connections': 1})
        server.start()
        first = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        second = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            first.connect(str(root / 'desktop.sock'))
            assert entered.wait(2)
            second.settimeout(3)
            second.connect(str(root / 'worker.sock'))
            assert second.recv(1) == b''
        finally:
            first.close()
            second.close()
            stop.set()
            server.join(timeout=4)
            for listener in listeners.values():
                listener.close()
        assert not server.is_alive()
