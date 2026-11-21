"""Tests of service-state semantics, not OS identity isolation."""
import sqlite3
import threading

import pytest

from alphonse.agent_v2.authority_service.prototype import (
    AuthenticatedPeer, AuthorityDenied, AuthorityService,
)


@pytest.fixture
def service(tmp_path):
    desktop, worker, other, untrusted = object(), object(), object(), object()
    peers = {desktop: AuthenticatedPeer('desktop', 'owner-a', 'desktop-a'),
             worker: AuthenticatedPeer('worker', 'owner-a', 'worker-a'),
             other: AuthenticatedPeer('desktop', 'owner-b', 'desktop-b')}
    current = [1000.0]
    def authenticate(connection):
        return peers.get(connection)
    store = AuthorityService(tmp_path / 'authority.sqlite', authenticate=authenticate, clock=lambda: current[0])
    return store, desktop, worker, other, untrusted, current


def _claimed(service):
    store, desktop, worker, *_ = service
    goal = 'Prepare the prospect follow-up list.'
    request_id = store.submit(desktop, goal=goal, idempotency_key='a' * 32)['request_id']
    claim = store.claim(worker, request_id=request_id, message_id='a' * 32, goal=goal, expected_actor='owner-a')
    return store, desktop, worker, claim


def test_identity_is_server_derived_and_exact_request_is_bound(service):
    store, desktop, worker, other, untrusted, _ = service
    with pytest.raises(AuthorityDenied):
        store.submit(untrusted, goal='Do work', idempotency_key='a' * 32)
    with pytest.raises(AuthorityDenied):
        store.submit(worker, goal='Do work', idempotency_key='a' * 32)
    request_id = store.submit(desktop, goal='Do work', idempotency_key='a' * 32)['request_id']
    assert store.submit(desktop, goal='Do work', idempotency_key='a' * 32)['request_id'] == request_id
    with pytest.raises(AuthorityDenied):
        store.submit(desktop, goal='Different work', idempotency_key='a' * 32)
    with pytest.raises(AuthorityDenied):
        store.claim(worker, request_id=request_id, message_id='a' * 32, goal='Different work', expected_actor='owner-a')
    with pytest.raises(AuthorityDenied):
        store.cancel(other, request_id=request_id)
    with pytest.raises(AuthorityDenied):
        store.claim(worker, request_id=request_id, message_id='b' * 32, goal='Do work', expected_actor='owner-a')
    with pytest.raises(AuthorityDenied):
        store.claim(worker, request_id=request_id, message_id='a' * 32, goal='Do work', expected_actor='other')
    assert store.inspect_request(desktop, request_id=request_id, message_id='a' * 32, goal='Do work') == {
        'actor': 'owner-a', 'status': 'pending', 'started_actions': 0}
    with pytest.raises(AuthorityDenied):
        store.inspect_request(desktop, request_id=request_id, message_id='b' * 32, goal='Do work')
    assert store.claim(worker, request_id=request_id, message_id='a' * 32, goal='Do work', expected_actor='owner-a')['request_id'] == request_id


def test_bash_capability_is_scoped_to_the_authenticated_owner(service):
    store, desktop, worker, other, untrusted, _ = service
    assert store.grant_bash(other, expected_generation=store.bash_capability(other)['generation'])['enabled'] is True
    assert store.bash_capability(desktop) == {'enabled': False, 'generation': 0}
    with pytest.raises(AuthorityDenied):
        store.grant_bash(worker, expected_generation=0)
    with pytest.raises(AuthorityDenied):
        store.grant_bash(untrusted, expected_generation=0)
    assert store.revoke_bash(other, expected_generation=store.bash_capability(other)['generation']) == {'enabled': False, 'generation': 2}


def test_pending_enable_cannot_overwrite_later_disable(service):
    store, desktop, *_ = service
    pending_generation = store.bash_capability(desktop)['generation']
    # Even an already-disabled state must invalidate a pending setup grant.
    assert store.revoke_bash(desktop, expected_generation=pending_generation) == {
        'enabled': False, 'generation': 1}
    with pytest.raises(AuthorityDenied):
        store.grant_bash(desktop, expected_generation=pending_generation)
    assert store.bash_capability(desktop) == {'enabled': False, 'generation': 1}
    assert store.grant_bash(desktop, expected_generation=1) == {'enabled': True, 'generation': 2}
    with pytest.raises(AuthorityDenied):
        store.revoke_bash(desktop, expected_generation=1)
    assert store.bash_capability(desktop) == {'enabled': True, 'generation': 2}
    for invalid in (-1, True, '2', 2**63):
        with pytest.raises(AuthorityDenied):
            store.grant_bash(desktop, expected_generation=invalid)


def test_worker_capability_discovery_requires_a_live_owner_claim(service):
    store, desktop, worker, claim = _claimed(service)
    assert store.check_bash_capability(worker, claim=claim) is False
    store.grant_bash(desktop, expected_generation=store.bash_capability(desktop)['generation'])
    assert store.check_bash_capability(worker, claim=claim) is True
    with pytest.raises(AuthorityDenied):
        store.check_bash_capability(desktop, claim=claim)
    store.cancel(desktop, request_id=claim['request_id'])
    with pytest.raises(AuthorityDenied):
        store.check_bash_capability(worker, claim=claim)


def test_multiple_actions_are_bound_individually_and_never_restarted(service):
    store, desktop, worker, claim = _claimed(service)
    store.grant_bash(desktop, expected_generation=store.bash_capability(desktop)['generation'])
    write = {'tool_id': 'native.bash', 'arguments': {'command': 'printf test > leads.csv'}}
    read = {'tool_id': 'native.bash', 'arguments': {'command': 'cat leads.csv'}}
    first = store.reserve_action(worker, claim=claim, action=write, idempotency_key='1' * 32)
    assert store.reserve_action(worker, claim=claim, action=write, idempotency_key='1' * 32) == first
    with pytest.raises(AuthorityDenied):
        store.reserve_action(worker, claim=claim, action=read, idempotency_key='1' * 32)
    with pytest.raises(AuthorityDenied):
        store.start_action(worker, claim=claim, action_id=first['action_id'], action=read)
    assert store.start_action(worker, claim=claim, action_id=first['action_id'], action=write)['effect'] == 'unverified'
    with pytest.raises(AuthorityDenied):
        store.start_action(worker, claim=claim, action_id=first['action_id'], action=write)
    assert store.record_process_return(worker, claim=claim, action_id=first['action_id'],
        process_result={'exit_code': 0}) == {'status': 'return_reported', 'effect': 'unverified'}
    assert store.reserve_action(worker, claim=claim, action=write, idempotency_key='1' * 32)['status'] == 'return_reported'
    second = store.reserve_action(worker, claim=claim, action=read, idempotency_key='2' * 32)
    assert second['action_id'] != first['action_id']
    assert store.start_action(worker, claim=claim, action_id=second['action_id'], action=read)['status'] == 'started'


def test_restart_retains_uncertain_dispatch_and_prevents_replay(service):
    store, _, worker, claim = _claimed(service)
    action = {'command': 'touch result'}
    action_id = store.reserve_action(worker, claim=claim, action=action, idempotency_key='1' * 32)['action_id']
    store.start_action(worker, claim=claim, action_id=action_id, action=action)
    restarted = AuthorityService(store.path, authenticate=store.authenticate, clock=store.clock)
    assert restarted.reserve_action(worker, claim=claim, action=action, idempotency_key='1' * 32) == {
        'action_id': action_id, 'status': 'started'}
    with pytest.raises(AuthorityDenied):
        restarted.start_action(worker, claim=claim, action_id=action_id, action=action)


def test_expired_or_cancelled_claim_cannot_dispatch(service):
    store, desktop, worker, claim = _claimed(service)
    action = {'command': 'touch test'}
    reserved = store.reserve_action(worker, claim=claim, action=action, idempotency_key='1' * 32)
    assert store.cancel(desktop, request_id=claim['request_id']) == {
        'status': 'cancelled', 'execution_may_have_started': False}
    with pytest.raises(AuthorityDenied):
        store.start_action(worker, claim=claim, action_id=reserved['action_id'], action=action)
    fresh = store.submit(desktop, goal='Later task', idempotency_key='b' * 32)['request_id']
    expired_claim = store.claim(worker, request_id=fresh, message_id='b' * 32, goal='Later task', expected_actor='owner-a', lease_seconds=1)
    service[-1][0] += 2
    with pytest.raises(AuthorityDenied):
        store.reserve_action(worker, claim=expired_claim, action=action, idempotency_key='2' * 32)
    with pytest.raises(AuthorityDenied):
        store.claim(worker, request_id=fresh, message_id='b' * 32, goal='Later task', expected_actor='owner-a')


def test_cancel_racing_dispatch_never_reports_not_started_after_start(service):
    store, desktop, worker, claim = _claimed(service)
    action = {'command': 'touch test'}
    action_id = store.reserve_action(worker, claim=claim, action=action, idempotency_key='1' * 32)['action_id']
    barrier = threading.Barrier(2)
    outcomes = {}
    def start():
        barrier.wait()
        try:
            outcomes['start'] = store.start_action(worker, claim=claim, action_id=action_id, action=action)
        except AuthorityDenied:
            outcomes['start'] = None
    def cancel():
        barrier.wait()
        outcomes['cancel'] = store.cancel(desktop, request_id=claim['request_id'])
    threads = [threading.Thread(target=start), threading.Thread(target=cancel)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert outcomes['cancel']['execution_may_have_started'] is (outcomes['start'] is not None)


def test_failed_second_lease_check_rolls_back_action_transition(tmp_path):
    # A long service transaction cannot commit a dispatch after its lease expires.
    desktop, worker = object(), object()
    peers = {desktop: AuthenticatedPeer('desktop', 'a', 'd'), worker: AuthenticatedPeer('worker', 'a', 'w')}
    ticks = iter([0.0, 0.0, 0.0, 0.0, 2.0])
    path = tmp_path / 'authority.sqlite'
    store = AuthorityService(path, authenticate=peers.get, clock=lambda: next(ticks))
    rid = store.submit(desktop, goal='Test', idempotency_key='a' * 32)['request_id']
    claim = store.claim(worker, request_id=rid, message_id='a' * 32, goal='Test', expected_actor='a', lease_seconds=1)
    action = {'command': 'touch test'}
    action_id = store.reserve_action(worker, claim=claim, action=action, idempotency_key='1' * 32)['action_id']
    with pytest.raises(AuthorityDenied):
        store.start_action(worker, claim=claim, action_id=action_id, action=action)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT state FROM actions WHERE id=?', (action_id,)).fetchone()[0] == 'reserved'
