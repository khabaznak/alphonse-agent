"""Synthetic fixture-only settlement, never a real provider or authorization."""
from datetime import timedelta
import sqlite3

import pytest

from test_single_loop_delivery_reservation import setup, reserve
from alphonse.agent_v2.evaluation.synthetic_delivery_settlement import settle_fixture_delivery
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded


def seed_fixture(f, *, grant=True, effect=True):
    _, _, questions, _, _, _, _, binding = f
    with questions._connect() as conn:
        conn.execute('''CREATE TABLE fixture_delivery_grants(actor TEXT,recipient TEXT,integration TEXT,
                     provider TEXT,channel TEXT,revision TEXT,active INTEGER)''')
        conn.execute('''CREATE TABLE fixture_delivery_effects(task_id TEXT,actor TEXT,action_id TEXT,
                     integration TEXT,provider TEXT,channel TEXT,recipient TEXT,message_digest TEXT,
                     idempotency_key TEXT,grant_revision TEXT,status TEXT)''')
        if grant:
            conn.execute('INSERT INTO fixture_delivery_grants VALUES (?,?,?,?,?,?,?)',
                         ('alice', binding.recipient_user_id, binding.integration_id, binding.provider_key,
                          binding.channel_target, binding.grant_revision, 1))
        if effect:
            # Independent synthetic provider observation; not an agent assertion.
            conn.execute('INSERT INTO fixture_delivery_effects VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                         ('task', 'alice', binding.action_id, binding.integration_id, binding.provider_key,
                          binding.channel_target, binding.recipient_user_id, binding.message_digest,
                          binding.idempotency_key, binding.grant_revision, 'delivered'))


def settle(f, version, **kwargs):
    _, _, _, store, claim, _, now, _ = f
    return settle_fixture_delivery(store, kwargs.pop('claim', claim), actor=kwargs.pop('actor', 'alice'),
        task_id='task', expected_version=version, clock=kwargs.pop('clock', lambda: now), **kwargs)


def test_verified_fixture_effect_settles_once_and_survives_restart(setup):
    _, _, questions, store, _, _, _, binding = setup
    pending = reserve(setup)
    seed_fixture(setup)
    settled = settle(setup, pending.version)
    assert settled.state.inflight == ''
    assert not settled.budget.usage()['in_flight']
    restored = ExperimentalCheckpointStore(questions).load('task', 'alice', expected_version=settled.version)
    assert restored.budget.to_dict()['reservations'][binding.action_id]['outcome'] == 'success'
    assert questions.load_task_checkpoint('task').metadata['fixture_delivery_settlement']['status'] == 'fixture_delivered'
    with pytest.raises(ValueError):
        settle(setup, pending.version)
    assert store.load('task', 'alice', expected_version=settled.version).state.inflight == ''


@pytest.mark.parametrize('mutation', ['missing_effect', 'failed', 'recipient', 'digest', 'provider', 'channel',
                                      'idempotency', 'grant', 'revoked', 'inactive', 'duplicate_effect', 'duplicate_grant'])
def test_unverified_or_unauthorized_fixture_keeps_pending_marker(setup, mutation):
    _, users, questions, store, _, _, _, binding = setup
    pending = reserve(setup)
    seed_fixture(setup, effect=mutation != 'missing_effect')
    if mutation == 'inactive':
        users.update_user('alice', is_active=False)
    else:
        changes = {'failed': ("UPDATE fixture_delivery_effects SET status='unknown'", ()),
                   'recipient': ("UPDATE fixture_delivery_effects SET recipient='bob'", ()),
                   'digest': ("UPDATE fixture_delivery_effects SET message_digest=?", ('b' * 64,)),
                   'provider': ("UPDATE fixture_delivery_effects SET provider='other'", ()),
                   'channel': ("UPDATE fixture_delivery_effects SET channel='other'", ()),
                   'idempotency': ("UPDATE fixture_delivery_effects SET idempotency_key='other'", ()),
                   'grant': ("UPDATE fixture_delivery_grants SET revision='new'", ()),
                   'revoked': ("UPDATE fixture_delivery_grants SET active=0", ()),
                   'duplicate_effect': ("INSERT INTO fixture_delivery_effects SELECT * FROM fixture_delivery_effects", ()),
                   'duplicate_grant': ("INSERT INTO fixture_delivery_grants SELECT * FROM fixture_delivery_grants", ())}
        if mutation in changes:
            with questions._connect() as conn:
                conn.execute(*changes[mutation])
    with pytest.raises(ClaimFenceUnavailable):
        settle(setup, pending.version)
    restored = store.load('task', 'alice', expected_version=pending.version)
    assert restored.state.inflight == binding.action_id
    assert 'fixture_delivery_settlement' not in questions.load_task_checkpoint('task').metadata


def test_cancelled_stale_and_wrong_actor_never_settle(setup):
    _, _, questions, store, _, _, _, binding = setup
    pending = reserve(setup)
    seed_fixture(setup)
    with pytest.raises(ValueError):
        settle(setup, pending.version + 1)
    with pytest.raises(ClaimFenceUnavailable):
        settle(setup, pending.version, actor='bob')
    assert questions.mark_task_checkpoint_terminal('task', status='cancelled')
    with pytest.raises(ValueError):
        settle(setup, pending.version)
    assert store.inspect_cancelled('task', 'alice', expected_version=pending.version)['inflight'] == binding.action_id


def test_reclaim_and_clock_expiry_rollback_settlement(setup):
    queue, _, questions, store, claim, _, now, binding = setup
    pending = reserve(setup)
    seed_fixture(setup)
    moments = iter((now, now + timedelta(days=1)))
    with pytest.raises(ClaimSuperseded):
        settle(setup, pending.version, clock=lambda: next(moments))
    assert store.load('task', 'alice', expected_version=pending.version).state.inflight == binding.action_id
    assert queue.reclaim_expired(now=now + timedelta(days=1)) == 1
    assert queue.claim_next().claim_receipt.claim_generation != claim.claim_generation
    with pytest.raises(ClaimSuperseded):
        settle(setup, pending.version)
    assert 'fixture_delivery_settlement' not in questions.load_task_checkpoint('task').metadata


def test_fixture_grant_revocation_cannot_race_locked_settlement(setup):
    _, _, questions, _, _, _, now, _ = setup
    pending = reserve(setup)
    seed_fixture(setup)
    calls = 0
    def clock():
        nonlocal calls
        calls += 1
        if calls == 2:
            with sqlite3.connect(questions.db_path, timeout=0) as other:
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    other.execute('UPDATE fixture_delivery_grants SET active=0')
        return now
    assert settle(setup, pending.version, clock=clock).state.inflight == ''
