"""Disposable stage → fixture dispatch → fixture settlement, not production delivery."""
from datetime import timedelta
import sqlite3

import pytest

from test_experimental_staged_outbox import setup, stage
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
from alphonse.agent_v2.evaluation.synthetic_delivery_settlement import settle_fixture_delivery
from alphonse.agent_v2.evaluation.synthetic_staged_dispatch import dispatch_fixture_staged


def seed_authority(f):
    path, _, _, _, _, _, _ = f
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE fixture_delivery_grants(actor TEXT,recipient TEXT,integration TEXT,
                     provider TEXT,channel TEXT,revision TEXT,active INTEGER)''')
        conn.execute('''CREATE TABLE fixture_delivery_effects(task_id TEXT,actor TEXT,action_id TEXT,
                     integration TEXT,provider TEXT,channel TEXT,recipient TEXT,message_digest TEXT,
                     idempotency_key TEXT,grant_revision TEXT,status TEXT)''')
        conn.execute('INSERT INTO fixture_delivery_grants VALUES (?,?,?,?,?,?,?)',
                     ('alice', 'alice', 'desktop', 'desktop', 'alice', 'grant', 1))


def dispatch(f, staged_id, claim=None, clock=None):
    _, _, questions, outbox, original_claim, version, now = f
    return dispatch_fixture_staged(outbox, ExperimentalCheckpointStore(questions), claim or original_claim,
        actor='alice', task_id='task', expected_version=version, staged_id=staged_id, clock=clock or (lambda: now))


def settle(f, claim=None, clock=None):
    _, _, questions, _, original_claim, version, now = f
    return settle_fixture_delivery(ExperimentalCheckpointStore(questions), claim or original_claim,
        actor='alice', task_id='task', expected_version=version, clock=clock or (lambda: now))


def test_staged_fixture_dispatch_and_authorized_settlement_survive_restart(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    dispatch(f, identity)
    # A lost dispatch response must not trigger a second fixture effect.
    with pytest.raises(ClaimFenceUnavailable):
        dispatch(f, identity)
    # Settlement uses a fresh checkpoint store, simulating a restarted worker;
    # it must independently read the already persisted fixture effect.
    settled = settle(f)
    restored = ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=settled.version)
    assert not restored.state.inflight
    assert restored.budget.to_dict()['reservations']['action']['outcome'] == 'success'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1
        assert conn.execute('SELECT status FROM v2_outbox WHERE outbox_message_id=?', (identity,)).fetchone()[0] == 'staged'


def test_wrong_staged_identity_never_dispatches(setup):
    f = setup
    seed_authority(f)
    stage(f)
    with pytest.raises(ClaimFenceUnavailable):
        dispatch(f, 'not-the-staged-message')
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0


def test_revoked_before_dispatch_produces_no_effect_or_settlement(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    with sqlite3.connect(f[0]) as conn:
        conn.execute('UPDATE fixture_delivery_grants SET active=0')
    with pytest.raises(ClaimFenceUnavailable):
        dispatch(f, identity)
    with pytest.raises(ClaimFenceUnavailable):
        settle(f)
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0


def test_revocation_after_effect_preserves_uncertain_marker(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    dispatch(f, identity)
    with sqlite3.connect(f[0]) as conn:
        conn.execute('UPDATE fixture_delivery_grants SET active=0')
    with pytest.raises(ClaimFenceUnavailable):
        settle(f)
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1


def test_reclaimed_claim_cannot_dispatch_or_settle_old_worker(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    assert f[1].reclaim_expired(now=f[6] + timedelta(days=1)) == 1
    assert f[1].claim_next().claim_receipt.claim_generation != f[4].claim_generation
    with pytest.raises(ClaimSuperseded):
        dispatch(f, identity)
    with pytest.raises(ClaimSuperseded):
        settle(f)
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0


def test_reclaim_after_effect_blocks_stale_settlement_without_replay(setup):
    f = setup
    seed_authority(f)
    identity = stage(f)
    dispatch(f, identity)
    assert f[1].reclaim_expired(now=f[6] + timedelta(days=1)) == 1
    new_claim = f[1].claim_next().claim_receipt
    assert new_claim.claim_generation != f[4].claim_generation
    with pytest.raises(ClaimSuperseded):
        settle(f)
    with pytest.raises(ClaimFenceUnavailable):
        dispatch(f, identity, claim=new_claim)
    assert ExperimentalCheckpointStore(f[2]).load('task', 'alice', expected_version=f[5]).state.inflight == 'action'
    with sqlite3.connect(f[0]) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1
