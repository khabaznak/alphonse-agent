"""Synthetic staged dispatch: no network, real grants, or provider effects."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import sqlite3

import pytest

from test_experimental_staged_outbox import setup, stage
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
from alphonse.agent_v2.evaluation.synthetic_staged_dispatch import dispatch_fixture_staged


def seed(f, *, active=1):
    _, _, questions, _, _, _, _ = f
    with questions._connect() as conn:
        conn.execute('''CREATE TABLE fixture_delivery_grants(actor TEXT,recipient TEXT,integration TEXT,
                     provider TEXT,channel TEXT,grant_id TEXT,revision INTEGER,connection_revision INTEGER,active INTEGER)''')
        conn.execute('''CREATE TABLE fixture_delivery_effects(task_id TEXT,actor TEXT,action_id TEXT,
                     integration TEXT,provider TEXT,channel TEXT,recipient TEXT,message_digest TEXT,
                     idempotency_key TEXT,grant_id TEXT,grant_revision INTEGER,connection_revision INTEGER,status TEXT)''')
        conn.execute('INSERT INTO fixture_delivery_grants VALUES (?,?,?,?,?,?,?,?,?)',
                     ('alice', 'alice', 'desktop', 'desktop', 'alice', 'grant', 1, 1, active))


def dispatch(f, identity, **kwargs):
    _, _, questions, outbox, claim, version, now = f
    return dispatch_fixture_staged(outbox, ExperimentalCheckpointStore(questions),
        kwargs.pop('claim', claim), actor=kwargs.pop('actor', 'alice'), task_id='task',
        expected_version=kwargs.pop('version', version), staged_id=identity,
        clock=kwargs.pop('clock', lambda: now), **kwargs)


def test_dispatch_records_one_fixture_effect_without_releasing_staged_row(setup):
    path, _, _, outbox, _, _, _ = setup
    identity = stage(setup)
    seed(setup)
    dispatch(setup, identity)
    with pytest.raises(ClaimFenceUnavailable, match='already_observed'):
        dispatch(setup, identity)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT task_id,action_id,status FROM fixture_delivery_effects').fetchall() == [('task', 'action', 'delivered')]
        assert conn.execute('SELECT status FROM v2_outbox WHERE outbox_message_id=?', (identity,)).fetchone() == ('staged',)
    assert outbox.claim_next() is None


@pytest.mark.parametrize('failure', ['revoked', 'wrong_grant', 'inactive', 'wrong_id', 'stale_version', 'expired'])
def test_failed_authority_or_ownership_never_records_effect(setup, failure):
    path, _, _, _, _, version, now = setup
    identity = stage(setup)
    seed(setup, active=0 if failure == 'revoked' else 1)
    with sqlite3.connect(path) as conn:
        if failure == 'wrong_grant':
            conn.execute("UPDATE fixture_delivery_grants SET revision=2")
        if failure == 'inactive':
            conn.execute("UPDATE v2_users SET is_active=0 WHERE user_id='alice'")
    options = {}
    if failure == 'stale_version': options['version'] = version + 1
    if failure == 'expired':
        moments = iter((now, now + timedelta(days=1)))
        options['clock'] = lambda: next(moments)
    with pytest.raises((ClaimFenceUnavailable, ClaimSuperseded, ValueError)):
        dispatch(setup, 'wrong-id' if failure == 'wrong_id' else identity, **options)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0


def test_simultaneous_dispatches_produce_only_one_fixture_effect(setup):
    path, _, _, _, _, _, _ = setup
    identity = stage(setup)
    seed(setup)
    def attempt():
        try:
            dispatch(setup, identity)
            return 'recorded'
        except ClaimFenceUnavailable:
            return 'blocked'
    with ThreadPoolExecutor(max_workers=2) as pool:
        calls = [pool.submit(attempt) for _ in range(2)]
        assert sorted(call.result() for call in calls) == ['blocked', 'recorded']
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1
