"""Synthetic staged dispatch: no network, real grants, or provider effects."""
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import sqlite3

import pytest

from test_experimental_staged_outbox import setup, stage
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
from alphonse.agent_v2.evaluation.synthetic_staged_dispatch import begin_fixture_dispatch, dispatch_fixture_staged


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


def begin(f, identity):
    _, _, questions, outbox, claim, version, now = f
    return begin_fixture_dispatch(outbox, ExperimentalCheckpointStore(questions), claim,
        actor='alice', task_id='task', expected_version=version, staged_id=identity, clock=lambda: now)


def test_crash_after_attempt_before_effect_never_automatically_retries(setup):
    path, _, questions, _, _, version, _ = setup
    identity = stage(setup)
    seed(setup)
    begin(setup, identity)  # A durable marker is the last step before the crash.
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT status FROM experimental_staged_dispatch_attempts').fetchone() == ('started',)
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError, match='requires_reconciliation'):
            conn.execute('DELETE FROM experimental_staged_dispatch_attempts')
    with pytest.raises(ClaimFenceUnavailable, match='already_observed'):
        dispatch(setup, identity)
    assert ExperimentalCheckpointStore(questions).load('task', 'alice', expected_version=version).state.inflight == 'action'


def test_clock_regression_between_attempt_and_effect_preserves_uncertainty(setup):
    path, _, questions, _, _, version, now = setup
    identity = stage(setup)
    seed(setup)
    moments = iter((now, now, now - timedelta(microseconds=1)))
    with pytest.raises(ClaimFenceUnavailable, match='clock_reversed'):
        dispatch(setup, identity, clock=lambda: next(moments))
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT status FROM experimental_staged_dispatch_attempts').fetchone() == ('started',)
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0
    assert ExperimentalCheckpointStore(questions).load('task', 'alice', expected_version=version).state.inflight == 'action'
    with pytest.raises(ClaimFenceUnavailable, match='already_observed'):
        dispatch(setup, identity)


def test_dispatch_rejects_checkpoint_adapter_from_another_database(setup, tmp_path):
    path, _, _, outbox, claim, version, now = setup
    identity = stage(setup)
    seed(setup)
    unrelated = ExperimentalCheckpointStore(SQLiteQuestionStore(tmp_path / 'other.sqlite'))
    with pytest.raises(ClaimFenceUnavailable, match='fixture_dispatch_unavailable'):
        dispatch_fixture_staged(outbox, unrelated, claim, actor='alice', task_id='task',
                                expected_version=version, staged_id=identity, clock=lambda: now)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM experimental_staged_dispatch_attempts').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0


def test_dispatch_attempt_guard_tampering_fails_before_effect(setup):
    path = setup[0]
    identity = stage(setup)
    seed(setup)
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TRIGGER experimental_dispatch_attempt_delete')
    with pytest.raises(ClaimFenceUnavailable, match='attempt_schema_unavailable'):
        dispatch(setup, identity)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM experimental_staged_dispatch_attempts').fetchone()[0] == 0


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
