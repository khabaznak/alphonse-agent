"""Expired-claim recovery in disposable fixtures only, never household data."""
from datetime import timedelta
import sqlite3
import subprocess
import sys

import pytest

from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.experimental_staged_outbox import _GUARDS
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable
from alphonse.agent_v2.evaluation.synthetic_delivery_settlement import recover_expired_fixture_effect
from alphonse.agent_v2.evaluation.synthetic_staged_dispatch import dispatch_fixture_staged
from test_experimental_staged_outbox import setup, stage
from test_synthetic_delivery_lifecycle import seed_authority
from unsafe_reclaim_fixture import simulate_old_writer_requeue


def ready(f):
    path, _, questions, outbox, claim, version, now = f
    seed_authority(f)
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE fixture_recovery_grants(actor TEXT, task_id TEXT, active INTEGER)')
        conn.execute('INSERT INTO fixture_recovery_grants VALUES (?,?,?)', ('alice', 'task', 1))
    identity = stage(f)
    dispatch_fixture_staged(outbox, ExperimentalCheckpointStore(questions), claim, actor='alice', task_id='task',
                            expected_version=version, staged_id=identity, clock=lambda: now)
    return identity


def recover(f, *, actor='alice', version=None, clock=None):
    _, _, questions, _, _, original_version, now = f
    return recover_expired_fixture_effect(ExperimentalCheckpointStore(questions), actor=actor, task_id='task',
        expected_version=original_version if version is None else version,
        clock=clock or (lambda: now + timedelta(days=1)))


def test_subprocess_reconciles_expired_effect_without_receipt_or_queue_mutation(setup):
    f = setup
    path, queue, questions, _, _, version, now = f
    ready(f)
    with sqlite3.connect(path) as conn:
        before_queue = conn.execute('SELECT * FROM v2_inbound_messages').fetchall()
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1
        assert conn.execute('SELECT count(*) FROM experimental_staged_dispatch_attempts').fetchone()[0] == 1
    # This is a separate interpreter: only a DB path, task identity, actor and
    # fixture clock cross the boundary, never the original receipt/generation.
    script = '''from datetime import datetime
import sys
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.evaluation.synthetic_delivery_settlement import recover_expired_fixture_effect
s = ExperimentalCheckpointStore(SQLiteQuestionStore(sys.argv[1]))
r = recover_expired_fixture_effect(s, actor='alice', task_id='task', expected_version=int(sys.argv[2]),
                                  clock=lambda: datetime.fromisoformat(sys.argv[3]))
print(r.version)
'''
    child = subprocess.run([sys.executable, '-c', script, path, str(version), (now + timedelta(days=1)).isoformat()],
                           capture_output=True, text=True, check=True)
    assert child.stdout.strip() == str(version + 1)
    assert child.stderr == ''
    restored = ExperimentalCheckpointStore(questions).load('task', 'alice', expected_version=version + 1)
    assert not restored.state.inflight
    assert restored.budget.to_dict()['reservations']['action']['outcome'] == 'success'
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT * FROM v2_inbound_messages').fetchall() == before_queue
        assert conn.execute('SELECT count(*) FROM fixture_delivery_effects').fetchone()[0] == 1
        assert conn.execute('SELECT count(*) FROM experimental_staged_dispatch_attempts').fetchone()[0] == 1
    assert queue.status_for('message') == 'processing'


@pytest.mark.parametrize('failure', ['active_claim', 'missing_effect', 'duplicate_effect', 'revoked_recovery',
    'revoked_delivery', 'cancelled', 'wrong_actor', 'stale_version', 'replacement_claim', 'clock_reversed',
    'clock_unavailable'])
def test_unknown_or_unauthorized_recovery_never_clears_marker(setup, failure):
    f = setup
    path, queue, questions, _, old_claim, version, now = f
    ready(f)
    if failure == 'missing_effect':
        with sqlite3.connect(path) as conn:
            conn.execute('DELETE FROM fixture_delivery_effects')
    elif failure == 'duplicate_effect':
        with sqlite3.connect(path) as conn:
            conn.execute('INSERT INTO fixture_delivery_effects SELECT * FROM fixture_delivery_effects')
    elif failure in {'revoked_recovery', 'revoked_delivery'}:
        with sqlite3.connect(path) as conn:
            table = 'fixture_recovery_grants' if failure == 'revoked_recovery' else 'fixture_delivery_grants'
            conn.execute(f'UPDATE {table} SET active=0')
    elif failure == 'cancelled':
        assert questions.mark_task_checkpoint_terminal('task', status='cancelled')
    elif failure == 'replacement_claim':
        simulate_old_writer_requeue(queue, 'message')
        assert queue.claim_next().claim_receipt.claim_generation != old_claim.claim_generation
    kwargs = {}
    if failure == 'active_claim':
        kwargs['clock'] = lambda: now
    elif failure == 'wrong_actor':
        kwargs['actor'] = 'bob'
    elif failure == 'stale_version':
        kwargs['version'] = version + 1
    elif failure == 'clock_reversed':
        moments = iter((now + timedelta(days=2), now + timedelta(days=1)))
        kwargs['clock'] = lambda: next(moments)
    elif failure == 'clock_unavailable':
        def unavailable():
            raise RuntimeError('clock unavailable')
        kwargs['clock'] = unavailable
    with pytest.raises(ClaimFenceUnavailable):
        recover(f, **kwargs)
    if failure == 'cancelled':
        pending = ExperimentalCheckpointStore(questions).inspect_cancelled('task', 'alice', expected_version=version)
        assert pending['inflight'] == 'action'
    else:
        assert ExperimentalCheckpointStore(questions).load('task', 'alice', expected_version=version).state.inflight == 'action'


@pytest.mark.parametrize('tamper', ['grant', 'queue', 'settlement'])
def test_checkpoint_trigger_cannot_change_grant_queue_or_evidence(setup, tamper):
    f = setup
    path, _, questions, _, _, version, _ = f
    ready(f)
    statements = {
        'grant': 'UPDATE fixture_recovery_grants SET active=0',
        'queue': "UPDATE v2_inbound_messages SET lease_owner='tampered' WHERE message_id='message'",
        'settlement': "UPDATE v2_task_checkpoints SET task_state_json=json_set(NEW.task_state_json, "
                      "'$.metadata.fixture_delivery_settlement.status', 'forged') WHERE task_id=NEW.task_id",
    }
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TRIGGER fixture_tamper AFTER UPDATE ON v2_task_checkpoints BEGIN ' +
                     statements[tamper] + '; END')
        before = conn.execute('SELECT * FROM v2_inbound_messages').fetchall()
    with pytest.raises(ClaimFenceUnavailable):
        recover(f)
    assert ExperimentalCheckpointStore(questions).load('task', 'alice', expected_version=version).state.inflight == 'action'
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT active FROM fixture_recovery_grants').fetchone()[0] == 1
        assert conn.execute('SELECT * FROM v2_inbound_messages').fetchall() == before


@pytest.mark.parametrize('mutation', ['missing_guard', 'pending_guard_restored', 'content_guard_restored',
                                     'metadata_guard_restored', 'correlation_guard_restored',
                                     'retry_guard_restored', 'timestamp_guard_restored', 'identity_guard_restored'])
def test_expired_recovery_rejects_dispatchable_or_tampered_staging_even_if_guard_restored(setup, mutation):
    f = setup
    path, _, questions, outbox, _, version, _ = f
    identity = ready(f)
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TRIGGER experimental_staged_outbox_update')
        if mutation == 'pending_guard_restored':
            conn.execute("UPDATE v2_outbox SET status='pending' WHERE outbox_message_id=?", (identity,))
        elif mutation == 'content_guard_restored':
            conn.execute("UPDATE v2_outbox SET message='altered' WHERE outbox_message_id=?", (identity,))
        elif mutation == 'metadata_guard_restored':
            conn.execute("UPDATE v2_outbox SET metadata_json='{}' WHERE outbox_message_id=?", (identity,))
        elif mutation == 'correlation_guard_restored':
            conn.execute("UPDATE v2_outbox SET correlation_id='other' WHERE outbox_message_id=?", (identity,))
        elif mutation == 'retry_guard_restored':
            conn.execute("UPDATE v2_outbox SET max_attempts=100 WHERE outbox_message_id=?", (identity,))
        elif mutation == 'timestamp_guard_restored':
            conn.execute("UPDATE v2_outbox SET updated_at='other' WHERE outbox_message_id=?", (identity,))
        elif mutation == 'identity_guard_restored':
            conn.execute("UPDATE v2_outbox SET outbox_message_id='other' WHERE outbox_message_id=?", (identity,))
        if mutation != 'missing_guard':
            conn.execute(_GUARDS['experimental_staged_outbox_update'])
    with pytest.raises(ClaimFenceUnavailable):
        recover(f)
    assert ExperimentalCheckpointStore(questions).load('task', 'alice', expected_version=version).state.inflight == 'action'
    if mutation == 'pending_guard_restored':
        # This corrupted fixture can still dispatch via the old API: refusal to
        # settle does not repair or make an externally modified outbox safe.
        assert outbox.claim_next().outbox_message_id == identity


def test_recovery_rolls_back_trigger_changing_outbox_during_settlement(setup):
    f = setup
    path, _, questions, outbox, _, version, _ = f
    ready(f)
    address = ChannelAddress('desktop', 'desktop', 'alice', alphonse_user_id='alice')
    other = outbox.enqueue(address=address, message='other fixture message', task_id='other')
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TRIGGER fixture_mutate_outbox AFTER UPDATE ON v2_task_checkpoints BEGIN "
                     "UPDATE v2_outbox SET last_error='changed' WHERE status='pending'; END")
    with pytest.raises(ClaimFenceUnavailable):
        recover(f)
    assert ExperimentalCheckpointStore(questions).load('task', 'alice', expected_version=version).state.inflight == 'action'
    assert outbox.get(other.outbox_message_id).last_error == ''
