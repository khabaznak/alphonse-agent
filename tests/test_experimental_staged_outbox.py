"""Disposable staging only; no production dispatch or recovery is enabled."""
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3
import subprocess
import sys

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.single_loop_delivery_reservation import DeliveryBinding, reserve_delivery
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, TaskBudget
from alphonse.agent_v2.core.io.experimental_staged_outbox import (
    install_experimental_staging, rollback_empty_experimental_staging, stage_experimental_response,
    restage_experimental_response_after_restart,
)
from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    users = V2UserStore(path)
    users.set_users_root(tmp_path / 'users')
    users.create_user(user_id='alice', display_name='Alice')
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    questions = SQLiteQuestionStore(path)
    outbox = SQLiteOutboundStore(path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(now, 'synthetic', 'alice'), message_id='message')
    claim = queue.claim_next().claim_receipt
    task = TaskState(task_id='task', message_id='message', user='alice', goal='synthetic',
                     intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    version = questions.save_task_checkpoint(task, status='running')
    checkpoints = ExperimentalCheckpointStore(questions)
    total = dict(actions=2, model_calls=2, input_tokens=100, output_tokens=100)
    final = dict(actions=0, model_calls=0, input_tokens=0, output_tokens=0)
    initial = checkpoints.initialize('task', 'alice', expected_version=version,
        state=CandidateState('task', 'alice', 'goal'),
        budget=TaskBudget(BudgetLimits(total, final, 10000, 0), started_at_ms=1000))
    message = 'synthetic private'
    binding = DeliveryBinding('action', 'desktop', 'desktop', 'alice', 'alice',
                              hashlib.sha256(message.encode('utf-8')).hexdigest(), 'key', 'grant', 1, 1)
    reserved = reserve_delivery(checkpoints, claim, task_id='task', actor='alice',
        expected_version=initial.version, binding=binding, message=message, now_ms=1100, clock=lambda: now)
    install_experimental_staging(outbox)
    return path, queue, questions, outbox, claim, reserved.version, now


def stage(f, **changes):
    path, queue, questions, outbox, claim, version, now = f
    args = dict(task_id='task', actor='alice', expected_version=version, integration_id='desktop',
                provider_key='desktop', channel_target='alice', message='synthetic private',
                idempotency_key='key', clock=lambda: now, checkpoints=ExperimentalCheckpointStore(questions))
    args.update(changes)
    return stage_experimental_response(outbox, claim, **args)


def test_staged_row_is_not_dispatchable_or_legacy_settleable(setup):
    path, _, _, outbox, _, _, _ = setup
    identity = stage(setup)
    assert outbox.claim_next() is None
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT status FROM v2_outbox WHERE outbox_message_id=?', (identity,)).fetchone()[0] == 'staged'
    with pytest.raises(sqlite3.IntegrityError, match='requires_reconciliation'):
        outbox.mark_delivered(identity)
    with sqlite3.connect(path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match='requires_reconciliation'):
            conn.execute('DELETE FROM v2_outbox WHERE outbox_message_id=?', (identity,))
        with pytest.raises(sqlite3.IntegrityError, match='requires_reconciliation'):
            conn.execute("INSERT OR REPLACE INTO v2_outbox SELECT * FROM v2_outbox WHERE outbox_message_id=?", (identity,))
    # Existing binary refuses the new schema instead of migrating a staged row.
    with pytest.raises(ValueError, match='outbox_status_unsupported|outbox_status_schema_unsupported'):
        SQLiteOutboundStore(path)


def test_empty_staging_rollback_restores_restart_without_changing_existing_rows(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    outbox = SQLiteOutboundStore(path)
    address = ChannelAddress('desktop', 'desktop', 'alice', alphonse_user_id='alice')
    message = outbox.enqueue(address=address, message='synthetic', task_id='task')
    with sqlite3.connect(path) as conn:
        before = conn.execute('SELECT * FROM v2_outbox').fetchall()
        indexes = conn.execute("SELECT name,sql FROM sqlite_master WHERE type='index' AND tbl_name='v2_outbox' ORDER BY name").fetchall()
    install_experimental_staging(outbox)
    # Reopen by pathname, as after a process restart: the ordinary outbox
    # constructor correctly refuses the experimental schema.
    rollback_empty_experimental_staging(path)
    restored = SQLiteOutboundStore(path)
    assert restored.get(message.outbox_message_id).message == 'synthetic'
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT * FROM v2_outbox').fetchall() == before
        assert conn.execute("SELECT name,sql FROM sqlite_master WHERE type='index' AND tbl_name='v2_outbox' ORDER BY name").fetchall() == indexes
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE type='trigger' AND name LIKE 'experimental_staged_outbox_%'").fetchone()[0] == 0


def test_rollback_refuses_staged_rows_and_preserves_guards_and_payload(setup):
    path, _, _, _, _, _, _ = setup
    identity = stage(setup)
    with sqlite3.connect(path) as conn:
        before = conn.execute('SELECT * FROM v2_outbox').fetchall()
        schema = conn.execute("SELECT type,name,sql FROM sqlite_master WHERE name LIKE 'experimental_staged_outbox_%' ORDER BY name").fetchall()
    with pytest.raises(ClaimFenceUnavailable, match='reconciliation_required'):
        rollback_empty_experimental_staging(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT * FROM v2_outbox').fetchall() == before
        assert conn.execute("SELECT type,name,sql FROM sqlite_master WHERE name LIKE 'experimental_staged_outbox_%' ORDER BY name").fetchall() == schema
        assert conn.execute('SELECT status FROM v2_outbox WHERE outbox_message_id=?', (identity,)).fetchone()[0] == 'staged'


def test_rollback_with_foreign_view_rolls_back_trigger_removal(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    outbox = SQLiteOutboundStore(path)
    install_experimental_staging(outbox)
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE VIEW household_view AS SELECT * FROM v2_outbox')
        before = conn.execute("SELECT type,name,sql FROM sqlite_master WHERE name LIKE 'experimental_staged_outbox_%' ORDER BY name").fetchall()
    with pytest.raises(ValueError, match='outbox_rebuild_schema_unsupported'):
        rollback_empty_experimental_staging(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT type,name,sql FROM sqlite_master WHERE name LIKE 'experimental_staged_outbox_%' ORDER BY name").fetchall() == before
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='household_view'").fetchone()[0] == 1


def test_rollback_holds_writer_lock_and_restores_guards_on_failure(tmp_path, monkeypatch):
    import alphonse.agent_v2.core.io.experimental_staged_outbox as staged_module
    path = tmp_path / 'runtime.sqlite'
    outbox = SQLiteOutboundStore(path)
    install_experimental_staging(outbox)
    with sqlite3.connect(path) as conn:
        before = conn.execute("SELECT type,name,sql FROM sqlite_master WHERE name LIKE 'experimental_staged_outbox_%' ORDER BY name").fetchall()
    def fail_under_lock(conn):
        with sqlite3.connect(path, timeout=0) as writer:
            with pytest.raises(sqlite3.OperationalError, match='locked'):
                writer.execute("DELETE FROM v2_outbox")
        raise RuntimeError('synthetic failure')
    monkeypatch.setattr(staged_module, '_validate_outbox_rebuild', fail_under_lock)
    with pytest.raises(RuntimeError, match='synthetic failure'):
        rollback_empty_experimental_staging(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT type,name,sql FROM sqlite_master WHERE name LIKE 'experimental_staged_outbox_%' ORDER BY name").fetchall() == before


def test_rollback_rejects_unknown_status_without_discarding_row(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    outbox = SQLiteOutboundStore(path)
    address = ChannelAddress('desktop', 'desktop', 'alice', alphonse_user_id='alice')
    item = outbox.enqueue(address=address, message='synthetic')
    install_experimental_staging(outbox)
    with sqlite3.connect(path) as conn:
        conn.execute('PRAGMA ignore_check_constraints=ON')
        conn.execute("UPDATE v2_outbox SET status='unknown' WHERE outbox_message_id=?", (item.outbox_message_id,))
    with pytest.raises(ClaimFenceUnavailable, match='schema_unavailable'):
        rollback_empty_experimental_staging(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT status FROM v2_outbox WHERE outbox_message_id=?', (item.outbox_message_id,)).fetchone()[0] == 'unknown'
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE type='trigger' AND name LIKE 'experimental_staged_outbox_%'").fetchone()[0] == 3


def test_restage_after_lost_commit_response_returns_same_inert_record(setup):
    path, _, _, outbox, _, _, _ = setup
    first = stage(setup)
    assert stage(setup) == first
    with sqlite3.connect(path) as conn:
        rows = conn.execute('SELECT outbox_message_id,status FROM v2_outbox').fetchall()
    assert rows == [(first, 'staged')]
    assert outbox.claim_next() is None


def test_explicit_restart_restages_without_opening_normal_outbox(setup):
    path, _, _, outbox, claim, version, now = setup
    identity = stage(setup)
    with pytest.raises(ValueError, match='outbox_status_unsupported|outbox_status_schema_unsupported'):
        SQLiteOutboundStore(path)
    # A fresh interpreter has no retained pre-install outbox object. The token
    # and content below are synthetic fixture data, never household values.
    code = '''
import sys
from datetime import datetime
from alphonse.agent_v2.core.messages.claim_fence import ClaimReceipt
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.io.experimental_staged_outbox import restage_experimental_response_after_restart
path, message_id, user, owner, generation, version, now = sys.argv[1:]
claim = ClaimReceipt(message_id, user, owner, generation)
print(restage_experimental_response_after_restart(path, claim,
    checkpoints=ExperimentalCheckpointStore(SQLiteQuestionStore(path)),
    task_id='task', actor='alice', expected_version=int(version), integration_id='desktop',
    provider_key='desktop', channel_target='alice', message='synthetic private',
    idempotency_key='key', clock=lambda: datetime.fromisoformat(now)))
'''
    result = subprocess.run([sys.executable, '-c', code, str(path), claim.message_id, claim.user_id,
                             claim.lease_owner, claim.claim_generation, str(version), now.isoformat()],
                            capture_output=True, text=True, check=True, timeout=15)
    assert result.stdout.strip() == identity
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT status, count(*) FROM v2_outbox').fetchone() == ('staged', 1)
    assert outbox.claim_next() is None


def test_restart_cannot_create_a_missing_staged_row(setup):
    path, _, questions, _, claim, version, now = setup
    with pytest.raises(ClaimFenceUnavailable, match='reconciliation_required'):
        restage_experimental_response_after_restart(path, claim,
            checkpoints=ExperimentalCheckpointStore(questions), task_id='task', actor='alice',
            expected_version=version, integration_id='desktop', provider_key='desktop',
            channel_target='alice', message='synthetic private', idempotency_key='key', clock=lambda: now)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM v2_outbox').fetchone()[0] == 0
    # The normal first-staging path remains available under its own checks.
    assert stage(setup)


@pytest.mark.parametrize('restart', [False, True])
def test_staging_rejects_checkpoint_store_for_another_database(setup, tmp_path, restart):
    path, _, _, outbox, claim, version, now = setup
    if restart:
        identity = stage(setup)
    other = ExperimentalCheckpointStore(SQLiteQuestionStore(tmp_path / 'different.sqlite'))
    common = dict(checkpoints=other, task_id='task', actor='alice', expected_version=version,
                  integration_id='desktop', provider_key='desktop', channel_target='alice',
                  message='synthetic private', idempotency_key='key', clock=lambda: now)
    with pytest.raises(ClaimFenceUnavailable, match='checkpoint_store_mismatch'):
        if restart:
            restage_experimental_response_after_restart(path, claim, **common)
        else:
            stage_experimental_response(outbox, claim, **common)
    with sqlite3.connect(path) as conn:
        rows = conn.execute('SELECT outbox_message_id,status FROM v2_outbox').fetchall()
    assert rows == ([(identity, 'staged')] if restart else [])


def test_staging_accepts_checkpoint_store_alias_to_same_database(setup, tmp_path):
    path, _, _, _, _, _, _ = setup
    alias = tmp_path / 'same-database.sqlite'
    alias.symlink_to(path)
    store = ExperimentalCheckpointStore(SQLiteQuestionStore(alias))
    identity = stage(setup, checkpoints=store)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT outbox_message_id,status FROM v2_outbox').fetchall() == [(identity, 'staged')]


@pytest.mark.parametrize('failure', ['expired', 'inactive', 'guard', 'binding', 'actor', 'version', 'cancelled'])
def test_restart_restaging_fails_closed_and_preserves_staged_row(setup, failure):
    path, _, questions, _, claim, version, now = setup
    identity = stage(setup)
    with sqlite3.connect(path) as conn:
        if failure == 'inactive':
            conn.execute("UPDATE v2_users SET is_active=0 WHERE user_id='alice'")
        if failure == 'guard':
            conn.execute('DROP TRIGGER experimental_staged_outbox_update')
        if failure == 'cancelled':
            conn.execute("UPDATE v2_task_checkpoints SET status='cancelled' WHERE task_id='task'")
        before = conn.execute('SELECT * FROM v2_outbox WHERE outbox_message_id=?', (identity,)).fetchone()
    moments = iter((now, now + timedelta(days=1)))
    kwargs = dict(task_id='task', actor='alice', expected_version=version, integration_id='desktop',
        provider_key='desktop', channel_target='alice', message='synthetic private', idempotency_key='key',
        clock=(lambda: next(moments)) if failure == 'expired' else (lambda: now),
        checkpoints=ExperimentalCheckpointStore(questions))
    if failure == 'binding':
        kwargs['message'] = 'changed'
    if failure == 'actor':
        kwargs['actor'] = 'bob'
    if failure == 'version':
        kwargs['expected_version'] += 1
    with pytest.raises((ClaimFenceUnavailable, ClaimSuperseded)):
        restage_experimental_response_after_restart(path, claim, **kwargs)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT * FROM v2_outbox WHERE outbox_message_id=?', (identity,)).fetchone() == before


def test_restart_path_must_already_exist_and_never_creates_database(setup, tmp_path):
    _, _, questions, _, claim, version, now = setup
    missing = tmp_path / 'missing.sqlite'
    with pytest.raises(FileNotFoundError):
        restage_experimental_response_after_restart(missing, claim,
            checkpoints=ExperimentalCheckpointStore(questions), task_id='task', actor='alice',
            expected_version=version, integration_id='desktop', provider_key='desktop',
            channel_target='alice', message='synthetic private', idempotency_key='key', clock=lambda: now)
    assert not missing.exists()


def test_concurrent_staging_produces_one_inert_record(setup):
    path, _, _, _, _, _, _ = setup
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(stage, setup)
        second = pool.submit(stage, setup)
        assert first.result() == second.result()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT count(*) FROM v2_outbox WHERE status='staged'").fetchone()[0] == 1


def test_restage_does_not_adopt_changed_record_even_if_guards_are_restored(setup):
    path, _, _, _, _, _, _ = setup
    identity = stage(setup)
    with sqlite3.connect(path) as conn:
        trigger = conn.execute("SELECT sql FROM sqlite_master WHERE name='experimental_staged_outbox_update'").fetchone()[0]
        conn.execute('DROP TRIGGER experimental_staged_outbox_update')
        conn.execute("UPDATE v2_outbox SET message='changed' WHERE outbox_message_id=?", (identity,))
        conn.execute(trigger)
    with pytest.raises(ValueError, match='idempotency_conflict'):
        stage(setup)


def test_restage_requires_live_claim_even_when_inert_record_exists(setup):
    _, _, _, _, _, _, now = setup
    stage(setup)
    moments = iter((now, now + timedelta(days=1)))
    with pytest.raises(ClaimSuperseded):
        stage(setup, clock=lambda: next(moments))


def test_superseded_checkpoint_and_lost_claim_never_stage(setup):
    path, _, questions, _, _, version, now = setup
    with questions._connect() as conn:
        conn.execute("UPDATE v2_task_checkpoints SET version=version+1 WHERE task_id='task'")
    with pytest.raises(ClaimFenceUnavailable):
        stage(setup)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM v2_outbox').fetchone()[0] == 0
    moments = iter((now, now + timedelta(days=1)))
    with pytest.raises(ClaimSuperseded):
        stage(setup, expected_version=version+1, clock=lambda: next(moments))
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM v2_outbox').fetchone()[0] == 0


def test_missing_guard_rejects_staging(setup):
    path, _, _, _, _, _, _ = setup
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TRIGGER experimental_staged_outbox_update')
    with pytest.raises(ClaimFenceUnavailable, match='guards_unavailable'):
        stage(setup)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM v2_outbox').fetchone()[0] == 0


def test_guard_with_wrong_case_sensitive_status_is_rejected(setup):
    path, _, _, _, _, _, _ = setup
    with sqlite3.connect(path) as conn:
        sql = conn.execute("SELECT sql FROM sqlite_master WHERE name='experimental_staged_outbox_update'").fetchone()[0]
        conn.execute('DROP TRIGGER experimental_staged_outbox_update')
        conn.execute(sql.replace("'staged'", "'STAGED'"))
    with pytest.raises(ClaimFenceUnavailable, match='guards_unavailable'):
        stage(setup)


def test_install_rejects_dependent_trigger_without_mutating_schema(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    outbox = SQLiteOutboundStore(path)
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TRIGGER household_trigger AFTER INSERT ON v2_outbox BEGIN SELECT 1; END")
        before = conn.execute("SELECT sql FROM sqlite_master WHERE name='v2_outbox'").fetchone()[0]
    with pytest.raises(ValueError, match='outbox_rebuild_schema_unsupported'):
        install_experimental_staging(outbox)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT sql FROM sqlite_master WHERE name='v2_outbox'").fetchone()[0] == before
        assert conn.execute("SELECT count(*) FROM sqlite_master WHERE name='household_trigger'").fetchone()[0] == 1


def test_added_insert_trigger_is_not_allowed_to_modify_other_state(setup):
    path, _, _, _, _, _, _ = setup
    with sqlite3.connect(path) as conn:
        conn.execute('CREATE TABLE unrelated(value TEXT)')
        conn.execute("CREATE TRIGGER unexpected AFTER INSERT ON v2_outbox BEGIN INSERT INTO unrelated VALUES ('effect'); END")
    with pytest.raises(ClaimFenceUnavailable, match='guards_unavailable'):
        stage(setup)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM unrelated').fetchone()[0] == 0
        assert conn.execute('SELECT count(*) FROM v2_outbox').fetchone()[0] == 0


@pytest.mark.parametrize('change', [
    {'integration_id': 'other'}, {'provider_key': 'other'}, {'channel_target': 'other'},
    {'message': 'synthetic private '}, {'idempotency_key': 'other'},
])
def test_staging_rejects_values_not_bound_to_pending_action(setup, change):
    path, _, _, _, _, _, _ = setup
    with pytest.raises(ClaimFenceUnavailable, match='binding_unavailable'):
        stage(setup, **change)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM v2_outbox').fetchone()[0] == 0


@pytest.mark.parametrize('part', ['recipient', 'grant_shape', 'pending_marker', 'schema', 'missing_action'])
def test_staging_rejects_tampered_reservation(setup, part):
    path, _, questions, _, _, _, _ = setup
    with questions._connect() as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        action = raw['metadata']['single_loop_delivery_action']
        if part == 'recipient': action['recipient_user_id'] = 'bob'
        if part == 'grant_shape': action['grant_revision'] = ''
        if part == 'pending_marker': raw['metadata']['single_loop_checkpoint']['state']['inflight'] = ''
        if part == 'schema': action['schema_version'] = True
        if part == 'missing_action': del raw['metadata']['single_loop_delivery_action']
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    with pytest.raises((ValueError, ClaimFenceUnavailable)):
        stage(setup)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM v2_outbox').fetchone()[0] == 0


@pytest.mark.parametrize('change', ['valid', 'actor', 'action', 'project', 'resources', 'revision', 'extra'])
def test_staging_checks_managed_preflight_shape(setup, change):
    path, _, questions, _, _, _, _ = setup
    with questions._connect() as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        action = raw['metadata']['single_loop_delivery_action']
        preflight = {'schema_version': 1, 'actor': 'alice', 'task_id': 'task', 'action_id': action['action_id'],
            'skill_id': 'skill', 'role': 'mail', 'binding_id': 'binding', 'binding_revision': 1,
            'required_scopes': [], 'required_executables': [], 'operation': 'send', 'resources': ['alice'],
            'project_id': None, 'recipient_user_ids': ['alice'], 'connection_id': action['integration_id'],
            'provider_key': action['provider_key'], 'channel_target': action['channel_target'],
            'message_digest': action['message_digest'], 'idempotency_key': action['idempotency_key'],
            'grant_id': action['grant_id'], 'grant_revision': action['grant_revision'],
            'connection_revision': action['connection_revision']}
        if change == 'actor': preflight['actor'] = 'bob'
        if change == 'action': preflight['action_id'] = 'other'
        if change == 'project': preflight['project_id'] = 'other'
        if change == 'resources': preflight['resources'] = ['other']
        if change == 'revision': preflight['grant_revision'] = True
        if change == 'extra': preflight['unexpected'] = 'value'
        raw['metadata']['single_loop_managed_preflight'] = preflight
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    if change == 'valid':
        # Shape alone is not current authority; the generic synthetic path
        # cannot stage a managed action.
        with pytest.raises(ClaimFenceUnavailable, match='managed_stage_authority_required'):
            stage(setup)
    else:
        with pytest.raises(ClaimFenceUnavailable, match='binding_unavailable'):
            stage(setup)
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT count(*) FROM v2_outbox').fetchone()[0] == 0


def test_staging_does_not_guess_upgrade_for_legacy_v1_action(setup):
    path, _, questions, _, _, _, _ = setup
    with questions._connect() as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        action = raw['metadata']['single_loop_delivery_action']
        action['schema_version'] = 1
        del action['grant_id']
        del action['connection_revision']
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    with pytest.raises(ClaimFenceUnavailable, match='binding_unavailable'):
        stage(setup)
    with sqlite3.connect(path) as conn:
        restored = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        assert restored['metadata']['single_loop_delivery_action'] == action
        assert restored['metadata']['single_loop_checkpoint']['state']['inflight'] == 'action'
        assert conn.execute('SELECT count(*) FROM v2_outbox').fetchone()[0] == 0
