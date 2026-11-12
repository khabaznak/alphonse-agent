"""Disposable staging only; no production dispatch or recovery is enabled."""
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sqlite3

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.single_loop_delivery_reservation import DeliveryBinding, reserve_delivery
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, TaskBudget
from alphonse.agent_v2.core.io.experimental_staged_outbox import install_experimental_staging, stage_experimental_response
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
                              hashlib.sha256(message.encode('utf-8')).hexdigest(), 'key', 'grant')
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


def test_restage_after_lost_commit_response_returns_same_inert_record(setup):
    path, _, _, outbox, _, _, _ = setup
    first = stage(setup)
    assert stage(setup) == first
    with sqlite3.connect(path) as conn:
        rows = conn.execute('SELECT outbox_message_id,status FROM v2_outbox').fetchall()
    assert rows == [(first, 'staged')]
    assert outbox.claim_next() is None


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
