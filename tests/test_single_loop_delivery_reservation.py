"""Isolated reservation tests; no actual authorization or delivery occurs."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import hashlib

import pytest

from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ENGINE, ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.single_loop_delivery_reservation import ACTION_KEY, DeliveryBinding, reserve_delivery
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, TaskBudget
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    users = V2UserStore(path)
    users.set_users_root(tmp_path / 'profiles')
    users.create_user(user_id='alice', display_name='Alice')
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    questions = SQLiteQuestionStore(path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(now, 'synthetic', 'alice'), message_id='message')
    claim = queue.claim_next(lease_seconds=60).claim_receipt
    task = TaskState(task_id='task', message_id='message', user='alice', goal='synthetic',
                     intelligence_engine=ENGINE, intelligence_schema_version=1)
    version = questions.save_task_checkpoint(task, status='running')
    store = ExperimentalCheckpointStore(questions)
    total = dict(actions=2, model_calls=2, input_tokens=100, output_tokens=100)
    final = dict(actions=0, model_calls=0, input_tokens=0, output_tokens=0)
    checkpoint = store.initialize('task', 'alice', expected_version=version,
        state=CandidateState('task', 'alice', 'goal'),
        budget=TaskBudget(BudgetLimits(total, final, 10000, 0), started_at_ms=1000))
    binding = DeliveryBinding('action', 'synthetic', 'fixture-provider', 'channel', 'alice', hashlib.sha256(b'synthetic message').hexdigest(), 'idempotency', 'grant-1', 1, 1)
    return queue, users, questions, store, claim, checkpoint.version, now, binding


def reserve(f, **kwargs):
    _, _, _, store, claim, version, now, binding = f
    return reserve_delivery(store, kwargs.pop('claim', claim), task_id='task', actor=kwargs.pop('actor', 'alice'),
        expected_version=kwargs.pop('version', version), binding=kwargs.pop('binding', binding),
        message=kwargs.pop('message', 'synthetic message'), now_ms=kwargs.pop('now_ms', 1100),
        clock=kwargs.pop('clock', lambda: now), **kwargs)


def test_binding_and_budget_survive_restart_without_clear_or_retry(setup):
    _, _, questions, store, claim, _, _, binding = setup
    saved = reserve(setup)
    restored = ExperimentalCheckpointStore(questions).load('task', 'alice', expected_version=saved.version)
    assert restored.state.inflight == binding.action_id
    assert restored.budget.usage()['in_flight'] == [binding.action_id]
    metadata = questions.load_task_checkpoint('task').metadata
    assert metadata[ACTION_KEY]['message_digest'] == binding.message_digest
    assert metadata[ACTION_KEY]['recipient_user_id'] == 'alice'
    assert claim.claim_generation not in str(metadata)
    with pytest.raises(ValueError, match='verified_evidence_integration_required'):
        store.settle('task', 'alice', expected_version=saved.version, reservation_id='action', outcome='success')
    with pytest.raises(ValueError, match='already_bound_or_inflight'):
        reserve(setup, version=saved.version, binding=replace(binding, channel_target='another-channel'))


def test_mismatched_message_digest_cannot_reserve_or_create_marker(setup):
    _, _, questions, store, _, version, _, _ = setup
    with pytest.raises(ValueError, match='message_mismatch'):
        reserve(setup, message='synthetic message ')
    assert ACTION_KEY not in questions.load_task_checkpoint('task').metadata
    assert not store.load('task', 'alice', expected_version=version).state.inflight


@pytest.mark.parametrize('field,value', [('grant_revision', True), ('grant_revision', 0),
                                        ('grant_revision', '1'), ('connection_revision', False),
                                        ('connection_revision', -1), ('connection_revision', '1'),
                                        ('grant_id', '')])
def test_binding_rejects_invalid_grant_identity(setup, field, value):
    with pytest.raises(ValueError, match='delivery_binding_'):
        replace(setup[-1], **{field: value})


def test_v1_action_is_not_adopted_or_reconciled_by_guessing(setup):
    import json
    from alphonse.agent_v2.accounts import AccountSession
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_delivery_binding

    _, _, questions, store, claim, _, now, binding = setup
    saved = reserve(setup)
    with questions._connect() as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        action = raw['metadata'][ACTION_KEY]
        action['schema_version'] = 1
        action.pop('grant_id')
        action.pop('connection_revision')
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    result = inspect_delivery_binding(store, AccountSession('alice'), claim, task_id='task',
                                      expected_version=saved.version, expected=binding, clock=lambda: now)
    assert result.status == 'unavailable'
    assert store.load('task', 'alice', expected_version=saved.version).state.inflight == binding.action_id


def test_message_digest_preserves_exact_unicode_bytes(setup):
    _, _, _, _, _, _, _, binding = setup
    text = 'café'
    matching = replace(binding, message_digest=hashlib.sha256(text.encode('utf-8')).hexdigest())
    with pytest.raises(ValueError, match='message_mismatch'):
        reserve(setup, message='cafe\u0301', binding=matching)
    assert reserve(setup, message=text, binding=matching).state.inflight == binding.action_id


def test_stale_or_expired_claim_and_inactive_owner_cannot_reserve(setup):
    queue, users, questions, _, claim, version, now, _ = setup
    moments = iter((now, now + timedelta(days=1)))
    with pytest.raises(ClaimSuperseded):
        reserve(setup, clock=lambda: next(moments))
    assert ACTION_KEY not in questions.load_task_checkpoint('task').metadata
    assert setup[3].load('task', 'alice', expected_version=version).state.inflight == ''
    users.update_user('alice', is_active=False)
    with pytest.raises(ClaimFenceUnavailable):
        reserve(setup)
    assert ACTION_KEY not in questions.load_task_checkpoint('task').metadata


def test_mismatched_owner_checkpoint_or_invalid_binding_never_reserves(setup):
    _, _, questions, _, _, version, _, binding = setup
    with pytest.raises(ClaimFenceUnavailable):
        reserve(setup, actor='bob')
    with pytest.raises(ValueError):
        reserve(setup, version=version + 1)
    with pytest.raises(ValueError, match='digest_invalid'):
        replace(binding, message_digest='no')
    assert ACTION_KEY not in questions.load_task_checkpoint('task').metadata


def test_same_owner_reclaim_invalidates_prior_generation(setup):
    queue, _, questions, _, old_claim, _, now, _ = setup
    assert queue.reclaim_expired(now=now + timedelta(days=1)) == 1
    replacement = queue.claim_next(lease_seconds=60).claim_receipt
    assert replacement.lease_owner == old_claim.lease_owner
    assert replacement.claim_generation != old_claim.claim_generation
    with pytest.raises(ClaimSuperseded):
        reserve(setup, claim=old_claim)
    assert ACTION_KEY not in questions.load_task_checkpoint('task').metadata


def test_cancelled_checkpoint_never_reserves(setup):
    _, _, questions, _, _, _, _, _ = setup
    assert questions.mark_task_checkpoint_terminal('task', status='cancelled')
    with pytest.raises(ValueError):
        reserve(setup)
    assert ACTION_KEY not in questions.load_task_checkpoint('task').metadata


def test_claim_for_different_message_is_not_bound_to_task(setup):
    from alphonse.agent_v2.core.messages.claim_fence import ClaimReceipt
    _, _, questions, _, claim, _, _, _ = setup
    other = ClaimReceipt('different-message', claim.user_id, claim.lease_owner, claim.claim_generation)
    with pytest.raises(ClaimSuperseded):
        reserve(setup, claim=other)
    assert ACTION_KEY not in questions.load_task_checkpoint('task').metadata


def test_rejected_deadline_persists_clock_without_reservation(setup):
    from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetExhausted
    _, _, questions, store, _, version, _, _ = setup
    with pytest.raises(BudgetExhausted):
        reserve(setup, now_ms=11000)
    restored = store.load('task', 'alice', expected_version=version + 1)
    assert restored.budget.last_observed_at_ms == 11000
    assert not restored.state.inflight
    assert ACTION_KEY not in questions.load_task_checkpoint('task').metadata


def test_competing_reservations_from_same_version_have_one_winner(setup):
    from concurrent.futures import ThreadPoolExecutor
    _, _, questions, store, _, version, _, binding = setup
    def attempt(action):
        try:
            return reserve(setup, binding=replace(binding, action_id=action)).state.inflight
        except ValueError:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, ('one', 'two')))
    assert results.count('conflict') == 1
    winner = next(result for result in results if result != 'conflict')
    saved = store.load('task', 'alice', expected_version=version + 1)
    assert saved.state.inflight == winner
    assert questions.load_task_checkpoint('task').metadata[ACTION_KEY]['action_id'] == winner


def test_recovery_inspection_requires_exact_binding_and_leaves_marker_untouched(setup):
    from alphonse.agent_v2.accounts import AccountSession
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_delivery_binding

    _, _, questions, store, claim, _, now, binding = setup
    saved = reserve(setup)
    def inspect(expected=binding, actor='alice', version=saved.version, clock=lambda: now):
        return inspect_delivery_binding(store, AccountSession(actor), claim, task_id='task',
                                        expected_version=version, expected=expected, clock=clock)
    assert inspect().status == 'binding_matches'
    for mismatch in (replace(binding, recipient_user_id='bob'),
                     replace(binding, provider_key='other-provider'),
                     replace(binding, message_digest='b' * 64),
                     replace(binding, grant_revision=2), replace(binding, grant_id='other-grant'),
                     replace(binding, connection_revision=2),
                     replace(binding, action_id='other')):
        assert inspect(expected=mismatch).status == 'unavailable'
    assert inspect(actor='bob').status == 'unavailable'
    assert inspect(version=saved.version + 1).status == 'unavailable'
    moments = iter((now, now + timedelta(days=1)))
    assert inspect(clock=lambda: next(moments)).status == 'unavailable'
    assert store.load('task', 'alice', expected_version=saved.version).state.inflight == binding.action_id
    assert questions.load_task_checkpoint('task').metadata[ACTION_KEY]['message_digest'] == binding.message_digest


def test_recovery_inspection_rejects_tampered_binding_and_inactive_owner(setup):
    import json
    from alphonse.agent_v2.accounts import AccountSession
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_delivery_binding

    _, users, questions, store, claim, _, now, binding = setup
    saved = reserve(setup)
    def inspect():
        return inspect_delivery_binding(store, AccountSession('alice'), claim, task_id='task',
                                        expected_version=saved.version, expected=binding, clock=lambda: now)
    users.update_user('alice', is_active=False)
    assert inspect().status == 'unavailable'
    users.update_user('alice', is_active=True)
    with questions._connect() as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        raw['metadata'][ACTION_KEY]['schema_version'] = True
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    assert inspect().status == 'unavailable'


@pytest.mark.parametrize('field', ['grant_revision', 'connection_revision'])
def test_recovery_inspection_rejects_boolean_revision_even_when_python_equal(setup, field):
    import json
    from alphonse.agent_v2.accounts import AccountSession
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_delivery_binding

    _, _, questions, store, claim, _, now, binding = setup
    saved = reserve(setup)
    with questions._connect() as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        raw['metadata'][ACTION_KEY][field] = True
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    result = inspect_delivery_binding(store, AccountSession('alice'), claim, task_id='task',
                                      expected_version=saved.version, expected=binding, clock=lambda: now)
    assert result.status == 'unavailable'
    assert store.load('task', 'alice', expected_version=saved.version).state.inflight == binding.action_id


def test_recovery_inspection_rejects_old_claim_after_same_owner_reclaim(setup):
    from alphonse.agent_v2.accounts import AccountSession
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_delivery_binding

    queue, _, _, store, claim, _, now, binding = setup
    saved = reserve(setup)
    assert queue.reclaim_expired(now=now + timedelta(days=1)) == 1
    replacement = queue.claim_next(lease_seconds=60).claim_receipt
    result = inspect_delivery_binding(store, AccountSession('alice'), claim, task_id='task',
                                      expected_version=saved.version, expected=binding, clock=lambda: now)
    assert result.status == 'unavailable'
    assert replacement.claim_generation != claim.claim_generation
    assert store.load('task', 'alice', expected_version=saved.version).state.inflight == binding.action_id


def test_recovery_inspection_locks_against_concurrent_revocation_then_rechecks(setup):
    import sqlite3
    from alphonse.agent_v2.accounts import AccountSession
    from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_delivery_binding

    _, users, questions, store, claim, _, now, binding = setup
    saved = reserve(setup)
    calls = 0
    def clock():
        nonlocal calls
        calls += 1
        if calls == 2:
            with sqlite3.connect(questions.db_path, timeout=0) as concurrent:
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    concurrent.execute("UPDATE v2_users SET is_active=0 WHERE user_id='alice'")
        return now
    def inspect(time_source):
        return inspect_delivery_binding(store, AccountSession('alice'), claim, task_id='task',
                                        expected_version=saved.version, expected=binding, clock=time_source)
    assert inspect(clock).status == 'binding_matches'
    users.update_user('alice', is_active=False)
    assert inspect(lambda: now).status == 'unavailable'
    assert store.load('task', 'alice', expected_version=saved.version).state.inflight == binding.action_id
