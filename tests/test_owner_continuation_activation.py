from datetime import datetime, timedelta, timezone
import json
import os

import pytest

from test_owner_atomic_checkpoint import setup  # shared disposable owner queue fixture
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_activation import activate_held_same_claim
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_grant import issue_grant, prepare_handoff
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_question import start_question, acknowledge_question, record_choice
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import CheckpointConflict
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, TaskBudget


def vec(a, m, i, o):
    return {'actions': a, 'model_calls': m, 'input_tokens': i, 'output_tokens': o}


def prepare(setup):
    _, _, _, queue, _, task, _, atomic = setup
    now = datetime.now(timezone.utc)
    atomic.clock = lambda: now
    ms = int(now.timestamp() * 1000)
    budget = TaskBudget(BudgetLimits(vec(3, 3, 1000, 300), vec(0, 1, 100, 50), 100000, 0), started_at_ms=ms-20)
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), budget)
    args = {'source_id': 'conversation:fixture'}
    decision = {'kind': 'tool', 'tool_id': 'native.read_memory_source', 'arguments': args}
    reserved = atomic.reserve(expected_version=start.version, reservation_id='model', kind='model',
                              amounts=vec(0, 1, 100, 50), now_ms=ms-15)
    selected = atomic.settle_model(expected_version=reserved.version, reservation_id='model', outcome='success',
                                   input_tokens=1, output_tokens=1, decision=decision, projection_fingerprint='a'*64)
    reserved = atomic.reserve(expected_version=selected.version, reservation_id='read', kind='tool',
                              amounts=vec(1, 0, 0, 0), now_ms=ms-10, tool_id=decision['tool_id'],
                              arguments=args, selected_decision=decision, projection_fingerprint='a'*64)
    settled = atomic.settle_read(expected_version=reserved.version, reservation_id='read', tool_id=decision['tool_id'],
                                 arguments=args, evidence_ref='e1', observation={'text': 'fixture'})
    paused = atomic.pause_for_continuation(expected_version=settled.version, metric='actions', interval=1,
                                           last_checkpoint=0, cancellation_checker=lambda: False)
    version, token = start_question(atomic, expected_version=paused.version, question='Continue?', cancellation_checker=lambda: False)
    version = acknowledge_question(atomic, expected_version=version, token=token, cancellation_checker=lambda: False)
    version = record_choice(atomic, expected_version=version, peer_uid=os.getuid(), approve=True, cancellation_checker=lambda: False)
    version = issue_grant(atomic, expected_version=version, now_ms=ms, lifetime_ms=10000, cancellation_checker=lambda: False)
    version = prepare_handoff(atomic, expected_version=version, now_ms=ms, cancellation_checker=lambda: False)
    return queue, task, atomic, version, now


def test_activation_preserves_reservations_and_same_claim(setup, monkeypatch):
    queue, task, atomic, version, now = prepare(setup)
    observed = []
    monkeypatch.setattr('alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery.restore_owner_read_observations',
                        lambda *args, **kwargs: observed.append(kwargs['expected_version']))
    context = CoreLoopContext(messages=queue, claim_receipt=atomic.receipt, cancellation_checker=lambda: False)
    before, bindings = atomic.load(expected_version=version)
    atomic.clock = lambda: now + timedelta(milliseconds=30)
    result = activate_held_same_claim(atomic, expected_version=version, context=context, cancellation_checker=lambda: False)
    assert observed == [version]
    assert result.budget.to_dict()['reservations'] == before.budget.to_dict()['reservations']
    assert result.budget.limits.total['actions'] == before.budget.limits.total['actions'] + 1
    assert result.budget.to_dict()['excluded_pause_ms'] == 30
    assert result.state.evidence_sources == ['e1']
    with queue._connect() as conn:
        raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()[0])
        assert raw['metadata']['owner_read_observations'] == bindings
        assert 'owner_continuation_pause' not in raw['metadata']
        assert len(raw['metadata']['owner_continuation_history']) == 1
    with pytest.raises(CheckpointConflict):
        activate_held_same_claim(atomic, expected_version=result.version, context=context, cancellation_checker=lambda: False)


def test_expired_grant_and_cancel_do_not_unpause(setup, monkeypatch):
    queue, _, atomic, version, now = prepare(setup)
    monkeypatch.setattr('alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery.restore_owner_read_observations',
                        lambda *args, **kwargs: None)
    context = CoreLoopContext(messages=queue, claim_receipt=atomic.receipt, cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict):
        activate_held_same_claim(atomic, expected_version=version, context=context, cancellation_checker=lambda: True)
    atomic.clock = lambda: now + timedelta(seconds=10)
    with pytest.raises(CheckpointConflict):
        activate_held_same_claim(atomic, expected_version=version, context=context, cancellation_checker=lambda: False)
    result, _ = atomic.load(expected_version=version)
    assert result.budget.to_dict()['schema_version'] == 1


def test_activation_fails_if_source_revalidation_fails(setup):
    queue, _, atomic, version, _ = prepare(setup)
    context = CoreLoopContext(messages=queue, claim_receipt=atomic.receipt, cancellation_checker=lambda: False)
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable
    with pytest.raises(OwnerRecoveryUnavailable):
        activate_held_same_claim(atomic, expected_version=version, context=context, cancellation_checker=lambda: False)
    result, _ = atomic.load(expected_version=version)
    assert result.budget.to_dict()['schema_version'] == 1
