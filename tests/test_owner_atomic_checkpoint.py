"""Disposable owner candidate tests for the shared-claim checkpoint mode."""
import json
import os
import threading
import tempfile
from types import SimpleNamespace
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint, observation_digest
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor, authorize_trusted_owner_claim
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import CheckpointConflict, ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetExhausted, BudgetLimits, SQLiteTaskBudgetStore, TaskBudget
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def setup(tmp_path):
    path = tmp_path / 'owner.db'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Fixture', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    SQLiteOutboundStore(path)
    questions = SQLiteQuestionStore(path)
    SQLiteConversationStore(path)
    message_id = queue.enqueue_trusted_local_owner(prompt='Find the apples note.', peer_uid=os.getuid())
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    store = ExperimentalCheckpointStore(questions)
    atomic = OwnerAtomicCheckpoint(queue, store, task, acquired.claim_receipt, clock=lambda: datetime.now(timezone.utc))
    return tmp_path, users, owner, queue, acquired, task, store, atomic


def make_budget():
    vector = lambda a, m, i, o: {'actions': a, 'model_calls': m, 'input_tokens': i, 'output_tokens': o}
    return TaskBudget(BudgetLimits(vector(3, 3, 1000, 300), vector(0, 1, 100, 50), 100000, 0), started_at_ms=100)



def reserve_selected_tool(atomic, start, tool_id, arguments):
    projection = 'a' * 64
    decision = {'kind': 'tool', 'tool_id': tool_id, 'arguments': arguments}
    reserved = atomic.reserve(expected_version=start.version, reservation_id='model-prerequisite', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=100)
    settled = atomic.settle_model(expected_version=reserved.version, reservation_id='model-prerequisite',
        outcome='success', input_tokens=1, output_tokens=1, decision=decision, projection_fingerprint=projection)
    return atomic.reserve(expected_version=settled.version, reservation_id='read-1', kind='tool',
        amounts={'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}, now_ms=101,
        tool_id=tool_id, arguments=arguments, selected_decision=decision, projection_fingerprint=projection)


def test_pause_persists_quiescent_state_and_fences_later_reservation(setup):
    _, _, _, queue, acquired, task, store, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'source_id': 'conversation:fixture'}
    reserved = reserve_selected_tool(atomic, start, 'native.read_memory_source', args)
    with pytest.raises(CheckpointConflict, match='owner_pause_unavailable'):
        atomic.pause_for_continuation(expected_version=reserved.version, metric='actions', interval=1,
                                      last_checkpoint=0, cancellation_checker=lambda: False)
    settled = atomic.settle_read(expected_version=reserved.version, reservation_id='read-1',
        tool_id='native.read_memory_source', arguments=args, evidence_ref='e1', observation={'text': 'x'})
    paused = atomic.pause_for_continuation(expected_version=settled.version, metric='actions', interval=1,
                                            last_checkpoint=0, cancellation_checker=lambda: False)
    # A new checkpoint object observes the same marker and cannot continue.
    replacement = OwnerAtomicCheckpoint(queue, store, task, acquired.claim_receipt,
                                        clock=lambda: datetime.now(timezone.utc))
    restored, bindings = replacement.load(expected_version=paused.version)
    assert restored.state.evidence_sources == ['e1'] and bindings['e1']['arguments'] == args
    assert restored.budget.usage()['charged']['actions'] == 1
    with pytest.raises(CheckpointConflict, match='owner_continuation_paused'):
        replacement.reserve(expected_version=paused.version, reservation_id='new-model', kind='model',
            amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=102)
    replacement.version = paused.version
    with pytest.raises(CheckpointConflict, match='owner_continuation_paused'):
        replacement.save_state(restored.state)
    ctx = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt,
                          cancellation_checker=lambda: False)
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_reminder_create import OWNER_REMINDER_CREATE_TOOL_ID
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_create import SCHEDULED_TASK_TOOL_ID
    for method, tool in ((replacement.commit_owner_reminder, OWNER_REMINDER_CREATE_TOOL_ID),
                         (replacement.commit_owner_schedule, SCHEDULED_TASK_TOOL_ID)):
        with pytest.raises(CheckpointConflict, match='owner_continuation_paused'):
            method(context=ctx, decision={'kind': 'tool', 'tool_id': tool, 'arguments': {}},
                   projection_fingerprint='', now_ms=102)
    with pytest.raises(CheckpointConflict):
        atomic.reserve(expected_version=settled.version, reservation_id='stale-model', kind='model',
            amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=102)
    with pytest.raises(CheckpointConflict, match='owner_continuation_paused'):
        replacement.pause_for_continuation(expected_version=paused.version, metric='actions', interval=1,
                                           last_checkpoint=0, cancellation_checker=lambda: False)


def test_pause_requires_due_checkpoint_and_current_non_cancelled_claim(setup):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    with pytest.raises(CheckpointConflict, match='owner_pause_unavailable'):
        atomic.pause_for_continuation(expected_version=start.version, metric='actions', interval=1,
                                      last_checkpoint=0, cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict, match='owner_pause_unavailable'):
        atomic.pause_for_continuation(expected_version=start.version, metric='actions', interval=1,
                                      last_checkpoint=0, cancellation_checker=lambda: True)
    args = {'source_id': 'conversation:fixture'}
    reserved = reserve_selected_tool(atomic, start, 'native.read_memory_source', args)
    settled = atomic.settle_read(expected_version=reserved.version, reservation_id='read-1',
        tool_id='native.read_memory_source', arguments=args, evidence_ref='e1', observation={'text': 'x'})
    for cancelled in (True, None):
        with pytest.raises(CheckpointConflict, match='owner_pause_unavailable'):
            atomic.pause_for_continuation(expected_version=settled.version, metric='actions', interval=1,
                                          last_checkpoint=0, cancellation_checker=lambda: cancelled)
    from alphonse.agent_v2.core.messages.claim_fence import ClaimSuperseded
    atomic.clock = lambda: datetime.now(timezone.utc) + timedelta(days=1)
    with pytest.raises(ClaimSuperseded):
        atomic.pause_for_continuation(expected_version=settled.version, metric='actions', interval=1,
                                      last_checkpoint=0, cancellation_checker=lambda: False)


def test_acknowledged_choice_can_only_prepare_one_unexpired_held_handoff(setup):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_question import (
        start_question, acknowledge_question, record_choice)
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_grant import issue_grant, prepare_handoff
    _, _, _, queue, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'source_id': 'conversation:fixture'}
    reserved = reserve_selected_tool(atomic, start, 'native.read_memory_source', args)
    settled = atomic.settle_read(expected_version=reserved.version, reservation_id='read-1',
        tool_id='native.read_memory_source', arguments=args, evidence_ref='e1', observation={'text': 'x'})
    paused = atomic.pause_for_continuation(expected_version=settled.version, metric='actions', interval=1,
                                            last_checkpoint=0, cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict, match='owner_continuation_grant_unavailable'):
        issue_grant(atomic, expected_version=paused.version, now_ms=1000, lifetime_ms=100,
                    cancellation_checker=lambda: False)
    version, token = start_question(atomic, expected_version=paused.version, question='Continue?',
                                    cancellation_checker=lambda: False)
    version = acknowledge_question(atomic, expected_version=version, token=token, cancellation_checker=lambda: False)
    version = record_choice(atomic, expected_version=version, peer_uid=os.getuid(), approve=True,
                            cancellation_checker=lambda: False)
    version = issue_grant(atomic, expected_version=version, now_ms=1000, lifetime_ms=100,
                          cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict, match='owner_continuation_handoff_unavailable'):
        prepare_handoff(atomic, expected_version=version, now_ms=1100, cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict, match='owner_continuation_question_unavailable'):
        prepare_handoff(atomic, expected_version=version, now_ms=1001, cancellation_checker=lambda: True)
    prepared_version = prepare_handoff(atomic, expected_version=version, now_ms=1001,
                                       cancellation_checker=lambda: False)
    with queue._connect() as conn:
        record = conn.execute('SELECT status, claim_generation FROM v2_inbound_messages WHERE message_id=?',
                              (task.message_id,)).fetchone()
        raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                      (task.task_id,)).fetchone()[0])
    assert record['status'] == 'processing' and record['claim_generation'] == atomic.receipt.claim_generation
    assert raw['metadata']['owner_continuation_grant']['status'] == 'consumed'
    assert raw['metadata']['owner_continuation_handoff']['status'] == 'held'
    with pytest.raises(CheckpointConflict, match='owner_continuation_handoff_unavailable'):
        prepare_handoff(atomic, expected_version=prepared_version, now_ms=1002, cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict, match='owner_continuation_paused'):
        atomic.reserve(expected_version=prepared_version, reservation_id='no-replay', kind='model',
            amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=102)


def test_disposable_bash_observation_requires_same_live_session(setup):
    from alphonse.agent_v2.evaluation.dojo_bash import DojoBash
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable, restore_owner_read_observations
    _, _, _, queue, acquired, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'command': 'printf fixture'}
    reserved = reserve_selected_tool(atomic, start, 'native.bash', args)
    ctx = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt,
        conversation_store=SQLiteConversationStore(queue.db_path), cancellation_checker=lambda: False)
    shell = object.__new__(DojoBash)
    shell._active = True
    ctx._dojo_bash_service = shell
    ctx._dojo_bash_session = 'a' * 32
    observation = {'candidate_complete': True, 'status': 'completed', 'exit_code': 0,
        'stdout': 'fixture', 'stderr': '', 'truncated': False, 'cleanup_verified': None, 'duration_ms': 1}
    with pytest.raises(ValueError, match='owner_atomic_dojo_bash_invalid'):
        atomic.settle_dojo_bash(expected_version=reserved.version, reservation_id='read-1', arguments=args,
            evidence_ref='e1', observation=observation, session_id='b' * 32, context=ctx)
    result = atomic.settle_dojo_bash(expected_version=reserved.version, reservation_id='read-1', arguments=args,
        evidence_ref='e1', observation=observation, session_id='a' * 32, context=ctx)
    checkpoint, bindings = restore_owner_read_observations(atomic, ctx, expected_version=result.version)
    assert checkpoint.state.inflight == '' and bindings['e1']['observation'] == observation
    shell._active = False
    with pytest.raises(OwnerRecoveryUnavailable):
        restore_owner_read_observations(atomic, ctx, expected_version=result.version)

def test_reservation_and_observation_are_atomic_and_reloadable(setup):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = reserve_selected_tool(atomic, start, 'native.read_memory_source',
                                     {'source_id': 'conversation:fixture'})
    checkpoint, observations = atomic.load(expected_version=reserved.version)
    assert checkpoint.state.inflight == 'read-1' and checkpoint.budget.usage()['in_flight'] == ['read-1']
    assert observations == {}
    with pytest.raises(CheckpointConflict):
        atomic.settle_read(expected_version=reserved.version, reservation_id='read-1', tool_id='native.read_memory_source',
            arguments={'source_id': 'conversation:other'}, evidence_ref='e1', observation={'text': 'x'})
    settled = atomic.settle_read(expected_version=reserved.version, reservation_id='read-1', tool_id='native.read_memory_source',
        arguments={'source_id': 'conversation:fixture'}, evidence_ref='e1', observation={'text': 'x'})
    restored, observations = atomic.load(expected_version=settled.version)
    assert restored.state.inflight == '' and restored.state.evidence_sources == ['e1']
    assert not restored.budget.usage()['in_flight']
    assert observations['e1'] == {'actor': task.user, 'tool_id': 'native.read_memory_source',
                                 'arguments': {'source_id': 'conversation:fixture'},
                                 'observation_sha256': observation_digest({'text': 'x'})}


def test_bash_return_is_unreconciled_and_cannot_be_settled_as_read(setup):
    _, _, _, queue, acquired, task, store, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'command': 'printf synthetic'}
    reserved = reserve_selected_tool(atomic, start, 'native.bash', args)
    with pytest.raises(ValueError, match='owner_atomic_shell_not_read'):
        atomic.settle_read(expected_version=reserved.version, reservation_id='read-1', tool_id='native.bash',
                           arguments=args, evidence_ref='e1', observation={'exit_code': 0})
    recorded = atomic.record_bash_return(expected_version=reserved.version, reservation_id='read-1',
        arguments=args, exit_code=0, timed_out=False, cancelled=False,
        stdout='synthetic-private-output', stderr='')
    # This is caller-reported process evidence, not proof of completion or effect.
    with queue._connect() as conn:
        raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                      (task.task_id,)).fetchone()[0])
    assert 'synthetic-private-output' not in json.dumps(raw)
    marker = raw['metadata']['owner_bash_return_unreconciled']
    assert marker['effect_status'] == 'unverified' and marker['exit_code'] == 0
    assert marker['stdout_bytes'] == len('synthetic-private-output')
    restored_atomic = OwnerAtomicCheckpoint(queue, store, task, acquired.claim_receipt,
                                             clock=lambda: datetime.now(timezone.utc))
    saved, observations = restored_atomic.load(expected_version=recorded.version)
    assert saved.state.inflight == 'read-1' and saved.budget.usage()['in_flight'] == ['read-1']
    assert observations == {}
    future = lambda: datetime.now(timezone.utc) + timedelta(days=1)
    assert queue.can_abandon_trusted_owner_message(task.message_id, clock=future) is False
    with pytest.raises(ValueError, match='owner_abandon_unavailable'):
        queue.abandon_trusted_owner_message(task.message_id, clock=future)
    with pytest.raises(CheckpointConflict, match='owner_atomic_inflight_requires_reconciliation'):
        restored_atomic.reserve(expected_version=recorded.version, reservation_id='another', kind='model',
            amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=102)
    with pytest.raises(CheckpointConflict, match='owner_atomic_bash_return_unavailable'):
        restored_atomic.record_bash_return(expected_version=recorded.version, reservation_id='read-1',
            arguments=args, exit_code=0, timed_out=False, cancelled=False, stdout='different', stderr='')


def test_staged_host_bash_executes_once_and_keeps_effect_unverified(setup):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_host_bash import dispatch_owner_host_bash
    root, _, owner, queue, acquired, task, store, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, owner.user_id, 'goal'), make_budget())
    args = {'command': 'printf x >> result.txt; cat result.txt', 'cwd': str(root)}
    reserve_selected_tool(atomic, start, 'native.bash', args)
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt,
                              cancellation_checker=lambda: False)
    authorize = lambda actor, task_id, command: (actor == owner.user_id and task_id == task.task_id and command == args)
    result = dispatch_owner_host_bash(atomic=atomic, context=context, arguments=args,
                                      reservation_id='read-1', authorize=authorize)
    assert result == {'exit_code': 0, 'stdout': 'x', 'stderr': '', 'timed_out': False,
                      'cancelled': False, 'effect_status': 'unverified'}
    assert (root / 'result.txt').read_text() == 'x'
    saved, observations = atomic.load(expected_version=atomic.version)
    assert saved.state.inflight == 'read-1' and observations == {}
    with pytest.raises(CheckpointConflict, match='owner_atomic_bash_dispatch_unavailable'):
        dispatch_owner_host_bash(atomic=atomic, context=context, arguments=args,
                                 reservation_id='read-1', authorize=authorize)
    assert (root / 'result.txt').read_text() == 'x'
    # A replacement object cannot replay the already-reserved action either.
    replacement = OwnerAtomicCheckpoint(queue, store, task, acquired.claim_receipt,
                                        clock=lambda: datetime.now(timezone.utc))
    replacement.version = atomic.version
    with pytest.raises(CheckpointConflict, match='owner_atomic_bash_dispatch_unavailable'):
        dispatch_owner_host_bash(atomic=replacement, context=context, arguments=args,
                                 reservation_id='read-1', authorize=authorize)
    assert (root / 'result.txt').read_text() == 'x'
    with pytest.raises(CheckpointConflict, match='owner_atomic_bash_read_unavailable'):
        atomic.settle_host_bash_return(expected_version=atomic.version, reservation_id='read-1',
            arguments=args, evidence_ref='host-output', authorize=lambda *_: False,
            cancellation_checker=lambda: False)
    settled = atomic.settle_host_bash_return(expected_version=atomic.version, reservation_id='read-1',
        arguments=args, evidence_ref='host-output', authorize=authorize, cancellation_checker=lambda: False)
    saved, observations = atomic.load(expected_version=settled.version)
    assert saved.state.inflight == '' and saved.state.evidence_sources == ['host-output']
    assert observations['host-output']['host_bash_observation']['stdout'] == 'x'
    assert observations['host-output']['host_bash_observation']['effect_status'] == 'unverified'
    with queue._connect() as conn:
        raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                      (task.task_id,)).fetchone()[0])
    assert raw['metadata']['owner_host_bash_history'][0]['effect_status'] == 'unverified'
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_host_bash import valid_host_binding
    context._owner_host_bash_enabled = True
    binding = observations['host-output']
    assert valid_host_binding(atomic, context, binding, raw['metadata']['owner_host_bash_history'])
    altered = dict(binding)
    altered['host_bash_observation'] = {**binding['host_bash_observation'], 'stdout': 'invented'}
    assert not valid_host_binding(atomic, context, altered, raw['metadata']['owner_host_bash_history'])
    context._owner_host_bash_enabled = False
    assert not valid_host_binding(atomic, context, binding, raw['metadata']['owner_host_bash_history'])
    assert queue.can_abandon_trusted_owner_message(task.message_id,
        clock=lambda: datetime.now(timezone.utc) + timedelta(days=1)) is False


def test_staged_host_bash_refuses_denial_and_never_replays_uncertain_dispatch(setup, monkeypatch):
    from alphonse.agent_v2.core.intelligence.simple_loop import owner_host_bash
    root, _, owner, queue, acquired, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, owner.user_id, 'goal'), make_budget())
    args = {'command': 'printf x > result.txt', 'cwd': str(root)}
    reserve_selected_tool(atomic, start, 'native.bash', args)
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt,
                              cancellation_checker=lambda: False)
    with pytest.raises(PermissionError, match='owner_host_bash_unavailable'):
        owner_host_bash.dispatch_owner_host_bash(atomic=atomic, context=context, arguments=args,
            reservation_id='read-1', authorize=lambda *_: False)
    assert not (root / 'result.txt').exists()
    def uncertain(*_args, **_kwargs):
        raise RuntimeError('synthetic dispatch interruption')
    monkeypatch.setattr(owner_host_bash, 'execute_bash', uncertain)
    with pytest.raises(RuntimeError, match='synthetic dispatch interruption'):
        owner_host_bash.dispatch_owner_host_bash(atomic=atomic, context=context, arguments=args,
            reservation_id='read-1', authorize=lambda *_: True)
    with pytest.raises(CheckpointConflict, match='owner_atomic_bash_dispatch_unavailable'):
        owner_host_bash.dispatch_owner_host_bash(atomic=atomic, context=context, arguments=args,
            reservation_id='read-1', authorize=lambda *_: True)
    assert not (root / 'result.txt').exists()


def test_bash_return_rejects_different_action_and_stale_claim(setup):
    _, _, _, queue, acquired, task, store, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'command': 'printf synthetic'}
    reserved = reserve_selected_tool(atomic, start, 'native.bash', args)
    with pytest.raises(CheckpointConflict, match='owner_atomic_bash_return_unavailable'):
        atomic.record_bash_return(expected_version=reserved.version, reservation_id='read-1',
            arguments={'command': 'different'}, exit_code=0, timed_out=False, cancelled=False, stdout='', stderr='')
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET status='pending', lease_owner='', lease_expires_at='' WHERE message_id=?", (task.message_id,))
    replacement = queue.claim_next(lease_owner='replacement')
    assert replacement.claim_receipt.claim_generation != acquired.claim_receipt.claim_generation
    new_worker = OwnerAtomicCheckpoint(queue, store, task, replacement.claim_receipt,
                                       clock=lambda: datetime.now(timezone.utc))
    with pytest.raises(CheckpointConflict, match='owner_atomic_bash_return_unavailable'):
        new_worker.record_bash_return(expected_version=reserved.version, reservation_id='read-1',
            arguments=args, exit_code=0, timed_out=False, cancelled=False, stdout='', stderr='')


def test_redacted_bash_return_requires_live_claim_action_and_authorization(setup):
    from alphonse.agent_v2.skill_bindings import ExecutionBindings
    from alphonse.agent_v2.core.messages.claim_fence import ClaimSuperseded
    _, _, _, queue, acquired, task, store, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'command': 'synthetic'}
    reserved = reserve_selected_tool(atomic, start, 'native.bash', args)
    secret = 'fixture-secret-123'
    recorded = atomic.record_bash_return(expected_version=reserved.version, reservation_id='read-1',
        arguments=args, exit_code=0, timed_out=False, cancelled=False,
        stdout='owner output ' + secret, stderr='diagnostic ' + secret,
        output_bindings=ExecutionBindings({}, (secret,)))
    with queue._connect() as conn:
        raw = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()[0]
    assert secret not in raw
    authorize = lambda actor, task_id, action: actor == task.user and task_id == task.task_id and action == args
    read_args = dict(expected_version=recorded.version, reservation_id='read-1', arguments=args,
                     authorize=authorize, cancellation_checker=lambda: False)
    result = atomic.read_bash_return(**read_args)
    assert result == {'exit_code': 0, 'timed_out': False, 'cancelled': False,
                      'effect_status': 'unverified', 'stdout': 'owner output [REDACTED]',
                      'stderr': 'diagnostic [REDACTED]'}
    saved, _ = atomic.load(expected_version=recorded.version)
    assert saved.state.inflight == 'read-1' and saved.budget.usage()['in_flight'] == ['read-1']
    for overrides in ({'authorize': lambda *_: False}, {'cancellation_checker': lambda: True},
                      {'arguments': {'command': 'other'}}, {'reservation_id': 'other'},
                      {'expected_version': reserved.version}):
        with pytest.raises((CheckpointConflict, ValueError)):
            atomic.read_bash_return(**{**read_args, **overrides})
    expired = OwnerAtomicCheckpoint(queue, store, task, acquired.claim_receipt,
                                    clock=lambda: datetime.now(timezone.utc) + timedelta(days=1))
    with pytest.raises(ClaimSuperseded):
        expired.read_bash_return(**read_args)


def test_redacted_bash_return_denies_inactive_actor_and_replaced_claim(setup):
    from alphonse.agent_v2.skill_bindings import ExecutionBindings
    from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
    _, users, owner, queue, acquired, task, store, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'command': 'fixture'}
    reserved = reserve_selected_tool(atomic, start, 'native.bash', args)
    recorded = atomic.record_bash_return(expected_version=reserved.version, reservation_id='read-1',
        arguments=args, exit_code=0, timed_out=False, cancelled=False, stdout='private', stderr='',
        output_bindings=ExecutionBindings({}, ()))
    read_args = dict(expected_version=recorded.version, reservation_id='read-1', arguments=args,
                     authorize=lambda *_: True, cancellation_checker=lambda: False)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner.user_id,))
    with pytest.raises(ClaimFenceUnavailable):
        atomic.read_bash_return(**read_args)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_users SET is_active=1 WHERE user_id=?', (owner.user_id,))
        conn.execute("UPDATE v2_inbound_messages SET status='pending', lease_owner='', lease_expires_at='' WHERE message_id=?", (task.message_id,))
    replacement = queue.claim_next(lease_owner='replacement')
    with pytest.raises(ClaimSuperseded):
        atomic.read_bash_return(**read_args)
    new_worker = OwnerAtomicCheckpoint(queue, store, task, replacement.claim_receipt,
                                       clock=lambda: datetime.now(timezone.utc))
    with pytest.raises(CheckpointConflict, match='owner_atomic_bash_read_unavailable'):
        new_worker.read_bash_return(**read_args)


def test_bash_output_read_refuses_tampering_and_revoked_access(setup):
    from alphonse.agent_v2.skill_bindings import ExecutionBindings
    _, _, _, queue, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'command': 'fixture'}
    reserved = reserve_selected_tool(atomic, start, 'native.bash', args)
    recorded = atomic.record_bash_return(expected_version=reserved.version, reservation_id='read-1',
        arguments=args, exit_code=1, timed_out=False, cancelled=False, stdout='partial', stderr='failure',
        output_bindings=ExecutionBindings({}, ()))
    calls = []
    def revoked_after_read(*_):
        calls.append(1)
        return len(calls) < 2
    with pytest.raises(CheckpointConflict, match='owner_atomic_bash_read_unavailable'):
        atomic.read_bash_return(expected_version=recorded.version, reservation_id='read-1', arguments=args,
                                authorize=revoked_after_read, cancellation_checker=lambda: False)
    with queue._connect() as conn:
        row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()
        raw = json.loads(row[0])
        raw['metadata']['owner_bash_return_unreconciled']['redacted_output']['stdout'] = 'tampered'
        conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?', (json.dumps(raw), task.task_id))
    with pytest.raises(CheckpointConflict, match='owner_atomic_bash_read_unavailable'):
        atomic.read_bash_return(expected_version=recorded.version, reservation_id='read-1', arguments=args,
                                authorize=lambda *_: True, cancellation_checker=lambda: False)


@pytest.mark.parametrize('change', ['actor', 'receipt'])
def test_bash_output_read_rechecks_claim_and_receipt_after_authorization_io(setup, change):
    from alphonse.agent_v2.skill_bindings import ExecutionBindings
    from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable
    _, _, owner, queue, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'command': 'fixture'}
    reserved = reserve_selected_tool(atomic, start, 'native.bash', args)
    recorded = atomic.record_bash_return(expected_version=reserved.version, reservation_id='read-1',
        arguments=args, exit_code=0, timed_out=False, cancelled=False, stdout='private', stderr='',
        output_bindings=ExecutionBindings({}, ()))
    calls = []
    def authorize(*_):
        calls.append(1)
        if len(calls) == 2:
            with queue._connect() as conn:
                if change == 'actor':
                    conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner.user_id,))
                else:
                    row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()
                    raw = json.loads(row[0])
                    raw['metadata']['owner_bash_return_unreconciled']['exit_code'] = 9
                    conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?', (json.dumps(raw), task.task_id))
        return True
    with pytest.raises((CheckpointConflict, ClaimFenceUnavailable)):
        atomic.read_bash_return(expected_version=recorded.version, reservation_id='read-1', arguments=args,
                                authorize=authorize, cancellation_checker=lambda: False)
    assert len(calls) == 2


@pytest.mark.parametrize(('exit_code', 'timed_out', 'cancelled'), [(1, False, False), (-1, True, False), (-1, False, True)])
def test_bash_failure_or_interruption_never_settles_an_effect(setup, exit_code, timed_out, cancelled):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'command': 'fixture'}
    reserved = reserve_selected_tool(atomic, start, 'native.bash', args)
    result = atomic.record_bash_return(expected_version=reserved.version, reservation_id='read-1',
        arguments=args, exit_code=exit_code, timed_out=timed_out, cancelled=cancelled, stdout='', stderr='')
    saved, _ = atomic.load(expected_version=result.version)
    assert saved.state.inflight == 'read-1' and saved.budget.usage()['in_flight'] == ['read-1']


@pytest.mark.parametrize(('exit_code', 'timed_out', 'cancelled'), [(0, True, False), (0, False, True), (-1, True, True)])
def test_bash_return_rejects_contradictory_exit_status(setup, exit_code, timed_out, cancelled):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'command': 'fixture'}
    reserved = reserve_selected_tool(atomic, start, 'native.bash', args)
    with pytest.raises(ValueError, match='owner_atomic_bash_return_invalid'):
        atomic.record_bash_return(expected_version=reserved.version, reservation_id='read-1',
            arguments=args, exit_code=exit_code, timed_out=timed_out, cancelled=cancelled, stdout='', stderr='')
    saved, _ = atomic.load(expected_version=reserved.version)
    assert saved.state.inflight == 'read-1'


def test_bash_return_accepts_bounded_multibyte_output_without_persisting_it(setup):
    _, _, _, queue, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'command': 'fixture'}
    reserved = reserve_selected_tool(atomic, start, 'native.bash', args)
    output = '水' * 12000 + '\n... [truncated]'
    recorded = atomic.record_bash_return(expected_version=reserved.version, reservation_id='read-1',
        arguments=args, exit_code=0, timed_out=False, cancelled=False, stdout=output, stderr='')
    with queue._connect() as conn:
        raw = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()[0]
    assert output not in raw
    assert json.loads(raw)['metadata']['owner_bash_return_unreconciled']['stdout_bytes'] == len(output.encode())
    saved, _ = atomic.load(expected_version=recorded.version)
    assert saved.state.inflight == 'read-1'


def test_replacement_claim_cannot_settle_prior_reservation(setup):
    _, _, _, queue, acquired, task, store, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = atomic.reserve(expected_version=start.version, reservation_id='m1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    # Simulate a recovery coordinator putting a lost lease back on the queue.
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET status='pending', lease_owner='', lease_expires_at='' WHERE message_id=?", (task.message_id,))
    replacement = queue.claim_next(lease_owner='replacement')
    assert replacement and replacement.claim_receipt.claim_generation != acquired.claim_receipt.claim_generation
    new_worker = OwnerAtomicCheckpoint(queue, store, task, replacement.claim_receipt, clock=lambda: datetime.now(timezone.utc))
    with pytest.raises(CheckpointConflict, match='owner_atomic_reservation_mismatch'):
        new_worker.settle_model(expected_version=reserved.version, reservation_id='m1', outcome='success', input_tokens=2, output_tokens=1)
    loaded, _ = new_worker.load(expected_version=reserved.version)
    assert loaded.state.inflight == 'm1'


@pytest.mark.parametrize('actual,allowed', [(101, True), (901, False)])
def test_actual_usage_controls_atomic_tool_dispatch(setup, actual, allowed):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = atomic.reserve(expected_version=start.version, reservation_id='m1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    decision = {'kind': 'tool', 'tool_id': 'native.read_memory_source',
                'arguments': {'source_id': 'conversation:fixture'}}
    settled = atomic.settle_model(expected_version=reserved.version, reservation_id='m1', outcome='success',
                                  input_tokens=actual, output_tokens=1, decision=decision,
                                  projection_fingerprint='a' * 64)
    def reserve():
        atomic.reserve(expected_version=settled.version, reservation_id='read', kind='tool',
            amounts={'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}, now_ms=102,
            tool_id='native.read_memory_source', arguments=decision['arguments'],
            selected_decision=decision, projection_fingerprint='a' * 64)
    if allowed:
        reserve()
    else:
        with pytest.raises(BudgetExhausted, match='input_tokens'):
            reserve()


@pytest.mark.parametrize('actual', [None, 901])
def test_atomic_answer_cannot_complete_with_unknown_or_exhausted_budget(setup, actual):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    pending = atomic.reserve(expected_version=start.version, reservation_id='m1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    decision = {'kind': 'answer', 'text': 'A synthetic answer.'}
    settled = atomic.settle_model(expected_version=pending.version, reservation_id='m1', outcome='success',
        input_tokens=actual, output_tokens=1, decision=decision, projection_fingerprint='a' * 64)
    settled.state.status = 'completed'
    settled.state.response = decision['text']
    with pytest.raises(CheckpointConflict, match='owner_atomic_model_budget_unavailable'):
        atomic.save_state(settled.state, consume_decision=True, selected_decision=decision,
                          projection_fingerprint='a' * 64)


def test_crash_after_model_settlement_does_not_replay_lost_decision(setup):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = atomic.reserve(expected_version=start.version, reservation_id='m1', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    settled = atomic.settle_model(expected_version=reserved.version, reservation_id='m1', outcome='success',
                                  input_tokens=10, output_tokens=5)
    with pytest.raises(CheckpointConflict, match='owner_atomic_pending_decision_unresolved'):
        atomic.load(expected_version=settled.version)
    with pytest.raises(CheckpointConflict, match='owner_atomic_pending_decision_unresolved'):
        atomic.reserve(expected_version=settled.version, reservation_id='m2', kind='model',
            amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=102)


def test_dispatch_check_requires_current_checkpoint_and_exact_reserved_action(setup):
    _, _, _, queue, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = reserve_selected_tool(atomic, start, 'native.read_memory_source',
                                     {'source_id': 'conversation:fixture'})
    assert atomic.check_claim('read-1', 'tool', tool_id='native.read_memory_source', arguments={'source_id': 'conversation:fixture'})
    assert not atomic.check_claim('read-1', 'tool', tool_id='native.read_memory_source', arguments={'source_id': 'conversation:other'})
    with queue._connect() as conn:
        conn.execute("UPDATE v2_task_checkpoints SET status='cancelled', version=? WHERE task_id=?", (reserved.version + 1, task.task_id))
    assert not atomic.check_claim('read-1', 'tool', tool_id='native.read_memory_source', arguments={'source_id': 'conversation:fixture'})


def test_rejected_reservation_retains_time_observation_and_can_save_blocker(setup):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    with pytest.raises(BudgetExhausted):
        atomic.reserve(expected_version=start.version, reservation_id='m1', kind='model',
            amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=100100)
    assert atomic.version == start.version + 1
    state = CandidateState(task.task_id, task.user, 'goal', status='blocked', blocker='budget exhausted')
    result = atomic.save_state(state)
    loaded, _ = atomic.load(expected_version=result.version)
    assert loaded.state.status == 'blocked' and loaded.budget.last_observed_at_ms == 100100


class Provider:
    supports_image_input = False
    def __init__(self, source_id):
        self.source_id, self.requests = source_id, []
    def generate_json(self, request):
        self.requests.append(request)
        actions = [
            {'kind': 'tool', 'tool_id': 'native.search_memory', 'arguments': {'query': 'apples', 'source_types': ['conversation']}, 'text': None},
            {'kind': 'tool', 'tool_id': 'native.read_memory_source', 'arguments': {'source_id': self.source_id, 'neighbors': 0}, 'text': None},
            {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic apples.'},
        ]
        result = actions[len(self.requests) - 1]
        return InferenceResult(json_value=result, raw_response=json.dumps(result), usage={'input_tokens': 100, 'output_tokens': 20})


@pytest.mark.parametrize('mode', ['acknowledged', 'legacy_origin', 'source_revoked', 'owner_deactivated', 'deactivated_ack', 'expired_ack', 'reclaimed', 'history_failure', 'legacy_handoff'])
def test_real_owner_processor_to_peer_verified_one_time_pickup(setup, mode, tmp_path):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
    _, users, owner, queue, acquired, task, store, _ = setup
    conversation = SQLiteConversationStore(queue.db_path)
    if mode == 'legacy_origin':
        # Simulate a genuine pre-upgrade in-flight request: no ingress event,
        # old payload digest, and version-zero origin. It remains recoverable.
        from alphonse.agent_v2.core.messages.sqlite_queue import _owner_ingress_digest
        with queue._connect() as conn:
            queued = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (task.message_id,)).fetchone()
            payload = [queued[key] for key in ('message_id', 'timestamp', 'prompt', 'user_id', 'project_id',
                       'tag', 'correlation_id', 'memory_session_id', 'metadata_json')]
            conn.execute('DELETE FROM v2_conversation_events WHERE source_message_id=?', (f'inbound:{task.message_id}',))
            conn.execute("UPDATE v2_trusted_owner_ingress SET history_version=0,history_event_id='',payload_sha256=? WHERE message_id=?",
                         (_owner_ingress_digest(payload, 0, ''), task.message_id))
    event = conversation.record(owner_user_id=owner.user_id, role='user', content='Synthetic apples.', source='test', source_message_id='one')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, tools=registry,
        inference=InferenceRouter(provider=Provider('conversation:' + event.event_id), default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=conversation, cancellation_checker=lambda: False)
    daemon = V2Daemon.__new__(V2Daemon)
    daemon.runtime = SimpleNamespace(queue=queue)
    daemon._owner_checkpoint_lock = threading.Lock()
    daemon._owner_checkpoints = {}
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'unused.db')),
        authorize_claim=authorize_trusted_owner_claim, save_checkpoint=lambda *_: pytest.fail('legacy checkpoint called'),
        atomic_checkpoints=store, selected_draft_sink=daemon.register_owner_checkpoint)
    result = processor.process(task, context)
    assert result.status == ProcessingStatus.WAITING, result.error
    if mode == 'source_revoked':
        assert conversation.delete(event_id=event.event_id, owner_user_id=owner.user_id)
    if mode == 'owner_deactivated':
        users.update_user(owner.user_id, is_active=False)
    with tempfile.TemporaryDirectory(prefix='owner-pick-', dir='/private/tmp') as socket_dir:
        server = V2DaemonServer(daemon, socket_path=os.path.join(socket_dir, 'p.sock'), experimental_owner_ingress=True)
        server.start()
        try:
            client = V2DaemonClient(server.socket_path)
            with pytest.raises(RuntimeError, match='owner_desktop_pickup_payload_denied'):
                client.request('experimental_owner_take_draft', message_id=task.message_id, user=owner.user_id)
            if mode in {'source_revoked', 'owner_deactivated'}:
                with pytest.raises(RuntimeError, match='owner_recovery_unavailable'):
                    client.request('experimental_owner_take_draft', message_id=task.message_id)
            else:
                picked = client.request('experimental_owner_take_draft', message_id=task.message_id)
                assert picked['message_id'] == task.message_id and picked['message'] == 'Synthetic apples.'
                assert picked['delivery_status'] == 'started_unconfirmed'
                token = picked['acknowledgement_token']
                with pytest.raises(RuntimeError, match='owner_desktop_pickup_unavailable'):
                    client.request('experimental_owner_take_draft', message_id=task.message_id)
                with pytest.raises(RuntimeError, match='owner_desktop_ack_unavailable'):
                    client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token='f' * 32)
                if mode == 'deactivated_ack':
                    users.update_user(owner.user_id, is_active=False)
                    with pytest.raises(RuntimeError, match='owner_claim_authorization_unavailable'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                elif mode == 'expired_ack':
                    with queue._connect() as conn:
                        conn.execute("UPDATE v2_inbound_messages SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE message_id=?", (task.message_id,))
                    with pytest.raises(RuntimeError, match='claim_expired'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                elif mode == 'reclaimed':
                    # Deliberately unsafe SQL fixture simulates an old worker
                    # replacing the generation under the same lease owner.
                    with queue._connect() as conn:
                        conn.execute("UPDATE v2_inbound_messages SET claim_generation=? WHERE message_id=?",
                                     ('a' * 32, task.message_id))
                    with pytest.raises(RuntimeError, match='claim_not_current'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                    assert [(item.role, item.content) for item in conversation.list(owner_user_id=owner.user_id)] == [('user', 'Find the apples note.'), ('user', 'Synthetic apples.')]
                elif mode == 'history_failure':
                    original = conversation.record
                    def fail_assistant(**kwargs):
                        if kwargs.get('role') == 'assistant':
                            raise ValueError('fixture_history_failure')
                        return original(**kwargs)
                    conversation.record = fail_assistant
                    with pytest.raises(RuntimeError, match='fixture_history_failure'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                    assert [(item.role, item.content) for item in conversation.list(owner_user_id=owner.user_id)] == [('user', 'Find the apples note.'), ('user', 'Synthetic apples.')]
                elif mode == 'legacy_handoff':
                    with queue._connect() as conn:
                        conn.execute('INSERT INTO v2_scheduled_handoffs(message_id,task_id,owner_id) VALUES (?,?,?)',
                                     (task.message_id, task.task_id, task.user))
                    with pytest.raises(RuntimeError, match='owner_desktop_ack_unavailable'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                    assert [(item.role, item.content) for item in conversation.list(owner_user_id=owner.user_id)] == [('user', 'Find the apples note.'), ('user', 'Synthetic apples.')]
                else:
                    assert client.request('experimental_owner_ack_draft', message_id=task.message_id,
                                          acknowledgement_token=token) == {'status': 'client_acknowledged'}
                    assert client.request('experimental_owner_ack_draft', message_id=task.message_id,
                                          acknowledgement_token=token) == {'status': 'client_acknowledged'}
                    # Idempotence uses the durable receipt, not an in-memory
                    # processor handoff, and is revoked with owner access.
                    assert task.message_id not in daemon._owner_checkpoints
                    restarted = V2Daemon.__new__(V2Daemon)
                    restarted.runtime = SimpleNamespace(queue=queue)
                    restarted._owner_checkpoint_lock = threading.Lock()
                    restarted._owner_checkpoints = {}
                    restarted.acknowledge_owner_desktop_draft(task.message_id, token)
                    events = conversation.list(owner_user_id=owner.user_id)
                    expected_users = ([('user', 'Synthetic apples.'), ('user', 'Find the apples note.')]
                                      if mode == 'legacy_origin' else
                                      [('user', 'Find the apples note.'), ('user', 'Synthetic apples.')])
                    assert [(event.role, event.content) for event in events] == [
                        *expected_users, ('assistant', 'Synthetic apples.')]
                    if mode == 'acknowledged':
                        with queue._connect() as conn:
                            conn.execute("UPDATE v2_conversation_events SET speaker_id='other' WHERE source_message_id=?",
                                         (f'inbound:{task.message_id}',))
                        with pytest.raises(ValueError, match='owner_desktop_ack_unavailable'):
                            restarted.acknowledge_owner_desktop_draft(task.message_id, token)
                        with queue._connect() as conn:
                            conn.execute('UPDATE v2_conversation_events SET speaker_id=? WHERE source_message_id=?',
                                         (owner.user_id, f'inbound:{task.message_id}'))
                    users.update_user(owner.user_id, is_active=False)
                    with pytest.raises(RuntimeError, match='owner_desktop_ack_unavailable'):
                        client.request('experimental_owner_ack_draft', message_id=task.message_id, acknowledgement_token=token)
                    # User data cleanup must also remove the retained owner
                    # ingress proof after the queue row was finalized.
                    assert daemon._delete_user_inbound_data(owner.user_id) == 0
                    with queue._connect() as cleanup_conn:
                        assert cleanup_conn.execute('SELECT 1 FROM v2_trusted_owner_ingress WHERE message_id=?',
                                                    (task.message_id,)).fetchone() is None
                with queue._connect() as conn:
                    raw = json.loads(store._row(conn, task.task_id)['task_state_json'])
                assert raw['metadata']['owner_read_desktop_pickup_started']['channel'] == 'desktop'
                assert raw['metadata']['owner_read_desktop_pickup_started']['status'] == (
                    'started_unconfirmed' if mode in {'expired_ack', 'deactivated_ack', 'reclaimed', 'history_failure', 'legacy_handoff'} else 'client_acknowledged')
                with queue._connect() as conn:
                    still_queued = conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (task.message_id,)).fetchone()
                assert (still_queued is None) == (mode in {'acknowledged', 'legacy_origin'})
        finally:
            server.stop()

def test_real_owner_processor_uses_atomic_checkpoint_and_persists_reads(setup):
    tmp_path, users, owner, queue, acquired, task, store, _ = setup
    conversation = SQLiteConversationStore(queue.db_path)
    event = conversation.record(owner_user_id=owner.user_id, role='user', content='Synthetic apples. Unique source 9fcb.', source='test', source_message_id='one')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    provider = Provider('conversation:' + event.event_id)
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=conversation, cancellation_checker=lambda: False)
    delivered_to_server = []
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'unused.db')),
        authorize_claim=authorize_trusted_owner_claim, save_checkpoint=lambda *_: pytest.fail('legacy checkpoint called'),
        atomic_checkpoints=store, selected_draft_sink=lambda checkpoint, context: delivered_to_server.append((checkpoint, context)))
    result = processor.process(task, context)
    assert result.status == ProcessingStatus.WAITING, result.error
    assert result.snapshot.metadata['draft'] == 'Synthetic apples.'
    assert len(delivered_to_server) == 1
    assert delivered_to_server[0][0].receipt == acquired.claim_receipt
    assert delivered_to_server[0][1] is context
    assert delivered_to_server[0][0].version > 0
    with queue._connect() as conn:
        row = store._row(conn, task.task_id)
        raw = json.loads(row['task_state_json'])
    state = raw['metadata']['single_loop_checkpoint']['state']
    observations = raw['metadata']['owner_read_observations']
    assert state['status'] == 'completed' and len(state['evidence_sources']) == 2
    assert set(observations) == set(state['evidence_sources'])
    assert all('observation' not in binding and len(binding['observation_sha256']) == 64
               for binding in observations.values())
    # The captured source text is not copied into durable evidence bindings.
    assert 'Synthetic apples.' not in json.dumps(observations)
    assert 'Unique source 9fcb.' not in row['task_state_json']
    assert raw['metadata']['single_loop_checkpoint']['budget']['reservations']
    assert 'owner_read_pending_action' not in raw['metadata']


def test_claim_loss_during_model_call_preserves_reservation_for_recovery(setup):
    tmp_path, users, _, queue, acquired, task, store, _ = setup
    class LostClaimProvider:
        supports_image_input = False
        def generate_json(self, request):
            with queue._connect() as conn:
                conn.execute("UPDATE v2_inbound_messages SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE message_id=?", (task.message_id,))
            decision = {'kind': 'answer', 'text': 'Synthetic.', 'tool_id': None, 'arguments': None}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 10, 'output_tokens': 5})
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, tools=registry,
        inference=InferenceRouter(provider=LostClaimProvider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=SQLiteConversationStore(tmp_path / 'conversation.db'), cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'unused.db')),
        authorize_claim=authorize_trusted_owner_claim, save_checkpoint=lambda *_: pytest.fail('legacy checkpoint called'),
        atomic_checkpoints=store)
    result = processor.process(task, context)
    assert result.status == ProcessingStatus.FAILED
    with queue._connect() as conn:
        raw = json.loads(store._row(conn, task.task_id)['task_state_json'])
    state = raw['metadata']['single_loop_checkpoint']['state']
    pending = raw['metadata']['owner_read_pending_action']
    assert state['inflight'] == pending['reservation_id'] and pending['kind'] == 'model'
    assert raw['metadata']['single_loop_checkpoint']['budget']['reservations'][state['inflight']]['status'] == 'reserved'


def test_claim_loss_immediately_before_dispatch_never_calls_provider(setup, monkeypatch):
    tmp_path, users, _, queue, acquired, task, store, _ = setup
    provider = Provider('conversation:unused')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt, tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=SQLiteConversationStore(tmp_path / 'conversation.db'), cancellation_checker=lambda: False)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'unused.db')),
        authorize_claim=authorize_trusted_owner_claim, save_checkpoint=lambda *_: pytest.fail('legacy checkpoint called'),
        atomic_checkpoints=store)
    monkeypatch.setattr(OwnerAtomicCheckpoint, 'check_claim', lambda *_: False)
    result = processor.process(task, context)
    assert result.status == ProcessingStatus.FAILED and provider.requests == []
    with queue._connect() as conn:
        raw = json.loads(store._row(conn, task.task_id)['task_state_json'])
    assert raw['metadata']['single_loop_checkpoint']['state']['inflight']


def test_terminal_draft_is_bound_to_exact_selected_decision_and_projection(setup):
    import hashlib
    _, _, _, queue, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = atomic.reserve(expected_version=start.version, reservation_id='model-selection', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    decision = {'kind': 'answer', 'text': 'selected fixture', 'usage': {'input_tokens': 3, 'output_tokens': 2}}
    projection = 'a' * 64
    atomic.settle_model(expected_version=reserved.version, reservation_id='model-selection', outcome='success',
                        input_tokens=3, output_tokens=2, decision=decision, projection_fingerprint=projection)
    state = CandidateState(task.task_id, task.user, 'goal', status='completed', response='selected fixture')
    with pytest.raises(CheckpointConflict, match='selected_decision_mismatch'):
        atomic.save_state(state, consume_decision=True, selected_decision={**decision, 'text': 'changed'},
                          projection_fingerprint=projection)
    with pytest.raises(CheckpointConflict, match='selected_decision_mismatch'):
        atomic.save_state(state, consume_decision=True, selected_decision=decision,
                          projection_fingerprint='b' * 64)
    result = atomic.save_state(state, consume_decision=True, selected_decision=decision,
                               projection_fingerprint=projection)
    with queue._connect() as conn:
        raw = json.loads(atomic.store._row(conn, task.task_id)['task_state_json'])
    binding = raw['metadata']['owner_read_selected_draft']
    assert binding['answer_sha256'] == hashlib.sha256(b'selected fixture').hexdigest()
    assert binding['projection_sha256'] == projection and binding['model_reservation_id'] == 'model-selection'
    assert binding['evidence_refs'] == [] and 'owner_read_pending_decision' not in raw['metadata']
    assert result.state.response == 'selected fixture'


def test_selected_tool_arguments_are_bound_to_model_decision(setup):
    _, _, _, _, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = atomic.reserve(expected_version=start.version, reservation_id='model-selection', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    decision = {'kind': 'tool', 'tool_id': 'native.read_memory_source',
                'arguments': {'source_id': 'conversation:fixture'}}
    projection = 'a' * 64
    settled = atomic.settle_model(expected_version=reserved.version, reservation_id='model-selection',
        outcome='success', input_tokens=3, output_tokens=2, decision=decision, projection_fingerprint=projection)
    tool_args = dict(expected_version=settled.version, reservation_id='read', kind='tool',
        amounts={'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}, now_ms=102,
        tool_id='native.read_memory_source', selected_decision=decision, projection_fingerprint=projection)
    with pytest.raises(CheckpointConflict, match='pending_decision_unresolved'):
        atomic.reserve(**tool_args, arguments={'source_id': 'conversation:other'})
    result = atomic.reserve(**tool_args, arguments=decision['arguments'])
    assert result.state.inflight == 'read'


def test_bounded_activity_intent_is_bound_to_decision_and_removed_on_settlement(setup):
    _, _, _, queue, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserved = atomic.reserve(expected_version=start.version, reservation_id='model-intent', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=101)
    args = {'source_id': 'conversation:fixture'}
    decision = {'kind': 'tool', 'tool_id': 'native.read_memory_source', 'arguments': args,
                'text': None, 'activity_intent': 'find_information'}
    projection = 'a' * 64
    settled = atomic.settle_model(expected_version=reserved.version, reservation_id='model-intent',
        outcome='success', input_tokens=3, output_tokens=2, decision=decision, projection_fingerprint=projection)
    kwargs = dict(expected_version=settled.version, reservation_id='read-intent', kind='tool',
        amounts={'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}, now_ms=102,
        tool_id='native.read_memory_source', arguments=args, projection_fingerprint=projection)
    with pytest.raises(CheckpointConflict, match='pending_decision_unresolved'):
        atomic.reserve(**kwargs, selected_decision={**decision, 'activity_intent': 'make_change'})
    tool = atomic.reserve(**kwargs, selected_decision=decision)
    with queue._connect() as conn:
        raw = json.loads(atomic.store._row(conn, task.task_id)['task_state_json'])
    marker = raw['metadata']['owner_activity_intent']
    assert marker['intent'] == 'find_information' and marker['reservation_id'] == 'read-intent'
    assert marker['actor'] == task.user and marker['task_id'] == task.task_id
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_work_log import owner_work_log
    assert owner_work_log(queue, task.message_id)['activity']['pending_intent'] == 'find_information'
    result = atomic.settle_read(expected_version=tool.version, reservation_id='read-intent',
        tool_id='native.read_memory_source', arguments=args, evidence_ref='e1', observation={'text': 'private'})
    assert result.state.inflight == ''
    with queue._connect() as conn:
        raw = json.loads(atomic.store._row(conn, task.task_id)['task_state_json'])
    assert 'owner_activity_intent' not in raw['metadata']


def test_explicit_failed_request_binds_exact_owner_and_revalidates_history(setup):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_failure_diagnostics import load_bound_failed_request
    _, _, _, queue, _, task, store, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    state = start.state
    state.status, state.blocker = 'blocked', 'candidate_model_timeout'
    atomic.save_state(state)
    first = task.message_id
    # Two independent owner requests have exactly the same user-visible text.
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET status='failed' WHERE message_id=?", (first,))
    second = queue.enqueue_trusted_local_owner(prompt='Find the apples note.', peer_uid=os.getuid())
    acquired2 = queue.claim_next()
    task2 = TaskState.from_message(acquired2.message, message_id=second)
    task2.task_id = second
    atomic2 = OwnerAtomicCheckpoint(queue, store, task2, acquired2.claim_receipt,
                                    clock=lambda: datetime.now(timezone.utc))
    state2 = atomic2.initialize(CandidateState(second, task.user, 'goal'), make_budget()).state
    state2.status, state2.blocker = 'blocked', 'candidate_model_unavailable'
    atomic2.save_state(state2)
    followup = queue.enqueue_trusted_local_owner(prompt='Why did this fail?', peer_uid=os.getuid(),
                                                 failed_request_message_id=first)
    with queue._connect() as conn:
        row = conn.execute('SELECT metadata_json FROM v2_inbound_messages WHERE message_id=?', (followup,)).fetchone()
        binding = json.loads(row['metadata_json'])['owner_failed_request']
        conn.execute('BEGIN')
        diagnostic = load_bound_failed_request(conn, task.user, binding)
        assert diagnostic['message_id'] == first and diagnostic['message_id'] != second
        assert diagnostic['description'] == 'The model provider timed out.'
        with pytest.raises(ValueError, match='owner_failure_reference_unavailable'):
            load_bound_failed_request(conn, 'other-owner', binding)
        conn.rollback()
        conn.execute('DELETE FROM v2_conversation_events WHERE source_message_id=?', ('inbound:' + first,))
        conn.commit()
        conn.execute('BEGIN')
        with pytest.raises(ValueError, match='owner_failure_reference_unavailable'):
            load_bound_failed_request(conn, task.user, binding)
        conn.rollback()
    with pytest.raises(ValueError, match='owner_failure_reference_unavailable'):
        queue.enqueue_trusted_local_owner(prompt='Why?', peer_uid=os.getuid(), failed_request_message_id=first)
    with pytest.raises(ValueError, match='owner_failure_reference_unavailable'):
        queue.enqueue_trusted_local_owner(prompt='Why?', peer_uid=os.getuid(),
            failed_request_message_id=second, reply_to_message_id=second)


def test_failed_reference_ipc_rejection_is_definite_and_does_not_enqueue(setup):
    from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
    _, _, _, queue, _, _, _, _ = setup
    daemon = SimpleNamespace(runtime=SimpleNamespace(queue=queue), experimental_owner_only=False)
    with tempfile.TemporaryDirectory(prefix='owner-reference-', dir='/private/tmp') as socket_dir:
        server = V2DaemonServer(daemon, socket_path=os.path.join(socket_dir, 'p.sock'), experimental_owner_ingress=True)
        server.start()
        try:
            with queue._connect() as conn:
                before = conn.execute('SELECT COUNT(*) FROM v2_inbound_messages').fetchone()[0]
            with pytest.raises(RuntimeError, match='^ValueError: owner_failure_reference_unavailable$'):
                V2DaemonClient(server.socket_path).request('experimental_owner_ingress',
                    prompt='Explain this.', failed_request_message_id='a' * 32)
            with queue._connect() as conn:
                assert conn.execute('SELECT COUNT(*) FROM v2_inbound_messages').fetchone()[0] == before
        finally:
            server.stop()


def test_lost_inflight_clock_can_be_explicitly_abandoned(setup):
    _, _, _, queue, _, task, _, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    reserve_selected_tool(atomic, start, 'native.owner_clock', {})
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET lease_expires_at='2000-01-01T00:00:00+00:00' WHERE message_id=?", (task.message_id,))
    assert queue.abandon_trusted_owner_message(task.message_id) == {
        'status': 'abandoned', 'delivery_status': 'no_pickup_recorded'}
