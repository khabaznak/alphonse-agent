"""Disposable same-claim checkpoint experiment; no daemon or model involved."""
from datetime import datetime, timezone
import os

import pytest

from test_owner_atomic_checkpoint import setup, make_budget, reserve_selected_tool
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.simple_loop.owner_host_bash import dispatch_owner_host_bash
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_question import start_question, acknowledge_question, record_choice
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_grant import issue_grant, prepare_handoff
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_activation import activate_held_same_claim
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState


def test_disposable_bash_effect_is_not_replayed_during_same_claim_continuation(setup):
    root, _, owner, queue, acquired, task, _, atomic = setup
    # Match the test budget's synthetic clock and keep the claim current.
    atomic.clock = lambda: datetime.fromtimestamp(0.102, tz=timezone.utc)
    start = atomic.initialize(CandidateState(task.task_id, owner.user_id, 'goal'), make_budget())
    args = {'command': 'printf x >> result.txt; cat result.txt', 'cwd': str(root)}
    reserve_selected_tool(atomic, start, 'native.bash', args)
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt,
        conversation_store=SQLiteConversationStore(queue.db_path), cancellation_checker=lambda: False)
    context._owner_host_bash_enabled = True
    authorize = lambda actor, task_id, command: actor == owner.user_id and task_id == task.task_id and command == args
    result = dispatch_owner_host_bash(atomic=atomic, context=context, arguments=args,
                                      reservation_id='read-1', authorize=authorize)
    assert result['stdout'] == 'x' and (root / 'result.txt').read_text() == 'x'
    settled = atomic.settle_host_bash_return(expected_version=atomic.version, reservation_id='read-1',
        arguments=args, evidence_ref='host-output', authorize=authorize, cancellation_checker=lambda: False)
    paused = atomic.pause_for_continuation(expected_version=settled.version, metric='actions', interval=1,
                                            last_checkpoint=0, cancellation_checker=lambda: False)
    version, token = start_question(atomic, expected_version=paused.version, question='Continue?',
                                    cancellation_checker=lambda: False)
    version = acknowledge_question(atomic, expected_version=version, token=token, cancellation_checker=lambda: False)
    version = record_choice(atomic, expected_version=version, peer_uid=os.getuid(), approve=True,
                            cancellation_checker=lambda: False)
    version = issue_grant(atomic, expected_version=version, now_ms=102, lifetime_ms=100,
                          cancellation_checker=lambda: False)
    version = prepare_handoff(atomic, expected_version=version, now_ms=102, cancellation_checker=lambda: False)
    activated = activate_held_same_claim(atomic, expected_version=version, context=context,
                                        cancellation_checker=lambda: False)
    assert activated.state.evidence_sources == ['host-output']
    assert (root / 'result.txt').read_text() == 'x'
    with pytest.raises(PermissionError):
        dispatch_owner_host_bash(atomic=atomic, context=context, arguments=args,
                                 reservation_id='read-1', authorize=authorize)
    assert (root / 'result.txt').read_text() == 'x'
    read_args = {'command': 'cat result.txt', 'cwd': str(root)}
    decision = {'kind': 'tool', 'tool_id': 'native.bash', 'arguments': read_args}
    projection = 'a' * 64
    reserved = atomic.reserve(expected_version=activated.version, reservation_id='model-2', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=103)
    selected = atomic.settle_model(expected_version=reserved.version, reservation_id='model-2',
        outcome='success', input_tokens=1, output_tokens=1, decision=decision, projection_fingerprint=projection)
    atomic.reserve(expected_version=selected.version, reservation_id='read-2', kind='tool', tool_id='native.bash',
        arguments=read_args, selected_decision=decision, projection_fingerprint=projection,
        amounts={'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}, now_ms=104)
    read = dispatch_owner_host_bash(atomic=atomic, context=context, arguments=read_args, reservation_id='read-2',
        authorize=lambda actor, task_id, command: actor == owner.user_id and task_id == task.task_id and command == read_args)
    assert read['stdout'] == 'x' and (root / 'result.txt').read_text() == 'x'


def test_disposable_bash_recovered_under_fresh_claim_without_replay(setup):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_reclaim import reclaim_approved_pause
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
    root, _, owner, queue, acquired, task, store, atomic = setup
    clock = lambda: datetime.now(timezone.utc)
    atomic.clock = clock
    start = atomic.initialize(CandidateState(task.task_id, owner.user_id, 'goal'), make_budget())
    args = {'command':'printf x >> result.txt; cat result.txt', 'cwd':str(root)}
    reserve_selected_tool(atomic, start, 'native.bash', args)
    context = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt,
        conversation_store=SQLiteConversationStore(queue.db_path), cancellation_checker=lambda:False)
    context._owner_host_bash_enabled = True
    auth = lambda actor, task_id, command: actor == owner.user_id and task_id == task.task_id and command == args
    dispatch_owner_host_bash(atomic=atomic, context=context, arguments=args, reservation_id='read-1', authorize=auth)
    settled = atomic.settle_host_bash_return(expected_version=atomic.version, reservation_id='read-1',
        arguments=args, evidence_ref='host-output', authorize=auth, cancellation_checker=lambda:False)
    paused = atomic.pause_for_continuation(expected_version=settled.version, metric='actions', interval=1,
        last_checkpoint=0, cancellation_checker=lambda:False)
    increment = {'actions':1,'model_calls':2,'input_tokens':100,'output_tokens':50}
    window = {'increment':increment, 'increment_caps':increment,
        'elapsed_increment_ms':1000, 'elapsed_increment_cap_ms':1000,
        'total_caps':{'actions':4,'model_calls':5,'input_tokens':1100,'output_tokens':350},
        'total_elapsed_cap_ms':101000}
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_question import render_window_question
    with atomic._transaction() as conn:
        _, raw, _, _ = atomic._load(conn, paused.version, allow_paused=True)
        question = render_window_question(raw['metadata']['owner_continuation_pause'], window)
    version, token = start_question(atomic, expected_version=paused.version, question=question, cancellation_checker=lambda:False)
    version = acknowledge_question(atomic, expected_version=version, token=token, cancellation_checker=lambda:False)
    version = record_choice(atomic, expected_version=version, peer_uid=os.getuid(), approve=True, cancellation_checker=lambda:False)
    now_ms = int(clock().timestamp() * 1000)
    version = issue_grant(atomic, expected_version=version, now_ms=now_ms, lifetime_ms=60000, cancellation_checker=lambda:False)
    prepare_handoff(atomic, expected_version=version, now_ms=now_ms, cancellation_checker=lambda:False)
    new, version = reclaim_approved_pause(queue, store, task.message_id, lease_owner='restarted', clock=clock,
                                           cancellation_checker=lambda:False)
    fresh = OwnerAtomicCheckpoint(queue, store, task, new.claim_receipt, clock=clock)
    fresh.version = version
    fresh_context = CoreLoopContext(messages=queue, claim_receipt=new.claim_receipt,
        conversation_store=SQLiteConversationStore(queue.db_path), cancellation_checker=lambda:False)
    fresh_context._owner_host_bash_enabled = True
    fresh_context.owner_continuation_budget_window = window
    activated = activate_held_same_claim(fresh, expected_version=version, context=fresh_context,
                                        cancellation_checker=lambda:False)
    assert activated.state.evidence_sources == ['host-output']
    assert activated.budget.limits.total['model_calls'] == 5
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import restore_owner_read_observations
    _, rebound = restore_owner_read_observations(fresh, fresh_context, expected_version=activated.version)
    assert rebound['host-output']['observation']['stdout'] == 'x'
    assert (root / 'result.txt').read_text() == 'x'
    with pytest.raises(Exception):
        atomic.reserve(expected_version=atomic.version, reservation_id='stale', kind='model',
            amounts={'actions':0,'model_calls':1,'input_tokens':100,'output_tokens':50}, now_ms=103)
    # Simulate a crash after activation, then expiry of its claim. Recovery
    # must retain the original Bash output without replaying the command.
    from datetime import timedelta
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_reclaim import reclaim_quiescent_activated
    with queue._connect() as conn:
        conn.execute('UPDATE v2_inbound_messages SET lease_expires_at=? WHERE message_id=?',
            ((datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(), task.message_id))
    after_crash, version = reclaim_quiescent_activated(queue, store, task.message_id,
        lease_owner='recovered-again', cancellation_checker=lambda:False)
    after_atomic = OwnerAtomicCheckpoint(queue, store, task, after_crash.claim_receipt, clock=clock)
    after_context = CoreLoopContext(messages=queue, claim_receipt=after_crash.claim_receipt,
        conversation_store=SQLiteConversationStore(queue.db_path), cancellation_checker=lambda:False)
    after_context._owner_host_bash_enabled = True
    _, rebound = restore_owner_read_observations(after_atomic, after_context, expected_version=version)
    assert rebound['host-output']['observation']['stdout'] == 'x'
    assert (root / 'result.txt').read_text() == 'x'
