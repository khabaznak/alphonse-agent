"""Bash evidence remains valid through a crash and another approved window."""
from datetime import datetime, timedelta, timezone
import os

from test_owner_atomic_checkpoint import setup
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_activation import activate_held_same_claim
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_grant import issue_grant, prepare_handoff
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_question import start_question, acknowledge_question, record_choice
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_reclaim import reclaim_quiescent_activated
from alphonse.agent_v2.core.intelligence.simple_loop.owner_host_bash import dispatch_owner_host_bash
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import restore_owner_read_observations
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import BudgetLimits, TaskBudget


def test_original_bash_survives_crash_and_second_approved_window(setup):
    root, _, owner, queue, acquired, task, store, atomic = setup
    clock = lambda: datetime.now(timezone.utc)
    atomic.clock = clock
    vector = lambda a,m,i,o: {'actions':a,'model_calls':m,'input_tokens':i,'output_tokens':o}
    now_ms = lambda: int(clock().timestamp() * 1000)
    budget = TaskBudget(BudgetLimits(vector(4,6,10000,1000), vector(0,1,100,50), 100000, 0), started_at_ms=now_ms())
    start = atomic.initialize(CandidateState(task.task_id, owner.user_id, 'goal'), budget)
    ctx = CoreLoopContext(messages=queue, claim_receipt=acquired.claim_receipt,
        conversation_store=SQLiteConversationStore(queue.db_path), cancellation_checker=lambda: False)
    ctx._owner_host_bash_enabled = True

    def bash(checkpoint, context, version, reservation_id, args):
        model_id = 'model-' + reservation_id
        decision = {'kind':'tool', 'tool_id':'native.bash', 'arguments':args}
        reserved = checkpoint.reserve(expected_version=version, reservation_id=model_id, kind='model',
            amounts=vector(0,1,100,50), now_ms=now_ms())
        selected = checkpoint.settle_model(expected_version=reserved.version, reservation_id=model_id,
            outcome='success', input_tokens=1, output_tokens=1, decision=decision, projection_fingerprint='a'*64)
        checkpoint.reserve(expected_version=selected.version, reservation_id=reservation_id, kind='tool',
            tool_id='native.bash', arguments=args, selected_decision=decision, projection_fingerprint='a'*64,
            amounts=vector(1,0,0,0), now_ms=now_ms())
        authorize = lambda actor, task_id, actual: actor == owner.user_id and task_id == task.task_id and actual == args
        dispatch_owner_host_bash(atomic=checkpoint, context=context, arguments=args,
                                 reservation_id=reservation_id, authorize=authorize)
        return checkpoint.settle_host_bash_return(expected_version=checkpoint.version, reservation_id=reservation_id,
            arguments=args, evidence_ref='evidence-' + reservation_id, authorize=authorize, cancellation_checker=lambda: False)

    def approve(checkpoint, context, version, previous):
        paused = checkpoint.pause_for_continuation(expected_version=version, metric='actions', interval=1,
                                                   last_checkpoint=previous, cancellation_checker=lambda: False)
        version, token = start_question(checkpoint, expected_version=paused.version, question='Continue?', cancellation_checker=lambda: False)
        version = acknowledge_question(checkpoint, expected_version=version, token=token, cancellation_checker=lambda: False)
        version = record_choice(checkpoint, expected_version=version, peer_uid=os.getuid(), approve=True, cancellation_checker=lambda: False)
        timestamp = now_ms()
        version = issue_grant(checkpoint, expected_version=version, now_ms=timestamp, lifetime_ms=60000, cancellation_checker=lambda: False)
        version = prepare_handoff(checkpoint, expected_version=version, now_ms=timestamp, cancellation_checker=lambda: False)
        return activate_held_same_claim(checkpoint, expected_version=version, context=context, cancellation_checker=lambda: False)

    first = bash(atomic, ctx, start.version, 'write', {'command':'printf x >> record; cat record', 'cwd':str(root)})
    approve(atomic, ctx, first.version, 0)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_inbound_messages SET lease_expires_at=? WHERE message_id=?',
            ((clock()-timedelta(seconds=1)).isoformat(), task.message_id))
    reclaimed, version = reclaim_quiescent_activated(queue, store, task.message_id, lease_owner='restart-one',
                                                       cancellation_checker=lambda: False)
    fresh = OwnerAtomicCheckpoint(queue, store, task, reclaimed.claim_receipt, clock=clock)
    fresh.version = version
    fresh_ctx = CoreLoopContext(messages=queue, claim_receipt=reclaimed.claim_receipt,
        conversation_store=SQLiteConversationStore(queue.db_path), cancellation_checker=lambda: False)
    fresh_ctx._owner_host_bash_enabled = True
    second = bash(fresh, fresh_ctx, version, 'read', {'command':'cat record', 'cwd':str(root)})
    resumed = approve(fresh, fresh_ctx, second.version, 1)
    _, evidence = restore_owner_read_observations(fresh, fresh_ctx, expected_version=resumed.version)
    assert evidence['evidence-write']['observation']['stdout'] == 'x'
    assert evidence['evidence-read']['observation']['stdout'] == 'x'
    assert (root / 'record').read_text() == 'x'
    with queue._connect() as conn:
        conn.execute('UPDATE v2_inbound_messages SET lease_expires_at=? WHERE message_id=?',
            ((clock()-timedelta(seconds=1)).isoformat(), task.message_id))
    second_recovery, version = reclaim_quiescent_activated(queue, store, task.message_id,
        lease_owner='restart-two', cancellation_checker=lambda: False)
    latest = OwnerAtomicCheckpoint(queue, store, task, second_recovery.claim_receipt, clock=clock)
    latest_ctx = CoreLoopContext(messages=queue, claim_receipt=second_recovery.claim_receipt,
        conversation_store=SQLiteConversationStore(queue.db_path), cancellation_checker=lambda: False)
    latest_ctx._owner_host_bash_enabled = True
    _, evidence = restore_owner_read_observations(latest, latest_ctx, expected_version=version)
    assert evidence['evidence-write']['observation']['stdout'] == 'x'
    assert (root / 'record').read_text() == 'x'
