import os
from datetime import datetime, timedelta, timezone

import pytest

from test_owner_atomic_checkpoint import setup, make_budget, reserve_selected_tool
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_detached import open_detached_paused_checkpoint
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_question import start_question, acknowledge_question, record_choice
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState


def test_paused_question_can_be_acknowledged_after_lease_expiry(setup):
    _, _, _, queue, _, task, store, atomic = setup
    state = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'source_id': 'conversation:fixture'}
    reserved = reserve_selected_tool(atomic, state, 'native.read_memory_source', args)
    settled = atomic.settle_read(expected_version=reserved.version, reservation_id='read-1',
        tool_id='native.read_memory_source', arguments=args, evidence_ref='e1', observation={'text':'x'})
    paused = atomic.pause_for_continuation(expected_version=settled.version, metric='actions', interval=1,
        last_checkpoint=0, cancellation_checker=lambda:False)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_inbound_messages SET lease_expires_at=? WHERE message_id=?',
                     ((datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(),task.message_id))
    detached = open_detached_paused_checkpoint(queue, store, task.message_id)
    assert detached.version == paused.version
    version, token = start_question(detached, expected_version=detached.version, question='Continue?',
                                    cancellation_checker=lambda:False)
    version = acknowledge_question(detached, expected_version=version, token=token, cancellation_checker=lambda:False)
    record_choice(detached, expected_version=version, peer_uid=os.getuid(), approve=True,
                  cancellation_checker=lambda:False)


def test_detached_requires_paused_checkpoint(setup):
    _, _, _, queue, _, task, store, atomic = setup
    atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    with pytest.raises(ValueError):
        open_detached_paused_checkpoint(queue, store, task.message_id)


def test_persisted_cancellation_veto_survives_reopen_and_blocks_pause_choice(setup):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_detached import persist_continuation_cancellation
    _, _, _, queue, _, task, store, atomic = setup
    state = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'source_id':'conversation:fixture'}
    reserved = reserve_selected_tool(atomic, state, 'native.read_memory_source', args)
    settled = atomic.settle_read(expected_version=reserved.version, reservation_id='read-1',
        tool_id='native.read_memory_source', arguments=args, evidence_ref='e1', observation={'text':'x'})
    atomic.pause_for_continuation(expected_version=settled.version, metric='actions', interval=1,
                                  last_checkpoint=0, cancellation_checker=lambda:False)
    persist_continuation_cancellation(queue, task.message_id, task.user)
    with pytest.raises(ValueError):
        open_detached_paused_checkpoint(queue, store, task.message_id)
    # A fresh connection observes the durable veto, independently of the
    # process-local cancellation checker.
    with queue._connect() as conn:
        assert conn.execute('SELECT owner_id FROM v2_owner_continuation_cancellations WHERE message_id=?',
                            (task.message_id,)).fetchone()[0] == task.user
