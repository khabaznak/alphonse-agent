"""Disposable tests for the inactive continuation-question records."""
import json
import os

import pytest

from test_owner_atomic_checkpoint import setup, make_budget, reserve_selected_tool
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_question import (
    start_question, acknowledge_question, record_choice,
)
from alphonse.agent_v2.core.intelligence.simple_loop.owner_continuation_grant import issue_grant
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import CheckpointConflict


def paused_checkpoint(setup):
    _, _, _, queue, acquired, task, store, atomic = setup
    start = atomic.initialize(CandidateState(task.task_id, task.user, 'goal'), make_budget())
    args = {'source_id': 'conversation:fixture'}
    reserved = reserve_selected_tool(atomic, start, 'native.read_memory_source', args)
    settled = atomic.settle_read(expected_version=reserved.version, reservation_id='read-1',
        tool_id='native.read_memory_source', arguments=args, evidence_ref='e1', observation={'text': 'x'})
    paused = atomic.pause_for_continuation(expected_version=settled.version, metric='actions', interval=1,
        last_checkpoint=0, cancellation_checker=lambda: False)
    return queue, acquired, task, store, atomic, paused.version


def test_legacy_v1_pause_can_still_be_acknowledged(setup):
    queue, _, task, _, atomic, version = paused_checkpoint(setup)
    with queue._connect() as conn:
        row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()
        raw = json.loads(row['task_state_json'])
        pause = raw['metadata']['owner_continuation_pause']
        pause['schema_version'] = 1
        del pause['progress_assessment']
        del pause['evidence_refs']
        conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?',
                     (json.dumps(raw), task.task_id))
    started, token = start_question(atomic, expected_version=version, question='Continue?',
                                    cancellation_checker=lambda: False)
    acknowledged = acknowledge_question(atomic, expected_version=started, token=token, cancellation_checker=lambda: False)
    choice = record_choice(atomic, expected_version=acknowledged, peer_uid=os.getuid(), approve=True,
                           cancellation_checker=lambda: False)
    issue_grant(atomic, expected_version=choice, now_ms=1000, lifetime_ms=100, cancellation_checker=lambda: False)


def test_choice_requires_acknowledged_exact_question_and_is_not_resume(setup):
    queue, _, task, _, atomic, version = paused_checkpoint(setup)
    version, token = start_question(atomic, expected_version=version, question='Continue this task?',
                                    cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict):
        record_choice(atomic, expected_version=version, peer_uid=os.getuid(), approve=True,
                      cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict):
        acknowledge_question(atomic, expected_version=version, token='wrong', cancellation_checker=lambda: False)
    version = acknowledge_question(atomic, expected_version=version, token=token, cancellation_checker=lambda: False)
    with pytest.raises(ValueError):
        record_choice(atomic, expected_version=version, peer_uid=os.getuid() + 1, approve=True,
                      cancellation_checker=lambda: False)
    choice_version = record_choice(atomic, expected_version=version, peer_uid=os.getuid(), approve=True,
                                   cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict):
        record_choice(atomic, expected_version=choice_version, peer_uid=os.getuid(), approve=True,
                      cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict, match='owner_continuation_paused'):
        atomic.reserve(expected_version=choice_version, reservation_id='no-resume', kind='model',
            amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 50}, now_ms=102)
    with queue._connect() as conn:
        assert conn.execute('SELECT status FROM v2_inbound_messages WHERE message_id=?',
                            (task.message_id,)).fetchone()['status'] == 'processing'


def test_choice_rejects_stale_version_cancelled_and_duplicate_question(setup):
    _, _, _, _, atomic, version = paused_checkpoint(setup)
    started, token = start_question(atomic, expected_version=version, question='Continue?', cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict):
        start_question(atomic, expected_version=started, question='Continue again?', cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict):
        acknowledge_question(atomic, expected_version=version, token=token, cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict):
        acknowledge_question(atomic, expected_version=started, token=token, cancellation_checker=lambda: True)
    acknowledged = acknowledge_question(atomic, expected_version=started, token=token, cancellation_checker=lambda: False)
    with pytest.raises(CheckpointConflict):
        record_choice(atomic, expected_version=acknowledged, peer_uid=os.getuid(), approve=True,
                      cancellation_checker=lambda: True)
    record_choice(atomic, expected_version=acknowledged, peer_uid=os.getuid(), approve=False,
                  cancellation_checker=lambda: False)


def test_cancellation_during_question_write_rolls_back(setup):
    queue, _, task, _, atomic, version = paused_checkpoint(setup)
    calls = iter([False, True])
    with pytest.raises(CheckpointConflict):
        start_question(atomic, expected_version=version, question='Continue?',
                       cancellation_checker=lambda: next(calls))
    with queue._connect() as conn:
        row = conn.execute('SELECT version,task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                           (task.task_id,)).fetchone()
    assert row['version'] == version
    assert 'owner_continuation_question' not in json.loads(row['task_state_json'])['metadata']


def test_changed_question_cannot_be_acknowledged(setup):
    queue, _, task, _, atomic, version = paused_checkpoint(setup)
    version, token = start_question(atomic, expected_version=version, question='Continue?', cancellation_checker=lambda: False)
    with queue._connect() as conn:
        row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()
        raw = json.loads(row['task_state_json'])
        raw['metadata']['owner_continuation_question']['question'] = 'Changed?'
        conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?',
                     (json.dumps(raw), task.task_id))
    with pytest.raises(CheckpointConflict):
        acknowledge_question(atomic, expected_version=version, token=token, cancellation_checker=lambda: False)


@pytest.mark.parametrize('approved,tamper', [(False, None), (True, 'question'), (True, 'choice')])
def test_grant_rejects_declined_or_modified_acknowledged_choice(setup, approved, tamper):
    queue, _, task, _, atomic, version = paused_checkpoint(setup)
    version, token = start_question(atomic, expected_version=version, question='Continue?', cancellation_checker=lambda: False)
    version = acknowledge_question(atomic, expected_version=version, token=token, cancellation_checker=lambda: False)
    version = record_choice(atomic, expected_version=version, peer_uid=os.getuid(), approve=approved,
                            cancellation_checker=lambda: False)
    if tamper:
        with queue._connect() as conn:
            row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?', (task.task_id,)).fetchone()
            raw = json.loads(row['task_state_json'])
            if tamper == 'question':
                raw['metadata']['owner_continuation_question']['question'] = 'Changed?'
            else:
                raw['metadata']['owner_continuation_choice']['owner_id'] = 'different-owner'
            conn.execute('UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id=?',
                         (json.dumps(raw), task.task_id))
    with pytest.raises(CheckpointConflict):
        issue_grant(atomic, expected_version=version, now_ms=100, lifetime_ms=1000,
                    cancellation_checker=lambda: False)
