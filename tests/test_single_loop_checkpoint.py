"""Transactional checkpoint prototype against disposable runtime SQLite stores."""
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import (
    ENGINE, STATE_KEY, CheckpointConflict, ExperimentalCheckpointStore,
)
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetExhausted, BudgetLimits, TaskBudget
from alphonse.agent_v2.core.questions import SQLiteQuestionStore


def v(actions=0, model_calls=0, input_tokens=0, output_tokens=0):
    return dict(actions=actions, model_calls=model_calls, input_tokens=input_tokens, output_tokens=output_tokens)


@pytest.fixture
def setup(tmp_path):
    questions = SQLiteQuestionStore(tmp_path / 'runtime.sqlite3')
    task = TaskState(task_id='task', user='alice', goal='find value', intelligence_engine=ENGINE,
                     intelligence_schema_version=1,
                     metadata={'intelligence_engine': ENGINE, 'unrelated': {'preserve': 'yes'}})
    version = questions.save_task_checkpoint(task, status='running')
    state = CandidateState('task', 'alice', 'goal:1')
    budget = TaskBudget(BudgetLimits(v(2, 3, 1000, 300), v(0, 1, 100, 50), 10000, 1000), started_at_ms=1000)
    return questions, ExperimentalCheckpointStore(questions), version, state, budget


def init(setup):
    _, store, version, state, budget = setup
    return store.initialize('task', 'alice', expected_version=version, state=state, budget=budget)


def test_initialize_reserve_settle_atomic_and_preserve_task_fields(setup):
    questions, store, _, _, _ = setup
    first = init(setup)
    assert first.version == 2
    run = store.reserve('task', 'alice', expected_version=first.version, reservation_id='model1',
                        kind='model', amounts=v(model_calls=1, input_tokens=100, output_tokens=30), now_ms=1100)
    restored = store.load('task', 'alice', expected_version=run.version)
    assert restored.state.inflight == 'model1'
    assert restored.budget.usage()['in_flight'] == ['model1']
    finished = store.settle('task', 'alice', expected_version=run.version, reservation_id='model1',
                            outcome='success', input_tokens=90, output_tokens=20)
    assert finished.state.inflight == ''
    assert finished.state.evidence_sources == []
    assert finished.budget.usage()['charged']['model_calls'] == 1
    saved = questions.load_task_checkpoint('task')
    assert saved.goal == 'find value'
    assert saved.metadata['unrelated'] == {'preserve': 'yes'}
    assert saved.intelligence_engine == ENGINE


def test_concurrent_workers_cannot_reserve_from_same_version(setup):
    _, store, _, _, _ = setup
    first = init(setup)
    def worker(identifier):
        try:
            return store.reserve('task', 'alice', expected_version=first.version, reservation_id=identifier,
                                 kind='tool', amounts=v(actions=1), now_ms=1100).state.inflight
        except CheckpointConflict:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(worker, ('one', 'two')))
    assert outcomes.count('conflict') == 1
    winner = next(item for item in outcomes if item != 'conflict')
    assert store.load('task', 'alice', expected_version=3).state.inflight == winner


def test_crash_before_settlement_keeps_reservation_and_never_replays(setup):
    _, store, _, _, _ = setup
    first = init(setup)
    reserved = store.reserve('task', 'alice', expected_version=first.version, reservation_id='read',
                             kind='tool', amounts=v(actions=1), now_ms=1100)
    # New host reads the same durable row after a crash during external I/O.
    recovered = ExperimentalCheckpointStore(setup[0]).load('task', 'alice', expected_version=reserved.version)
    assert recovered.state.inflight == 'read'
    with pytest.raises(CheckpointConflict, match='reconciliation'):
        store.reserve('task', 'alice', expected_version=recovered.version, reservation_id='another',
                      kind='tool', amounts=v(actions=1), now_ms=1200)
    # Missing independently persisted evidence cannot clear the in-flight marker.
    with pytest.raises(CheckpointConflict, match='verified_evidence_integration_required'):
        store.settle('task', 'alice', expected_version=recovered.version, reservation_id='read', outcome='success')
    assert store.load('task', 'alice', expected_version=recovered.version).state.inflight == 'read'


def test_arbitrary_or_forged_tool_receipt_cannot_clear_inflight(setup):
    _, store, _, _, _ = setup
    first = init(setup)
    reserved = store.reserve('task', 'alice', expected_version=first.version, reservation_id='read',
                             kind='tool', amounts=v(actions=1), now_ms=1100)
    with pytest.raises(CheckpointConflict, match='verified_evidence_integration_required'):
        store.settle('task', 'alice', expected_version=reserved.version, reservation_id='read',
                     outcome='success', evidence_ref='forged-evidence')
    assert store.load('task', 'alice', expected_version=reserved.version).state.inflight == 'read'


@pytest.mark.parametrize('matching_task,matching_audience,delivered', [
    (True, True, True), (True, False, True), (False, True, True), (True, True, False),
])
def test_existing_outbox_receipt_cannot_prove_an_unbound_action(setup, matching_task, matching_audience, delivered):
    """A genuine row in the shared DB is not proof of this particular action."""
    from alphonse.agent_v2.core.io.channels import ChannelAddress
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    questions, store, _, _, _ = setup
    first = init(setup)
    reserved = store.reserve('task', 'alice', expected_version=first.version, reservation_id='read',
                             kind='tool', amounts=v(actions=1), now_ms=1100)
    outbox = SQLiteOutboundStore(questions.db_path)
    record = outbox.enqueue(
        address=ChannelAddress('fixture', 'telegram', 'fixture-chat', alphonse_user_id='alice'),
        message='synthetic message', task_id='task' if matching_task else 'other-task',
        audience_user_id='alice' if matching_audience else 'bob',
        metadata={'occurrence_key': 'synthetic-occurrence'},
    )
    if delivered:
        assert outbox.mark_delivered(record.outbox_message_id, provider_message_id='fixture-provider-id')
    with pytest.raises(CheckpointConflict, match='verified_evidence_integration_required'):
        store.settle('task', 'alice', expected_version=reserved.version, reservation_id='read',
                     outcome='success', evidence_ref=record.outbox_message_id)
    state = store.load('task', 'alice', expected_version=reserved.version)
    assert state.state.inflight == 'read'
    assert state.state.evidence_sources == []


@pytest.mark.parametrize('run_id,owner,resource,state', [
    ('task', 'alice', 'requested', 'available'),
    ('other-task', 'alice', 'requested', 'available'),
    ('task', 'bob', 'requested', 'available'),
    ('task', 'alice', 'unrelated', 'available'),
    ('task', 'alice', 'requested', 'deleted'),
])
def test_existing_source_acquisition_row_cannot_prove_unbound_action(setup, run_id, owner, resource, state):
    """Acquisition records belong to check runs; a row is not task-action authority."""
    from datetime import datetime, timezone
    from types import SimpleNamespace
    from alphonse.agent_v2.services.source_acquisition import SourceAcquisition

    questions, store, _, _, _ = setup
    first = init(setup)
    reserved = store.reserve('task', 'alice', expected_version=first.version, reservation_id='read',
                             kind='tool', amounts=v(actions=1), now_ms=1100)
    # Invoke the real acquisition schema setup against this disposable task DB;
    # dispatcher, events and provider I/O are intentionally not exercised.
    events = SimpleNamespace(get_event_type=lambda *_: None, get_worker=lambda *_: None,
                             register_event_type=lambda **_: None, register_worker=lambda **_: None,
                             bind_managed_adapter=lambda *_: None)
    SourceAcquisition(SimpleNamespace(store=questions), events,
                      clock=lambda: datetime.now(timezone.utc))
    definition = {'evidence_id': 'source-evidence', 'executing_user_id': owner,
                  'resource': resource, 'state': state}
    with questions._connect() as conn:
        conn.execute('INSERT INTO v2_source_evidence VALUES (?,?,?,?)',
                     ('source-evidence', run_id, 'scope', json.dumps(definition)))
    with pytest.raises(CheckpointConflict, match='verified_evidence_integration_required'):
        store.settle('task', 'alice', expected_version=reserved.version, reservation_id='read',
                     outcome='success', evidence_ref='source-evidence')
    restored = store.load('task', 'alice', expected_version=reserved.version)
    assert restored.state.inflight == 'read'
    assert restored.state.evidence_sources == []


def test_unknown_model_usage_blocks_tool_dispatch(setup):
    _, store, _, _, _ = setup
    first = init(setup)
    reserved = store.reserve('task', 'alice', expected_version=first.version, reservation_id='model',
                             kind='model', amounts=v(model_calls=1, input_tokens=100, output_tokens=30), now_ms=1100)
    settled = store.settle('task', 'alice', expected_version=reserved.version,
                           reservation_id='model', outcome='success', input_tokens=50)
    with pytest.raises(CheckpointConflict, match='model_usage_unresolved'):
        store.reserve('task', 'alice', expected_version=settled.version, reservation_id='action',
                      kind='tool', amounts=v(actions=1), now_ms=1200)


def test_model_reservation_overrun_blocks_further_dispatch(setup):
    _, store, _, _, _ = setup
    first = init(setup)
    reserved = store.reserve('task', 'alice', expected_version=first.version, reservation_id='model',
                             kind='model', amounts=v(model_calls=1, input_tokens=100, output_tokens=30), now_ms=1100)
    settled = store.settle('task', 'alice', expected_version=reserved.version,
                           reservation_id='model', outcome='success', input_tokens=50, output_tokens=31)
    with pytest.raises(CheckpointConflict, match='model_reservation_overrun'):
        store.reserve('task', 'alice', expected_version=settled.version, reservation_id='action',
                      kind='tool', amounts=v(actions=1), now_ms=1200)


@pytest.mark.parametrize('outcome', ['failure', 'timeout', 'cancelled'])
def test_unconfirmed_tool_outcome_keeps_inflight_for_reconciliation(setup, outcome):
    _, store, _, _, _ = setup
    first = init(setup)
    reserved = store.reserve('task', 'alice', expected_version=first.version, reservation_id='action',
                             kind='tool', amounts=v(actions=1), now_ms=1100)
    with pytest.raises(CheckpointConflict, match='reconciliation'):
        store.settle('task', 'alice', expected_version=reserved.version,
                     reservation_id='action', outcome=outcome)
    recovered = store.load('task', 'alice', expected_version=reserved.version)
    assert recovered.state.inflight == 'action'
    assert recovered.budget.usage()['in_flight'] == ['action']


def test_cancelled_checkpoint_blocks_stale_reservation_and_settlement(setup):
    questions, store, _, _, _ = setup
    first = init(setup)
    reserved = store.reserve('task', 'alice', expected_version=first.version, reservation_id='read',
                             kind='tool', amounts=v(actions=1), now_ms=1100)
    with questions._connect() as conn:
        conn.execute("UPDATE v2_task_checkpoints SET status='cancelled' WHERE task_id='task'")
    with pytest.raises(CheckpointConflict):
        store.settle('task', 'alice', expected_version=reserved.version, reservation_id='read',
                     outcome='success', evidence_ref='evidence:1')
    with questions._connect() as conn:
        saved = conn.execute("SELECT status, task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()
    assert saved['status'] == 'cancelled'
    assert json.loads(saved['task_state_json'])['metadata'][STATE_KEY]['state']['inflight'] == 'read'
    inspection = store.inspect_cancelled('task', 'alice', expected_version=reserved.version)
    assert inspection == {'version': reserved.version, 'cancelled': True, 'inflight': 'read', 'kind': 'tool'}
    with pytest.raises(CheckpointConflict):
        store.inspect_cancelled('task', 'mallory', expected_version=reserved.version)
    with pytest.raises(CheckpointConflict):
        store.load('task', 'alice', expected_version=reserved.version)


def test_existing_engine_other_actor_and_stale_version_are_not_reinterpreted(setup):
    questions, store, version, state, budget = setup
    with pytest.raises(CheckpointConflict):
        store.initialize('task', 'mallory', expected_version=version, state=CandidateState('task', 'mallory', 'goal'), budget=budget)
    with questions._connect() as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        raw['intelligence_engine'] = 'hierarchical_v3'
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    with pytest.raises(CheckpointConflict, match='identity_or_engine'):
        store.initialize('task', 'alice', expected_version=version, state=state, budget=budget)
    assert STATE_KEY not in questions.load_task_checkpoint('task').metadata


def test_unsupported_schema_and_tampered_inflight_fail_closed(setup):
    questions, store, _, _, _ = setup
    first = init(setup)
    with questions._connect() as conn:
        row = conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()
        raw = json.loads(row[0]); raw['metadata'][STATE_KEY]['schema_version'] = 2
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    with pytest.raises(ValueError, match='version_unsupported'):
        store.load('task', 'alice', expected_version=first.version)


def test_unknown_task_schema_is_never_reinterpreted(setup):
    questions, store, version, state, budget = setup
    with questions._connect() as conn:
        raw = json.loads(conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0])
        raw['intelligence_schema_version'] = 999
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (json.dumps(raw),))
    with pytest.raises(CheckpointConflict, match='identity_or_engine'):
        store.initialize('task', 'alice', expected_version=version, state=state, budget=budget)


def test_stale_version_cannot_overwrite_a_later_checkpoint(setup):
    questions, store, _, _, _ = setup
    first = init(setup)
    # Another transaction updates the row and advances its version.
    with questions._connect() as conn:
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=json_set(task_state_json, "
                     "'$.metadata.unrelated.preserve', 'updated'), version=version+1 WHERE task_id='task'")
    with pytest.raises(CheckpointConflict):
        store.reserve('task', 'alice', expected_version=first.version, reservation_id='read',
                      kind='tool', amounts=v(actions=1), now_ms=1100)
    assert questions.load_task_checkpoint('task').metadata['unrelated']['preserve'] == 'updated'


def test_legacy_writer_loaded_before_reservation_cannot_erase_it(setup):
    questions, store, _, _, _ = setup
    first = init(setup)
    stale_task = questions.load_task_checkpoint('task')
    reserved = store.reserve('task', 'alice', expected_version=first.version, reservation_id='read',
                             kind='tool', amounts=v(actions=1), now_ms=1100)
    with pytest.raises(ValueError, match='experimental_writer_required'):
        questions.save_task_checkpoint(stale_task, status='running')
    actual = store.load('task', 'alice', expected_version=reserved.version)
    assert actual.state.inflight == 'read'
    assert actual.budget.usage()['in_flight'] == ['read']


def test_simultaneous_legacy_writer_never_replaces_experimental_reservation(setup):
    questions, store, _, _, _ = setup
    first = init(setup)
    stale = questions.load_task_checkpoint('task')
    def legacy():
        try:
            questions.save_task_checkpoint(stale, status='running')
        except ValueError as exc:
            assert str(exc) == 'task_checkpoint_experimental_writer_required'
            return 'blocked'
        return 'overwrote'
    def reserve():
        return store.reserve('task', 'alice', expected_version=first.version, reservation_id='read',
                             kind='tool', amounts=v(actions=1), now_ms=1100)
    with ThreadPoolExecutor(max_workers=2) as executor:
        old = executor.submit(legacy)
        new = executor.submit(reserve)
        assert old.result() == 'blocked'
        result = new.result()
    assert store.load('task', 'alice', expected_version=result.version).state.inflight == 'read'


def test_cancellation_still_works_for_protected_experimental_checkpoint(setup):
    questions, store, _, _, _ = setup
    first = init(setup)
    assert questions.mark_task_checkpoint_terminal('task', status='cancelled')
    with pytest.raises(CheckpointConflict):
        store.reserve('task', 'alice', expected_version=first.version, reservation_id='read',
                      kind='tool', amounts=v(actions=1), now_ms=1100)


@pytest.mark.parametrize('status', ['done', 'failed'])
def test_legacy_terminal_update_cannot_bypass_inflight_or_completion(setup, status):
    questions, store, _, _, _ = setup
    first = init(setup)
    reserved = store.reserve('task', 'alice', expected_version=first.version, reservation_id='read',
                             kind='tool', amounts=v(actions=1), now_ms=1100)
    assert not questions.mark_task_checkpoint_terminal('task', status=status)
    assert store.load('task', 'alice', expected_version=reserved.version).state.inflight == 'read'


def test_legacy_task_can_still_be_marked_done_or_failed(setup):
    questions, _, _, _, _ = setup
    for index, status in enumerate(('done', 'failed')):
        task = TaskState(task_id=f'legacy-{index}', user='alice', goal='legacy')
        questions.save_task_checkpoint(task, status='running')
        assert questions.mark_task_checkpoint_terminal(task.task_id, status=status)


def test_delegated_question_uuid_collision_never_replaces_experimental_task(setup, monkeypatch):
    questions, store, _, _, _ = setup
    first = init(setup)
    parent = TaskState(task_id='parent', user='alice', goal='ask bob')
    # create_question obtains the delegated child ID first, then question ID.
    identifiers = iter(('task', 'question-id'))
    monkeypatch.setattr('alphonse.agent_v2.core.questions.uuid.uuid4', lambda: next(identifiers))
    with pytest.raises(ValueError, match='experimental_writer_required'):
        questions.create_question(task=parent, question='question', respondent_user_id='bob')
    assert store.load('task', 'alice', expected_version=first.version).state.inflight == ''
    assert questions.load_task_checkpoint('parent') is None


def test_delegated_child_collision_retains_existing_legacy_replace_behavior(setup, monkeypatch):
    questions, _, _, _, _ = setup
    legacy = TaskState(task_id='legacy-child', user='bob', goal='earlier task')
    questions.save_task_checkpoint(legacy, status='running')
    parent = TaskState(task_id='parent', user='alice', goal='ask bob')
    identifiers = iter(('legacy-child', 'question-id'))
    monkeypatch.setattr('alphonse.agent_v2.core.questions.uuid.uuid4', lambda: next(identifiers))
    question = questions.create_question(task=parent, question='new question', respondent_user_id='bob')
    assert question.metadata['child_task_id'] == 'legacy-child'
    assert questions.load_task_checkpoint('legacy-child').goal == 'new question'
    with questions._connect() as conn:
        row = conn.execute("SELECT version, status FROM v2_task_checkpoints WHERE task_id='legacy-child'").fetchone()
    assert (row['version'], row['status']) == (1, 'waiting_user')


@pytest.mark.parametrize('invalid', ['not-json', '[]'])
def test_legacy_writer_cannot_replace_malformed_existing_checkpoint(setup, invalid):
    questions, _, _, _, _ = setup
    stale = questions.load_task_checkpoint('task')
    with questions._connect() as conn:
        conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='task'", (invalid,))
    with pytest.raises(ValueError, match='existing_state_invalid'):
        questions.save_task_checkpoint(stale, status='running')
    with questions._connect() as conn:
        assert conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='task'").fetchone()[0] == invalid


def test_budget_rejection_time_is_committed_to_same_row(setup):
    _, store, _, _, _ = setup
    first = init(setup)
    with pytest.raises(BudgetExhausted, match='elapsed'):
        store.reserve('task', 'alice', expected_version=first.version, reservation_id='late', kind='tool',
                      amounts=v(actions=1), now_ms=10000)
    restored = store.load('task', 'alice', expected_version=first.version + 1)
    assert restored.budget.last_observed_at_ms == 10000
    assert restored.state.inflight == ''
    with pytest.raises(ValueError, match='clock'):
        store.reserve('task', 'alice', expected_version=restored.version, reservation_id='rollback',
                      kind='tool', amounts=v(actions=1), now_ms=2000)
