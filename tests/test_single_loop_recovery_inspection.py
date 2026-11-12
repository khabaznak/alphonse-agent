"""Read-only experimental recovery inspection; no effect dispatch or release."""
import json

import pytest

from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ENGINE, ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.single_loop_recovery_inspection import inspect_recovery
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits, TaskBudget
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def fixture(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    users = V2UserStore(path)
    users.set_users_root(tmp_path / 'profiles')
    users.create_user(user_id='alice', display_name='Alice')
    users.create_user(user_id='bob', display_name='Bob')
    questions = SQLiteQuestionStore(path)
    task = TaskState(task_id='task', user='alice', goal='private request', intelligence_engine=ENGINE,
                     intelligence_schema_version=1, metadata={'private': 'secret'})
    version = questions.save_task_checkpoint(task, status='running')
    store = ExperimentalCheckpointStore(questions)
    total = {'actions': 2, 'model_calls': 2, 'input_tokens': 500, 'output_tokens': 200}
    final = {'actions': 0, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}
    checkpoint = store.initialize('task', 'alice', expected_version=version,
        state=CandidateState('task', 'alice', 'private-goal'),
        budget=TaskBudget(BudgetLimits(total, final, 10000, 0), started_at_ms=1000))
    return users, questions, store, checkpoint.version


def read(store, actor='alice', task='task', version=2):
    return inspect_recovery(store, AccountSession(actor), task_id=task, expected_version=version)


@pytest.mark.parametrize('kind,amounts', [
    ('tool', {'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}),
    ('model', {'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 30}),
])
def test_owner_sees_pending_kind_but_inspection_does_not_change_or_release(fixture, kind, amounts):
    _, questions, store, version = fixture
    assert read(store, version=version).status == 'no_pending'
    checkpoint = store.reserve('task', 'alice', expected_version=version, reservation_id='private-action',
                               kind=kind, amounts=amounts, now_ms=1100)
    with questions._connect() as conn:
        before = tuple(conn.execute('SELECT status,version,task_state_json FROM v2_task_checkpoints').fetchone())
    result = read(store, version=checkpoint.version)
    assert (result.status, result.checkpoint_status, result.pending_kind) == ('requires_reconciliation', 'running', kind)
    assert 'private' not in repr(result) and 'secret' not in repr(result)
    with questions._connect() as conn:
        after = tuple(conn.execute('SELECT status,version,task_state_json FROM v2_task_checkpoints').fetchone())
    assert before == after
    assert store.load('task', 'alice', expected_version=checkpoint.version).state.inflight == 'private-action'


def test_other_inactive_missing_and_stale_are_indistinguishable(fixture):
    users, _, store, version = fixture
    unavailable = read(store, actor='bob', task='missing', version=version)
    assert unavailable.status == 'unavailable'
    assert read(store, actor='bob', version=version) == unavailable
    assert read(store, version=version + 1) == unavailable
    users.update_user('alice', is_active=False)
    assert read(store, version=version) == unavailable


def test_cancelled_pending_remains_uncertain(fixture):
    _, questions, store, version = fixture
    checkpoint = store.reserve('task', 'alice', expected_version=version, reservation_id='action', kind='tool',
                               amounts={'actions': 1, 'model_calls': 0, 'input_tokens': 0, 'output_tokens': 0}, now_ms=1100)
    assert questions.mark_task_checkpoint_terminal('task', status='cancelled')
    # Terminal cancellation can preserve the checkpoint version.
    result = read(store, version=checkpoint.version)
    assert (result.status, result.checkpoint_status, result.pending_kind) == ('requires_reconciliation', 'cancelled', 'tool')
    assert store.inspect_cancelled('task', 'alice', expected_version=checkpoint.version)['inflight'] == 'action'


def test_settled_unknown_usage_is_not_reported_as_clear(fixture):
    _, _, store, version = fixture
    checkpoint = store.reserve('task', 'alice', expected_version=version, reservation_id='model', kind='model',
        amounts={'actions': 0, 'model_calls': 1, 'input_tokens': 100, 'output_tokens': 30}, now_ms=1100)
    settled = store.settle('task', 'alice', expected_version=checkpoint.version, reservation_id='model',
                           outcome='success', input_tokens=50)
    result = read(store, version=settled.version)
    assert (result.status, result.pending_kind) == ('usage_unresolved', 'model')


def test_corrupt_envelope_does_not_project_status(fixture):
    _, questions, store, version = fixture
    with questions._connect() as conn:
        row = conn.execute('SELECT task_state_json FROM v2_task_checkpoints').fetchone()
        task = json.loads(row[0]); task['intelligence_schema_version'] = 2
        conn.execute('UPDATE v2_task_checkpoints SET task_state_json=?', (json.dumps(task),))
    assert read(store, version=version).status == 'unavailable'
