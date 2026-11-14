"""Disposable acquisition → bootstrap → claim-bound candidate decision."""
from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, CoreMessage
from alphonse.agent_v2.core.intelligence.v3.single_loop_atomic_step import run_atomic_step
from alphonse.agent_v2.core.intelligence.v3.single_loop_candidate import CandidateHost
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import CheckpointConflict, ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.task_budget import BudgetLimits
from alphonse.agent_v2.core.messages import InMemoryMessageQueue, SQLiteMessageQueue
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimSuperseded
from alphonse.agent_v2.core.messages.experimental_claim_bootstrap import bootstrap_fixture_task
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.users import V2UserStore
from unsafe_reclaim_fixture import simulate_old_writer_requeue


@pytest.fixture
def fixture(tmp_path):
    path = tmp_path / 'runtime.sqlite'
    users = V2UserStore(path)
    users.set_users_root(tmp_path / 'users')
    users.create_user(user_id='alice', display_name='Alice')
    queue = SQLiteMessageQueue(path, lease_owner='worker', experimental_claim_fencing=True)
    questions = SQLiteQuestionStore(path)
    now = datetime.now(timezone.utc)
    queue.enqueue(CoreMessage(now, 'Synthetic request', 'alice'), message_id='message')
    claim = queue.claim_next(lease_seconds=60).claim_receipt
    limits = BudgetLimits(dict(actions=2, model_calls=2, input_tokens=4000, output_tokens=200),
                          dict(actions=0, model_calls=0, input_tokens=0, output_tokens=0), 100000, 0)
    return queue, users, questions, ExperimentalCheckpointStore(questions), claim, limits, now


def bootstrap(f, **kwargs):
    _, _, _, store, claim, limits, now = f
    return bootstrap_fixture_task(store, kwargs.pop('claim', claim), actor=kwargs.pop('actor', 'alice'),
                                  limits=limits, clock=kwargs.pop('clock', lambda: now), **kwargs)


def test_acquired_message_bootstraps_candidate_and_runs_claim_bound_model(fixture):
    _, _, questions, store, claim, _, now = fixture
    created = bootstrap(fixture)
    task = questions.load_task_checkpoint('message')
    assert task.goal == 'Synthetic request' and task.user == 'alice' and task.message_id == 'message'
    calls = []
    def choose(prompt, cap):
        calls.append(prompt)
        return {'kind': 'tool', 'usage': {'input_tokens': 10, 'output_tokens': 5}}
    def forbidden(*_):
        pytest.fail('unexpected fixture side effect')
    host = CandidateHost(task=task, context=CoreLoopContext(messages=InMemoryMessageQueue()), budgets=None,
        checkpoint=forbidden, choose=choose, authorize_source=lambda actor, ref: actor == 'alice' and ref == 'queued_goal',
        load_source=lambda actor, ref: task.goal, authorize_read=forbidden, record_evidence=forbidden,
        verify_response=forbidden, now_ms=lambda: int(now.timestamp() * 1000), input_reservation=2000,
        output_reservation=50)
    result = run_atomic_step(created.state, host, store, expected_version=created.version,
                             claim=claim, claim_clock=lambda: now)
    assert result.reason == 'tool_dispatch_and_evidence_not_integrated'
    assert len(calls) == 1 and 'Synthetic request' in calls[0]
    assert not store.load('message', 'alice', expected_version=result.checkpoint_version).state.inflight
    assert claim.claim_generation not in str(questions.load_task_checkpoint('message').metadata)


def test_stale_or_expired_claim_never_creates_checkpoint(fixture):
    queue, _, questions, _, claim, _, now = fixture
    times = iter((now, now, now + timedelta(days=1)))
    with pytest.raises(ClaimSuperseded):
        bootstrap(fixture, clock=lambda: next(times))
    assert questions.load_task_checkpoint('message') is None
    simulate_old_writer_requeue(queue, 'message')
    queue.claim_next()
    with pytest.raises(ClaimSuperseded):
        bootstrap(fixture, claim=claim)
    assert questions.load_task_checkpoint('message') is None


def test_existing_checkpoint_and_inactive_actor_are_not_replaced(fixture):
    _, users, questions, _, _, _, _ = fixture
    users.update_user('alice', is_active=False)
    with pytest.raises(ClaimFenceUnavailable):
        bootstrap(fixture)
    assert questions.load_task_checkpoint('message') is None
    users.update_user('alice', is_active=True)
    bootstrap(fixture)
    before = questions.load_task_checkpoint('message').to_checkpoint_dict()
    with pytest.raises(CheckpointConflict, match='checkpoint_exists'):
        bootstrap(fixture)
    assert questions.load_task_checkpoint('message').to_checkpoint_dict() == before


def test_unrecognized_metadata_and_trigger_tampering_roll_back(fixture):
    _, _, questions, _, _, _, _ = fixture
    with sqlite3.connect(questions.db_path) as conn:
        conn.execute("UPDATE v2_inbound_messages SET metadata_json=?", ('{"task_state":{}}',))
    with pytest.raises(ClaimFenceUnavailable, match='unsupported_message'):
        bootstrap(fixture)
    with sqlite3.connect(questions.db_path) as conn:
        conn.execute("UPDATE v2_inbound_messages SET metadata_json='{}'")
        conn.execute('''CREATE TRIGGER corrupt_new AFTER INSERT ON v2_task_checkpoints
            BEGIN UPDATE v2_task_checkpoints SET task_state_json=json_set(NEW.task_state_json, '$.goal', 'wrong')
            WHERE task_id=NEW.task_id; END''')
    with pytest.raises(ClaimFenceUnavailable, match='changed'):
        bootstrap(fixture)
    assert questions.load_task_checkpoint('message') is None


def test_scheduled_handoff_never_becomes_a_new_task(fixture):
    _, _, questions, _, _, _, _ = fixture
    with sqlite3.connect(questions.db_path) as conn:
        conn.execute("INSERT INTO v2_scheduled_handoffs(message_id,task_id,owner_id) VALUES ('message','old','alice')")
    with pytest.raises(ClaimFenceUnavailable, match='unsupported_message'):
        bootstrap(fixture)
    assert questions.load_task_checkpoint('message') is None


@pytest.mark.parametrize('field', ['project_id', 'correlation_id', 'created_at', 'updated_at'])
def test_checkpoint_trigger_cannot_change_auxiliary_columns(fixture, field):
    _, _, questions, _, _, _, _ = fixture
    with sqlite3.connect(questions.db_path) as conn:
        conn.execute(f'''CREATE TRIGGER corrupt_new AFTER INSERT ON v2_task_checkpoints
            BEGIN UPDATE v2_task_checkpoints SET {field}='changed' WHERE task_id=NEW.task_id; END''')
    with pytest.raises(ClaimFenceUnavailable, match='changed'):
        bootstrap(fixture)
    assert questions.load_task_checkpoint('message') is None


def test_checkpoint_trigger_cannot_create_duplicate_for_same_message(fixture):
    _, _, questions, _, _, _, _ = fixture
    with sqlite3.connect(questions.db_path) as conn:
        conn.execute('''CREATE TRIGGER duplicate_new AFTER INSERT ON v2_task_checkpoints
            WHEN NEW.task_id='message' BEGIN
            INSERT INTO v2_task_checkpoints
            (task_id,task_state_json,status,owner_id,project_id,correlation_id,version,created_at,updated_at)
            VALUES ('other',NEW.task_state_json,NEW.status,NEW.owner_id,NEW.project_id,NEW.correlation_id,
                    NEW.version,NEW.created_at,NEW.updated_at); END''')
    with pytest.raises(ClaimFenceUnavailable, match='changed'):
        bootstrap(fixture)
    with sqlite3.connect(questions.db_path) as conn:
        assert conn.execute('SELECT count(*) FROM v2_task_checkpoints').fetchone()[0] == 0
