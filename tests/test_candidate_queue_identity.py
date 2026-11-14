"""Stable identity for explicitly registered candidate queue tasks."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import AlphonseCore, CoreMessage, LoopStepStatus, ProcessingResult, ProcessingStatus, StateSnapshot
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.processor import EngineRoutingProcessor
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.state import reset_state
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry


class Recorder:
    def __init__(self):
        self.seen = []

    def process(self, task, context):
        self.seen.append((task.task_id, task.message_id, task.user, task.intelligence_engine,
                          task.intelligence_schema_version, task.goal))
        return ProcessingResult(snapshot=StateSnapshot(task_owner=task.user), status=ProcessingStatus.WAITING)


def RegisteredRouter(candidate):
    return EngineRoutingProcessor(v3=None, owner_read_candidate=candidate)


def make_core(tmp_path, router):
    queue = InMemoryMessageQueue()
    questions = SQLiteQuestionStore(tmp_path / 'questions.sqlite')
    state = SimpleNamespace(value=None)
    state.update = lambda value: setattr(state, 'value', value)
    state.snapshot = lambda: state.value or StateSnapshot()
    core = AlphonseCore(intelligence=router, messages=queue, tools=InMemoryToolRegistry(), prompts=object(),
                        state=state, memory=None, question_store=questions)
    return core, queue, questions


def enqueue(queue, *, metadata=None, owner='owner'):
    data = {'intelligence_engine': 'single_loop_experimental', 'intelligence_schema_version': 1}
    data.update(metadata or {})
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='original', user=owner, metadata=data),
                  message_id='msg')


def saved(questions, *, owner='owner'):
    task = TaskState(task_id='msg', message_id='msg', user=owner, goal='saved',
                     intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    questions.save_task_checkpoint(task)


@pytest.fixture(autouse=True)
def clear_global_state():
    reset_state()
    yield
    reset_state()


def test_candidate_uses_queue_id_and_restores_matching_checkpoint(tmp_path):
    recorder = Recorder()
    core, queue, questions = make_core(tmp_path, RegisteredRouter(recorder))
    enqueue(queue)
    assert core.step().status is LoopStepStatus.WAITING
    assert recorder.seen == [('msg', 'msg', 'owner', 'single_loop_experimental', 1, 'original')]

    reset_state()
    saved(questions)
    queue = InMemoryMessageQueue()
    core.messages = queue
    enqueue(queue)
    assert core.step().status is LoopStepStatus.WAITING
    assert recorder.seen[-1] == ('msg', 'msg', 'owner', 'single_loop_experimental', 1, 'saved')


@pytest.mark.parametrize('mutation', ['db_owner', 'raw_owner', 'message', 'engine', 'schema', 'status', 'malformed'])
def test_candidate_checkpoint_mismatch_is_never_dispatched(tmp_path, mutation):
    import json
    recorder = Recorder()
    core, queue, questions = make_core(tmp_path, RegisteredRouter(recorder))
    saved(questions)
    with questions._connect() as conn:
        row = conn.execute("SELECT task_state_json FROM v2_task_checkpoints WHERE task_id='msg'").fetchone()
        raw = json.loads(row['task_state_json'])
        if mutation == 'db_owner':
            conn.execute("UPDATE v2_task_checkpoints SET owner_id='other' WHERE task_id='msg'")
        elif mutation == 'malformed':
            conn.execute("UPDATE v2_task_checkpoints SET task_state_json='not-json' WHERE task_id='msg'")
        else:
            key, value = {'raw_owner': ('user', 'other'), 'message': ('message_id', 'other'),
                          'engine': ('intelligence_engine', 'hierarchical_v3'),
                          'schema': ('intelligence_schema_version', 3), 'status': ('status', 'completed')}[mutation]
            raw[key] = value
            conn.execute("UPDATE v2_task_checkpoints SET task_state_json=? WHERE task_id='msg'", (json.dumps(raw),))
    enqueue(queue)
    result = core.step()
    assert (result.status, result.error) == (LoopStepStatus.FAILED, 'candidate_checkpoint_unavailable')
    assert recorder.seen == []


def test_candidate_rejects_embedded_resume_and_default_router_stays_disabled(tmp_path):
    recorder = Recorder()
    core, queue, _ = make_core(tmp_path, RegisteredRouter(recorder))
    enqueue(queue, metadata={'task_state': {'user': 'owner'}})
    result = core.step()
    assert (result.status, result.error) == (LoopStepStatus.FAILED, 'candidate_checkpoint_unavailable')
    assert not recorder.seen
    reset_state()
    core, queue, _ = make_core(tmp_path, EngineRoutingProcessor(v3=None))
    enqueue(queue)
    result = core.step()
    assert (result.status, result.error) == (LoopStepStatus.FAILED, 'intelligence_engine_unregistered')


@pytest.mark.parametrize('schema', [True, '1', 0, 2, None])
def test_candidate_rejects_invalid_queue_schema_before_dispatch(tmp_path, schema):
    recorder = Recorder()
    core, queue, _ = make_core(tmp_path, RegisteredRouter(recorder))
    enqueue(queue, metadata={'intelligence_schema_version': schema})
    result = core.step()
    assert (result.status, result.error) == (LoopStepStatus.FAILED, 'candidate_checkpoint_unavailable')
    assert not recorder.seen
