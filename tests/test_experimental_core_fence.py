"""The core must not consume an experimental fenced claim prematurely."""
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import (
    AlphonseCore, CoreMessage, LoopStepStatus, ProcessingResult, StateSnapshot,
)
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.state import reset_state


class SpyMemory:
    def __init__(self):
        self.started = []
        self.finished = []

    def start_task(self, task):
        self.started.append(task.task_id)
        return ''

    def finish_task(self, task):
        self.finished.append(task.task_id)


class SpyState:
    def __init__(self):
        self.updates = []

    def update(self, snapshot):
        self.updates.append(snapshot)


class SpyProcessor:
    def __init__(self):
        self.tasks = []

    def process(self, task, context):
        self.tasks.append(task)
        return ProcessingResult(snapshot=StateSnapshot(current_work=task.goal))


def build_core(queue):
    reset_state()
    processor, state, memory, activity, ui, delivery, callback = (
        SpyProcessor(), SpyState(), SpyMemory(), [], [], [], []
    )
    core = AlphonseCore(
        intelligence=processor, messages=queue, tools=SimpleNamespace(),
        prompts=SimpleNamespace(), state=state, memory=memory,
        activity_sink=activity.append, ui_event_sink=ui.append,
        delivery_sink=lambda value: delivery.append(value),
        active_task_callback=lambda *value: callback.append(value),
    )
    return core, processor, state, memory, activity, ui, delivery, callback


def enqueue(queue):
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='synthetic', user='alice'), message_id='m')


def test_experimental_mode_does_not_claim_or_start_any_work(tmp_path):
    queue = SQLiteMessageQueue(tmp_path / 'q.sqlite', experimental_claim_fencing=True)
    enqueue(queue)
    core, processor, state, memory, activity, ui, delivery, callback = build_core(queue)

    result = core.step()

    assert result.status is LoopStepStatus.BUSY
    assert result.error == 'claim_fencing_runtime_unavailable'
    assert result.state_before == result.state_after
    assert result.queued_message_id is None
    assert not processor.tasks and not state.updates and not memory.started and not memory.finished
    assert not activity and not ui and not delivery and not callback
    with queue._connect() as conn:
        row = conn.execute('SELECT status, claim_generation, attempt_count FROM v2_inbound_messages WHERE message_id=?', ('m',)).fetchone()
        assert tuple(row) == ('pending', '', 0)


def test_default_queue_still_processes_and_finishes(tmp_path):
    queue = SQLiteMessageQueue(tmp_path / 'q.sqlite')
    enqueue(queue)
    core, processor, state, memory, activity, ui, delivery, callback = build_core(queue)

    result = core.step()

    assert result.status is LoopStepStatus.PROCESSED
    assert len(processor.tasks) == len(memory.started) == len(memory.finished) == len(state.updates) == 1
    assert len(callback) == 1


def test_acquired_envelope_is_used_without_a_second_dequeue(tmp_path, monkeypatch):
    queue = SQLiteMessageQueue(tmp_path / 'q.sqlite')
    enqueue(queue)
    original = queue.dequeue
    acquired = []

    def dequeue(selector=None):
        assert not acquired, 'a core step must not claim twice'
        item = original(selector)
        acquired.append(item)
        return item

    monkeypatch.setattr(queue, 'dequeue', dequeue)
    core, processor, state, memory, activity, ui, delivery, callback = build_core(queue)
    result = core.step()
    assert result.status is LoopStepStatus.PROCESSED
    assert len(acquired) == 1
    assert callback[0][0] is acquired[0]
    assert len(processor.tasks) == 1


def test_unadvertised_fenced_receipt_is_not_processed_or_disposed(tmp_path):
    queue = SQLiteMessageQueue(tmp_path / 'q.sqlite', experimental_claim_fencing=True)
    enqueue(queue)

    class Wrapper:
        # A queue wrapper can omit the capability flag while returning the real
        # claimed message. Core must not treat that receipt as a legacy claim.
        def dequeue(self, selector=None):
            return queue.dequeue(selector)

    core, processor, state, memory, activity, ui, delivery, callback = build_core(Wrapper())
    result = core.step()

    assert result.status is LoopStepStatus.BUSY
    assert result.error == 'claim_fencing_runtime_unavailable'
    assert result.state_before == result.state_after
    assert result.queued_message_id is None
    assert not processor.tasks and not state.updates and not memory.started and not memory.finished
    assert not activity and not ui and not delivery and not callback
    with queue._connect() as conn:
        row = conn.execute('SELECT status, claim_generation, attempt_count FROM v2_inbound_messages WHERE message_id=?', ('m',)).fetchone()
        assert row['status'] == 'processing' and row['claim_generation'] and row['attempt_count'] == 1


def test_acquired_boundary_refuses_exact_fenced_envelope_without_dequeue(tmp_path, monkeypatch):
    queue = SQLiteMessageQueue(tmp_path / 'q.sqlite', experimental_claim_fencing=True)
    enqueue(queue)
    claimed = queue.claim_next()
    core, processor, state, memory, activity, ui, delivery, callback = build_core(queue)
    monkeypatch.setattr(queue, 'dequeue', lambda *_: pytest.fail('must not acquire again'))
    from alphonse.agent_v2.core.state.runtime import State

    with core._step_lock:
        result = core._process_acquired(claimed, State.snapshot())

    assert result.status is LoopStepStatus.BUSY
    assert result.error == 'claim_fencing_runtime_unavailable'
    assert not processor.tasks and not state.updates and not memory.started and not memory.finished
    assert not activity and not ui and not delivery and not callback
    with queue._connect() as conn:
        row = conn.execute("SELECT status,claim_generation FROM v2_inbound_messages WHERE message_id='m'").fetchone()
    assert row['status'] == 'processing' and row['claim_generation'] == claimed.claim_receipt.claim_generation
