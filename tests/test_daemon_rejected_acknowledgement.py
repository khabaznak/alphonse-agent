"""A rejected legacy disposition cannot be reported as terminal success."""
from datetime import datetime, timezone

import pytest

from alphonse.agent_v2.core.core import CoreMessage, LoopStepStatus, ProcessingResult, ProcessingStatus, StateSnapshot
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.core.state import reset_state
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host


@pytest.mark.parametrize(('processor_status', 'queue_result'), [
    (ProcessingStatus.COMPLETED, False), (ProcessingStatus.CANCELLED, False),
    (ProcessingStatus.FAILED, False), (ProcessingStatus.COMPLETED, None), (ProcessingStatus.FAILED, 0),
])
def test_rejected_queue_ack_skips_terminal_checkpoint_and_progress(tmp_path, monkeypatch, processor_status, queue_result):
    reset_state()
    class Processor:
        def process(self, task, context):
            return ProcessingResult(snapshot=StateSnapshot(current_work='synthetic'), status=processor_status)

    queue = SQLiteMessageQueue(tmp_path / 'runtime.sqlite', lease_owner='worker')
    queue.enqueue(CoreMessage(datetime.now(timezone.utc), 'synthetic', 'alice',
                              metadata={'automation_execution_id': 'fixture-event'}), message_id='message')
    runtime = build_runtime_host(messages=queue, processor=Processor())
    runtime.core.conversation_store = None
    daemon = V2Daemon(runtime)
    monkeypatch.setattr(daemon, '_run_retention_if_due', lambda: None)
    monkeypatch.setattr(daemon, '_reconcile_unconfirmed_scheduled_deliveries', lambda: None)
    monkeypatch.setattr(runtime.core.memory, 'start_task', lambda *_args, **_kwargs: '')
    monkeypatch.setattr(runtime.core.memory, 'finish_task', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(queue, 'ack', lambda *_args, **_kwargs: queue_result)
    monkeypatch.setattr(queue, 'retry', lambda *_args, **_kwargs: queue_result)
    monkeypatch.setattr(daemon, '_project_snapshot_delivery', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(daemon, '_mark_projected_occurrence', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(daemon, '_emit_scheduled_task_card', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(daemon, '_mark_snapshot_checkpoint_terminal', lambda *_args, **_kwargs: pytest.fail('terminal checkpoint'))
    monkeypatch.setattr(daemon, '_publish_task_progress', lambda *_args, **_kwargs: pytest.fail('terminal progress'))
    monkeypatch.setattr(daemon.kill_switch, 'clear', lambda *_args, **_kwargs: pytest.fail('kill-switch clearance'))
    monkeypatch.setattr(runtime.core, 'clear_failure', lambda *_args, **_kwargs: pytest.fail('FSM reset'))
    monkeypatch.setattr(daemon.event_store, 'execution_lifecycle', lambda *_args, **_kwargs: None)
    monkeypatch.setattr(daemon.event_dispatcher, 'reconcile', lambda *_args, **_kwargs: pytest.fail('event reconciliation'))

    result = daemon.run_once()

    assert result.status is LoopStepStatus.SUPERSEDED
    assert result.queued_message_id is None and result.error is None
    assert queue.status_for('message') == 'processing'
    assert daemon.active_work() == {}
    assert daemon.activity_status()['state'] == 'error'
