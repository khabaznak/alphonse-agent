"""A daemon must not partially run against its unsupported fenced queue."""

from datetime import datetime, timezone
from contextlib import contextmanager
from threading import Event, Thread

import pytest

from alphonse.agent_v2.core.core import CoreMessage, LoopStepStatus
from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host


def test_concurrent_daemon_turn_does_not_enter_prework(tmp_path, monkeypatch):
    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite3')
    daemon = V2Daemon(build_runtime_host(messages=queue))
    entered, release = Event(), Event()
    calls = []

    def retention():
        calls.append('retention')
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(daemon, '_run_retention_if_due', retention)
    monkeypatch.setattr(daemon, '_reconcile_unconfirmed_scheduled_deliveries', lambda: calls.append('reconcile'))
    results = []
    worker = Thread(target=lambda: results.append(daemon.run_once()))
    worker.start()
    try:
        assert entered.wait(5)
        result = daemon.run_once()
        assert result.status is LoopStepStatus.BUSY
        assert result.queued_message_id is None
        assert calls == ['retention']
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    assert results[0].status is LoopStepStatus.EMPTY
    assert calls == ['retention', 'reconcile']


def test_daemon_turn_lock_released_after_exception(tmp_path, monkeypatch):
    daemon = V2Daemon(build_runtime_host(messages=SQLiteMessageQueue(tmp_path / 'queue.sqlite3')))
    monkeypatch.setattr(daemon, '_run_retention_if_due', lambda: (_ for _ in ()).throw(RuntimeError('unavailable')))
    with pytest.raises(RuntimeError, match='unavailable'):
        daemon.run_once()
    monkeypatch.setattr(daemon, '_run_retention_if_due', lambda: None)
    monkeypatch.setattr(daemon, '_reconcile_unconfirmed_scheduled_deliveries', lambda: None)
    assert daemon.run_once().status is LoopStepStatus.EMPTY


def test_recursive_daemon_turn_is_busy_before_prework(tmp_path, monkeypatch):
    daemon = V2Daemon(build_runtime_host(messages=SQLiteMessageQueue(tmp_path / 'queue.sqlite3')))
    nested = []

    def retention():
        nested.append(daemon.run_once())

    monkeypatch.setattr(daemon, '_run_retention_if_due', retention)
    monkeypatch.setattr(daemon, '_reconcile_unconfirmed_scheduled_deliveries', lambda: None)
    assert daemon.run_once().status is LoopStepStatus.EMPTY
    assert len(nested) == 1
    assert nested[0].status is LoopStepStatus.BUSY
    assert nested[0].queued_message_id is None


def test_fenced_queue_daemon_exits_before_work_or_queue_claim(tmp_path, monkeypatch):
    queue = SQLiteMessageQueue(tmp_path / "queue.sqlite3", experimental_claim_fencing=True)
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt="synthetic", user="alice"), message_id="msg")
    runtime = build_runtime_host(messages=queue)
    daemon = V2Daemon(runtime)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported fenced daemon must not begin work")

    monkeypatch.setattr(daemon, "_run_retention_if_due", forbidden)
    monkeypatch.setattr(daemon, "_reconcile_unconfirmed_scheduled_deliveries", forbidden)
    monkeypatch.setattr(daemon, "_set_active_work", forbidden)
    monkeypatch.setattr(queue, "peek", forbidden)
    monkeypatch.setattr(runtime.core, "step", forbidden)
    monkeypatch.setattr(runtime.presence_projector, "processing", forbidden)

    result = daemon.run_once()

    assert result.status is LoopStepStatus.BUSY
    assert result.error == "claim_fencing_runtime_unavailable"
    assert result.queued_message_id is None
    assert result.state_before == result.state_after
    with queue._connect() as conn:
        row = conn.execute("SELECT status,claim_generation FROM v2_inbound_messages WHERE message_id='msg'").fetchone()
    assert tuple(row) == ("pending", "")


def test_fenced_queue_start_fails_before_lock_or_startup_effects(tmp_path, monkeypatch):
    queue = SQLiteMessageQueue(tmp_path / "queue.sqlite3", experimental_claim_fencing=True)
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt="synthetic", user="alice"), message_id="msg")
    daemon = V2Daemon(build_runtime_host(messages=queue))

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported fenced daemon must not start")

    monkeypatch.setattr(daemon, "_acquire_single_instance_lock", forbidden)
    monkeypatch.setattr(daemon, "_ensure_home_projects", forbidden)
    monkeypatch.setattr(daemon, "_run_retention_if_due", forbidden)
    monkeypatch.setattr(queue, "reclaim_expired", forbidden)
    monkeypatch.setattr(daemon.ipc, "start", forbidden)

    with pytest.raises(RuntimeError, match="claim_fencing_runtime_unavailable"):
        daemon.start()

    assert daemon._processor_thread is None
    with queue._connect() as conn:
        row = conn.execute("SELECT status,claim_generation FROM v2_inbound_messages WHERE message_id='msg'").fetchone()
    assert tuple(row) == ("pending", "")


def test_superseded_daemon_skips_snapshot_and_queue_disposition(tmp_path, monkeypatch):
    from alphonse.agent_v2.core.state import reset_state
    reset_state()
    class Processor:
        def process(self, task, context):
            return ProcessingResult(snapshot=StateSnapshot(current_work='stale private work'),
                                    status=ProcessingStatus.SUPERSEDED, error='private error')

    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite3', lease_owner='worker')
    queue.enqueue(CoreMessage(datetime.now(timezone.utc), 'synthetic', 'alice',
                              metadata={'automation_execution_id': 'synthetic-event'}), message_id='msg')
    runtime = build_runtime_host(messages=queue, processor=Processor())
    runtime.core.conversation_store = None
    daemon = V2Daemon(runtime)
    monkeypatch.setattr(daemon, '_run_retention_if_due', lambda: None)
    monkeypatch.setattr(daemon, '_reconcile_unconfirmed_scheduled_deliveries', lambda: None)
    monkeypatch.setattr(daemon.event_store, 'execution_lifecycle', lambda *args: None)
    monkeypatch.setattr(daemon.event_dispatcher, 'reconcile', lambda *args: pytest.fail('stale event reconciliation'))
    monkeypatch.setattr(runtime.visible_state, 'snapshot', lambda: pytest.fail('stale snapshot read'))
    monkeypatch.setattr(queue, 'ack', lambda *args, **kwargs: pytest.fail('stale acknowledgement'))
    monkeypatch.setattr(queue, 'retry', lambda *args, **kwargs: pytest.fail('stale retry'))
    # Do not consult the default household memory path for a disposable actor.
    monkeypatch.setattr(runtime.core.memory, 'start_task', lambda *args, **kwargs: '')
    monkeypatch.setattr(runtime.core.memory, 'finish_task', lambda *args, **kwargs: pytest.fail('stale memory finish'))

    result = daemon.run_once()
    assert result.status is LoopStepStatus.SUPERSEDED
    assert result.queued_message_id is None and result.error is None
    assert queue.status_for('msg') == 'processing'


def test_superseded_tui_does_not_read_snapshot_or_drain_outbox(monkeypatch):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.core import LoopStepResult
    from alphonse.agent_v2.core.state import State, reset_state
    from alphonse.agent_v2.integrations.presence import PresenceProjector
    from alphonse.agent_v2.interfaces import tui

    reset_state()
    state = State.snapshot()
    step = LoopStepResult(status=LoopStepStatus.SUPERSEDED, state_before=state, state_after=state)
    runtime = SimpleNamespace(
        queue=SimpleNamespace(peek=lambda: None),
        presence_projector=PresenceProjector(),
        core=SimpleNamespace(step=lambda: step),
        visible_state=SimpleNamespace(snapshot=lambda: pytest.fail('stale snapshot read')),
    )
    monkeypatch.setattr(tui, '_drain_tui_outbox', lambda *_: pytest.fail('stale outbox drain'))
    result = tui.process_tui_queue_once(runtime)
    assert result.step_status is LoopStepStatus.SUPERSEDED
    assert result.response == '' and result.queued_message_id is None


def test_unadvertised_receipt_diagnostic_daemon_prework_precedes_core_refusal(tmp_path, monkeypatch):
    """Documents a gap: core refusal occurs after daemon event/presence prework."""
    queue = SQLiteMessageQueue(tmp_path / 'queue.sqlite3', experimental_claim_fencing=True)
    queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt='synthetic', user='alice',
                              metadata={'automation_execution_id': 'synthetic-event'}), message_id='msg')

    class Wrapper:
        experimental_claim_fencing = False

        def __getattr__(self, name):
            return getattr(queue, name)

    runtime = build_runtime_host(messages=Wrapper())
    daemon = V2Daemon(runtime)
    seen = []
    monkeypatch.setattr(daemon, '_run_retention_if_due', lambda: seen.append('retention'))
    monkeypatch.setattr(daemon, '_reconcile_unconfirmed_scheduled_deliveries', lambda: seen.append('reconcile'))
    monkeypatch.setattr(daemon.event_store, 'execution_lifecycle', lambda *args: seen.append(('event', args)))
    monkeypatch.setattr(daemon.event_dispatcher, 'reconcile', lambda *args: seen.append(('event_reconcile', args)))
    monkeypatch.setattr(queue, 'ack', lambda *args, **kwargs: pytest.fail('unexpected legacy ack'))
    monkeypatch.setattr(queue, 'retry', lambda *args, **kwargs: pytest.fail('unexpected legacy retry'))
    monkeypatch.setattr(runtime.visible_state, 'snapshot', lambda: pytest.fail('stale snapshot must not be inspected'))

    @contextmanager
    def processing(queued):
        seen.append('presence_enter')
        yield
        seen.append('presence_exit')

    monkeypatch.setattr(runtime.presence_projector, 'processing', processing)
    result = daemon.run_once()

    assert result.status is LoopStepStatus.BUSY
    assert result.error == 'claim_fencing_runtime_unavailable'
    assert 'presence_enter' in seen and 'presence_exit' in seen
    assert ('event', ('synthetic-event', 'processing')) in seen
    assert not any(isinstance(item, tuple) and item[0] == 'event_reconcile' for item in seen)
    with queue._connect() as conn:
        row = conn.execute("SELECT status,claim_generation FROM v2_inbound_messages WHERE message_id='msg'").fetchone()
    assert row['status'] == 'processing' and row['claim_generation']
