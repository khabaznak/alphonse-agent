"""A daemon must not partially run against its unsupported fenced queue."""

from datetime import datetime, timezone

import pytest

from alphonse.agent_v2.core.core import CoreMessage, LoopStepStatus
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host


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
