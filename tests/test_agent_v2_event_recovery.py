"""Synthetic file-backed crash/restart coverage; no providers or account data."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import time

import pytest

from alphonse.agent_v2.automations import EventAutomationStore
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.database import connect_database
from alphonse.agent_v2.retention import prune_operational_data
from alphonse.agent_v2.services.event_worker import EventDispatchWorker


def setup_store(path):
    store = EventAutomationStore(path)
    store.register_event_type(event_type="sensor", version="1", schema={"type": "object"}, max_history=1)
    store.register_worker(worker_id="sensor", display_name="Sensor", allowed_event_types=["sensor"])
    store.create_event_automation(owner_user_id="local", project_id="home", name="Check", prompt="Check sensor",
                                  event_type="sensor", event_version="1")
    return store


def publish(store, identity="one"):
    return store.publish(worker_id="sensor", event_id=identity, event_type="sensor", event_version="1",
                         occurred_at="2000-01-01T00:00:00Z", payload={"reading": 1})


def dispatcher(store, queue, **kwargs):
    def dispatch(execution):
        return queue.enqueue(CoreMessage(timestamp=datetime.now(timezone.utc), prompt=execution["prompt"],
                                         user=execution["owner_user_id"]),
                             message_id=f"event:{execution['execution_id']}").message_id
    return EventDispatchWorker(store=store, messages=queue, dispatch=dispatch, worker_id="test", **kwargs)


def test_concurrent_publish_and_claims_are_atomic(tmp_path):
    path = tmp_path / "events.db"
    setup_store(path)
    stores = [EventAutomationStore(path) for _ in range(4)]
    with ThreadPoolExecutor(4) as pool:
        results = list(pool.map(publish, stores))
    assert sum(not result["duplicate"] for result in results) == 1
    with ThreadPoolExecutor(4) as pool:
        claims = list(pool.map(lambda store: store.claim_event_executions(worker_id="worker"), stores))
    assert sum(map(len, claims)) == 1


@pytest.mark.parametrize("boundary", ["accepted", "claimed", "inserted", "inserted_processing", "inserted_acked", "marked", "processing", "acked"])
def test_restart_at_each_handoff_boundary(tmp_path, boundary):
    path = tmp_path / "events.db"
    store = setup_store(path)
    queue = SQLiteMessageQueue(path)
    publish(store)
    worker = dispatcher(store, queue)
    start = datetime.now(timezone.utc)
    if boundary != "accepted":
        execution = store.claim_event_executions(now=start)[0]
        identity = execution["execution_id"]
        if boundary not in {"claimed"}:
            message_id = worker.dispatch(execution)
            if boundary not in {"inserted", "inserted_processing", "inserted_acked"}:
                store.mark_execution_enqueued(identity, message_id, lease_token=execution["lease_token"])
            if boundary in {"processing", "acked", "inserted_processing", "inserted_acked"}:
                queue.dequeue()
            if boundary in {"acked", "inserted_acked"}:
                assert queue.ack(message_id)
    # A new store/queue/worker, without another publication.
    recovered = EventAutomationStore(path)
    queue = SQLiteMessageQueue(path)
    dispatcher(recovered, queue).run_once(now=start + timedelta(seconds=31))
    with connect_database(path) as conn:
        execution = dict(conn.execute("SELECT * FROM v2_automation_executions").fetchone())
    assert execution["status"] == {"processing": "processing", "acked": "completed", "inserted_processing": "processing", "inserted_acked": "completed"}.get(boundary, "queued")
    assert queue.status_counts()["pending"] == (0 if boundary in {"processing", "acked", "inserted_processing", "inserted_acked"} else 1)
    if boundary == "processing":
        assert queue.reclaim_expired(now=start + timedelta(seconds=120)) == 1
        assert queue.dequeue().message_id == message_id
    # Receipt blocks replay even after ack deleted queue bytes.
    if boundary in {"acked", "inserted_acked"}:
        worker = dispatcher(recovered, queue)
        worker.dispatch({**execution, "prompt": "Check", "owner_user_id": "local"})
        assert queue.size() == 0


def test_expired_claim_fences_stale_owner_and_retries(tmp_path):
    store = setup_store(tmp_path / "events.db")
    publish(store)
    start = datetime.now(timezone.utc)
    first = store.claim_event_executions(now=start)[0]
    assert store.claim_event_executions(now=start + timedelta(seconds=1)) == []
    second = store.claim_event_executions(now=start + timedelta(seconds=31), worker_id="next")[0]
    assert second["attempt_count"] == 2
    assert not store.mark_execution_enqueued(first["execution_id"], "wrong", lease_token=first["lease_token"])
    store.retry_execution(first["execution_id"], lease_token=first["lease_token"], now=start)
    assert store.get_execution(second["execution_id"])["status"] == "claimed"
    store.retry_execution(second["execution_id"], lease_token=second["lease_token"], now=start + timedelta(seconds=31))
    assert store.claim_event_executions(now=start + timedelta(seconds=40)) == []
    assert len(store.claim_event_executions(now=start + timedelta(seconds=41))) == 1


def test_repeated_dispatch_failure_backoff_and_attempt_limit(tmp_path):
    store = setup_store(tmp_path / "events.db")
    publish(store)
    queue = SQLiteMessageQueue(tmp_path / "events.db")
    worker = dispatcher(store, queue, max_attempts=2)
    worker.dispatch = lambda _: (_ for _ in ()).throw(RuntimeError("private provider content"))
    start = datetime.now(timezone.utc)
    first = worker.run_once(now=start)[0]["execution_id"]
    assert store.get_execution(first)["status"] == "retry_wait"
    assert worker.run_once(now=start + timedelta(seconds=4)) == []
    worker.run_once(now=start + timedelta(seconds=5))
    assert store.get_execution(first)["status"] == "failed"


def test_background_recovery_without_new_events_and_bounded_batch(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    for n in range(7):
        publish(store, str(n))
    queue = SQLiteMessageQueue(path)
    worker = dispatcher(EventAutomationStore(path), queue, batch_size=2, poll_interval_sec=.1)
    assert len(worker.run_once()) == 2
    assert queue.size() == 2
    worker.start()
    try:
        deadline = time.monotonic() + 3
        while queue.size() != 7 and time.monotonic() < deadline:
            time.sleep(.01)
        assert queue.size() == 7
    finally:
        worker.stop()


@pytest.mark.parametrize("when", ["pending", "claimed", "queued"])
def test_cancellation_blocks_late_insertion(tmp_path, when):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path)
    worker = dispatcher(store, queue)
    execution = store.claim_event_executions()[0]
    if when == "pending":
        with connect_database(path) as conn:
            conn.execute("UPDATE v2_automation_executions SET status='pending'")
    if when == "queued":
        message_id = worker.dispatch(execution)
        store.mark_execution_enqueued(execution["execution_id"], message_id, lease_token=execution["lease_token"])
    assert worker.cancel(execution["execution_id"])
    worker.dispatch(execution)  # A stale dispatch can't resurrect cancelled work.
    assert queue.size() == 0
    assert store.get_execution(execution["execution_id"])["status"] == "cancelled"
    assert worker.run_once() == []


def test_active_processing_cancellation_requires_kill_switch(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path)
    worker = dispatcher(store, queue)
    identity = worker.run_once()[0]["execution_id"]
    queued = queue.dequeue()
    assert not worker.cancel(identity)
    assert queue.ack(queued.message_id, outcome="cancelled")
    worker.run_once()
    assert store.get_execution(identity)["status"] == "cancelled"


def test_history_and_retention_protect_recovery_but_not_terminal_payloads(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    first = publish(store)
    publish(store, "two")
    assert len(store.list_events()) == 2  # pinned pending evidence exceeds history count
    queue = SQLiteMessageQueue(path)
    worker = dispatcher(store, queue)
    worker.run_once()
    while queued := queue.dequeue():
        queue.ack(queued.message_id)
    worker.run_once()
    assert len(store.list_events()) == 1
    assert publish(store)["duplicate"]
    future = datetime.now(timezone.utc) + timedelta(days=31)
    prune_operational_data(path, now=future)
    assert store.active_executions() == []
    assert publish(store)["event_key"] == first["event_key"]
    assert queue.size() == 0
    # Caller occurrence times never extend replay validity.
    with connect_database(path) as conn:
        conn.execute("UPDATE v2_event_receipts SET accepted_at=?", ((future - timedelta(days=122)).isoformat(),))
    assert publish(store)["reason"] == "event_replay_expired"


def test_transaction_rolls_back_fanout_and_receipt_together(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    with connect_database(path) as conn:
        conn.execute("""CREATE TRIGGER crash_fanout BEFORE INSERT ON v2_automation_executions
                        BEGIN SELECT RAISE(ABORT,'simulated interruption'); END""")
    with pytest.raises(Exception, match="simulated interruption"):
        publish(store)
    assert store.list_events() == []
    with connect_database(path) as conn:
        assert conn.execute("SELECT count(*) FROM v2_event_receipts").fetchone()[0] == 0
        conn.execute("DROP TRIGGER crash_fanout")
    assert not publish(store)["duplicate"]


def test_enqueue_callback_failure_after_insert_recovers_without_duplicate(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path)
    worker = dispatcher(store, queue)
    dispatch = worker.dispatch
    def interrupted(execution):
        dispatch(execution)
        raise RuntimeError("conversation projection interrupted")
    worker.dispatch = interrupted
    start = datetime.now(timezone.utc)
    identity = worker.run_once(now=start)[0]["execution_id"]
    assert queue.size() == 1
    worker.run_once(now=start + timedelta(seconds=5))
    assert queue.size() == 1
    assert store.get_execution(identity)["status"] == "queued"


def test_dispatch_does_not_hold_database_transaction(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path)
    worker = dispatcher(store, queue)
    dispatch = worker.dispatch
    def check(execution):
        # A separate writer can commit while the handoff callback is active.
        with connect_database(path) as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("UPDATE v2_event_workers SET display_name='still writable'")
        return dispatch(execution)
    worker.dispatch = check
    assert worker.run_once()[0]["status"] == "queued"


def test_migrates_legacy_claimed_and_enqueued_rows_without_changing_ids(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    execution = store.claim_event_executions()[0]
    identity = execution["execution_id"]
    # Rebuild the shipped pre-lease execution shape, including its stuck claim.
    with connect_database(path) as conn:
        conn.executescript("""
            ALTER TABLE v2_automation_executions RENAME TO old_executions;
            CREATE TABLE v2_automation_executions(
                execution_id TEXT PRIMARY KEY,automation_id TEXT NOT NULL,event_key TEXT NOT NULL,
                status TEXT NOT NULL,queued_message_id TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,updated_at TEXT NOT NULL,UNIQUE(automation_id,event_key)) STRICT;
            INSERT INTO v2_automation_executions
                SELECT execution_id,automation_id,event_key,status,queued_message_id,created_at,updated_at
                FROM old_executions;
            DROP TABLE old_executions;
            DROP TABLE v2_event_receipts;
        """)
    migrated = EventAutomationStore(path)
    assert publish(migrated)["duplicate"]
    queue = SQLiteMessageQueue(path)
    dispatcher(migrated, queue).run_once()
    assert migrated.get_execution(identity)["status"] == "queued"
    assert queue.peek().message_id == f"event:{identity}"
    with connect_database(path) as conn:
        conn.execute("UPDATE v2_automation_executions SET status='enqueued'")
    assert EventAutomationStore(path).get_execution(identity)["status"] == "queued"


def test_parked_event_is_processing_until_original_checkpoint_finishes(tmp_path):
    from types import SimpleNamespace
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host

    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path)
    worker = dispatcher(store, queue)
    identity = worker.run_once()[0]["execution_id"]
    queued = queue.dequeue()
    assert queue.ack(queued.message_id, outcome="processing")
    worker.run_once()
    assert store.get_execution(identity)["status"] == "processing"
    from alphonse.agent_v2.users import V2UserStore
    users = V2UserStore()
    users.set_users_root(tmp_path / "users")
    daemon = V2Daemon(build_runtime_host(messages=queue, user_store=users), event_store=store)
    daemon._mark_snapshot_checkpoint_terminal(SimpleNamespace(metadata={
        "task_state": {"metadata": {"automation_execution_id": identity}}
    }), status="done")
    assert store.get_execution(identity)["status"] == "completed"
    assert queue.status_for(queued.message_id) == "completed"


def test_terminal_failure_survives_pruned_queue_row(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path)
    identity = dispatcher(store, queue).run_once()[0]["execution_id"]
    queued = queue.dequeue()
    queue.retry(queued.message_id, max_attempts=1)
    # Model a crash before the execution lifecycle callback, followed by cleanup.
    with connect_database(path) as conn:
        conn.execute("DELETE FROM v2_inbound_messages WHERE status='failed'")
    dispatcher(store, queue).run_once()
    assert store.get_execution(identity)["status"] == "failed"


def test_recovery_expiry_uses_acceptance_time(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path)
    future = datetime.now(timezone.utc) + timedelta(days=91)
    outcomes = dispatcher(store, queue).run_once(now=future)
    assert outcomes[0]["status"] == "cancelled"
    assert queue.size() == 0


def test_cancel_cannot_overwrite_completion_receipt(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path)
    worker = dispatcher(store, queue)
    identity = worker.run_once()[0]["execution_id"]
    queued = queue.dequeue()
    queue.ack(queued.message_id)
    assert not worker.cancel(identity)
    worker.run_once()
    assert store.get_execution(identity)["status"] == "completed"


@pytest.mark.parametrize("outcome,expected", [
    ("completed", "completed"), ("cancelled", "cancelled"), ("failed", "failed"),
    ("parked", "processing"), ("waiting", "processing"),
])
def test_daemon_event_lifecycle_callbacks(tmp_path, outcome, expected):
    from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host

    path = tmp_path / "events.db"
    store = setup_store(path)
    queue = SQLiteMessageQueue(path)
    class Processor:
        def process(self, task, context):
            task.status = outcome
            return ProcessingResult(status=ProcessingStatus(outcome),
                                    snapshot=StateSnapshot(task_owner=task.user,
                                                           metadata={"task_state": task.to_dict()}),
                                    error="synthetic failure" if outcome == "failed" else "")
    from alphonse.agent_v2.users import V2UserStore
    users = V2UserStore()
    users.set_users_root(tmp_path / "users")
    daemon = V2Daemon(build_runtime_host(messages=queue, processor=Processor(), user_store=users),
                      event_store=store, inbound_max_attempts=1)
    daemon.publish_event(worker_id="sensor", event_id="one", event_type="sensor", event_version="1", payload={})
    queued = queue.peek()
    identity = queued.message.metadata["automation_execution_id"]
    daemon.run_once()
    assert store.get_execution(identity)["status"] == expected
    assert queue.status_for(queued.message_id) == expected


def test_queue_receipt_rolls_back_with_failed_insertion(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path)
    worker = dispatcher(store, queue)
    execution = store.claim_event_executions()[0]
    with connect_database(path) as conn:
        conn.execute("""CREATE TRIGGER crash_queue BEFORE INSERT ON v2_inbound_messages
                        BEGIN SELECT RAISE(ABORT,'interrupted insertion'); END""")
    with pytest.raises(Exception, match="interrupted insertion"):
        worker.dispatch(execution)
    assert queue.status_for(f"event:{execution['execution_id']}") == ""
    with connect_database(path) as conn:
        conn.execute("DROP TRIGGER crash_queue")
    worker.dispatch(execution)
    assert queue.size() == 1


def test_cancellation_inside_inflight_handoff_never_enqueues(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path)
    worker = dispatcher(store, queue)
    dispatch = worker.dispatch
    def cancelled_handoff(execution):
        assert worker.cancel(execution["execution_id"])
        return dispatch(execution)
    worker.dispatch = cancelled_handoff
    identity = worker.run_once()[0]["execution_id"]
    assert queue.size() == 0
    assert store.get_execution(identity)["status"] == "cancelled"


def test_tick_recovers_inbound_lease_that_expires_after_restart(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path, lease_owner="old-daemon")
    dispatcher(store, queue).run_once()
    queued = queue.dequeue()
    restarted = dispatcher(EventAutomationStore(path), SQLiteMessageQueue(path, lease_owner="new-daemon"))
    now = datetime.now(timezone.utc)
    restarted.run_once(now=now)
    assert restarted.messages.size() == 0
    restarted.run_once(now=now + timedelta(seconds=61))
    assert restarted.messages.peek().message_id == queued.message_id


def test_tick_preserves_current_daemon_long_running_model_call(tmp_path):
    path = tmp_path / "events.db"
    store = setup_store(path)
    publish(store)
    queue = SQLiteMessageQueue(path, lease_owner="test")
    worker = dispatcher(store, queue)
    worker.run_once()
    queued = queue.dequeue()
    worker.run_once(now=datetime.now(timezone.utc) + timedelta(seconds=120))
    assert queue.status_for(queued.message_id) == "processing"
    assert queue.size() == 0
