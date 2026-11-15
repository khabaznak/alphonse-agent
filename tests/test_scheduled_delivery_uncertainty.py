"""Synthetic scheduled sends cannot be repeated after an ambiguous outcome."""
from datetime import datetime, timedelta, timezone
import sqlite3

from alphonse.agent_v2.core.io.channels import ChannelAddress
from alphonse.agent_v2.core.io.outbox import OutboundSelector, SQLiteOutboundStore


def _enqueue(store, occurrence):
    return store.enqueue(
        address=ChannelAddress(integration_id="fixture", provider_key="fixture", channel_target="fixture-user"),
        message="Synthetic reminder", kind="scheduled_reminder", metadata={"occurrence_key": occurrence},
    )


def test_uncertain_send_is_not_retried_after_restart(tmp_path):
    path = tmp_path / "outbox.sqlite3"
    store = SQLiteOutboundStore(path)
    outbound = _enqueue(store, "fixture-occurrence")
    assert store.claim_next(OutboundSelector(integration_id="fixture")) is not None
    assert store.mark_failed(outbound.outbox_message_id, error="synthetic timeout")

    restarted = SQLiteOutboundStore(path)
    restored = restarted.get(outbound.outbox_message_id)
    assert restored.status == "unconfirmed"
    assert restored.metadata["delivery_state"] == "unconfirmed"
    assert restarted.claim_next(OutboundSelector(integration_id="fixture")) is None


def test_expired_scheduled_send_is_unconfirmed_instead_of_reclaimed(tmp_path):
    store = SQLiteOutboundStore(tmp_path / "outbox.sqlite3")
    outbound = _enqueue(store, "fixture-occurrence")
    store.claim_next(OutboundSelector(integration_id="fixture"), lease_seconds=1)
    assert store.reclaim_expired(now=datetime.now(timezone.utc) + timedelta(seconds=2)) == 1
    assert store.get(outbound.outbox_message_id).status == "unconfirmed"
    assert store.claim_next(OutboundSelector(integration_id="fixture")) is None




def test_outbox_status_migration_preserves_retry_wait(tmp_path):
    path = tmp_path / "outbox.sqlite3"
    store = SQLiteOutboundStore(path)
    outbound = _enqueue(store, "")
    store.claim_next(OutboundSelector(integration_id="fixture"))
    store.mark_failed(outbound.outbox_message_id, error="temporary", retry_after_seconds=60)
    with sqlite3.connect(path) as conn:
        schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()[0]
        schema = schema.replace(", 'unconfirmed'", "")
        conn.execute("ALTER TABLE v2_outbox RENAME TO old_outbox")
        conn.execute(schema)
        conn.execute("INSERT INTO v2_outbox SELECT * FROM old_outbox")
        conn.execute("DROP TABLE old_outbox")

    restarted = SQLiteOutboundStore(path)
    assert restarted.get(outbound.outbox_message_id).status == "retry_wait"
    assert restarted.status_counts()["retry_wait"] == 1


def test_migrated_scheduled_retry_with_prior_attempt_becomes_unconfirmed(tmp_path):
    path = tmp_path / "outbox.sqlite3"
    store = SQLiteOutboundStore(path)
    outbound = _enqueue(store, "fixture-occurrence")
    store.claim_next(OutboundSelector(integration_id="fixture"))
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE v2_outbox SET status='retry_wait' WHERE outbox_message_id=?", (outbound.outbox_message_id,))
        schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()[0]
        schema = schema.replace(", 'unconfirmed'", "")
        conn.execute("ALTER TABLE v2_outbox RENAME TO old_outbox")
        conn.execute(schema)
        conn.execute("INSERT INTO v2_outbox SELECT * FROM old_outbox")
        conn.execute("DROP TABLE old_outbox")
    restarted = SQLiteOutboundStore(path)
    assert restarted.get(outbound.outbox_message_id).status == "unconfirmed"
    assert restarted.claim_next(OutboundSelector(integration_id="fixture")) is None


def test_schedule_status_migration_preserves_existing_occurrence(tmp_path):
    from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore

    path = tmp_path / "schedule.sqlite3"
    store = ScheduledTaskStore(path)
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    task = store.create_task(owner_user_id="alex", name="Fixture", prompt="Fixture", schedule_kind="once",
                             run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                             now=now - timedelta(minutes=1))
    occurrence = store.claim_due_occurrences(worker_id="fixture", now=now)[0]
    store.mark_occurrence_enqueued(occurrence.occurrence_key, worker_id="fixture", message_id="fixture-message")
    with sqlite3.connect(path) as conn:
        schema = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='v2_scheduled_task_executions'").fetchone()[0]
        schema = schema.replace(", 'unconfirmed', 'cancelled'", "")
        conn.execute("ALTER TABLE v2_scheduled_task_executions RENAME TO old_executions")
        conn.execute(schema)
        conn.execute("INSERT INTO v2_scheduled_task_executions SELECT * FROM old_executions")
        conn.execute("DROP TABLE old_executions")
    restarted = ScheduledTaskStore(path)
    assert restarted.list_executions(scheduled_task_id=task.scheduled_task_id)[0].status == "enqueued"
    assert restarted.mark_occurrence_delivery_unconfirmed(occurrence.occurrence_key, response_outbox_id="fixture-outbox")
    assert restarted.list_executions(scheduled_task_id=task.scheduled_task_id)[0].status == "unconfirmed"
    assert not restarted.mark_question_delivery_confirmed(occurrence.occurrence_key, response_outbox_id="other-outbox")
    assert restarted.mark_question_delivery_confirmed(occurrence.occurrence_key, response_outbox_id="fixture-outbox")
    assert restarted.list_executions(scheduled_task_id=task.scheduled_task_id)[0].status == "processing"
