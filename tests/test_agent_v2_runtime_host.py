from __future__ import annotations

from datetime import datetime, timedelta, timezone

from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.core.io import ChannelAddress
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.services.scheduled_worker import ScheduledTaskWorker


def _respond_router() -> InferenceRouter:
    return InferenceRouter(
        provider=StubInferenceProvider(
            markdown_by_purpose={
                InferencePurpose.ACCEPTANCE_CRITERIA: "1.- [ ] User receives a greeting",
                InferencePurpose.CRITERIA_REVIEW: "1.- [x] User receives a greeting",
            },
            tool_call={
                "tool_id": "native.respond",
                "tool_name": "respond",
                "arguments": {"message": "Hello, Alex.", "tone": "warm"},
                "internal_state": "Answering the greeting.",
            },
        ),
        default_profile=ModelProfile(provider="stub", model="stub", profile_id="stub"),
    )


def test_runtime_host_builds_core_and_shared_stores() -> None:
    runtime = build_runtime_host(
        user="local",
        schedule_store=ScheduledTaskStore(":memory:"),
    )

    assert runtime.core.messages is runtime.queue
    assert runtime.channel.messages is runtime.queue
    assert runtime.core.schedule_store is runtime.schedule_store
    assert runtime.presence_projector is not None


def test_scheduled_worker_dispatches_due_task_and_records_stats() -> None:
    store = ScheduledTaskStore(":memory:")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    task = store.create_task(
        owner_user_id="u-alex",
        project_id="home",
        name="Reminder",
        prompt="Remind Alex",
        schedule_kind="once",
        run_at=(now - timedelta(seconds=1)).isoformat(),
        timezone_name="UTC",
        now=now - timedelta(minutes=1),
    )
    runtime = build_runtime_host(schedule_store=store)
    worker = ScheduledTaskWorker(store=store, messages=runtime.queue)

    outcomes = worker.run_once(now=now)

    assert outcomes[0]["status"] == "queued"
    assert worker.stats.queued == 1
    assert store.get_task(task.scheduled_task_id).status == "active"
    assert store.list_executions(scheduled_task_id=task.scheduled_task_id)[0].status == "enqueued"
    assert runtime.queue.size() == 1


def test_scheduled_worker_delivers_plain_reminders_without_queueing_pdca() -> None:
    store = ScheduledTaskStore(":memory:")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    task = store.create_task(
        owner_user_id="u-alex",
        project_id="home",
        name="Drink water",
        prompt="Reminder to drink water",
        schedule_kind="once",
        run_at=(now - timedelta(seconds=1)).isoformat(),
        timezone_name="UTC",
        delivery_mode="direct",
        now=now - timedelta(minutes=1),
    )
    runtime = build_runtime_host(schedule_store=store)
    worker = ScheduledTaskWorker(
        store=store,
        messages=runtime.queue,
        on_direct_delivery=lambda _occurrence: "outbox-1",
    )

    outcomes = worker.run_once(now=now)

    assert outcomes[0]["status"] == "delivered"
    assert runtime.queue.size() == 0
    assert store.list_executions(scheduled_task_id=task.scheduled_task_id)[0].status == "response_pending"


def test_scheduled_occurrence_is_claimed_by_one_worker_only() -> None:
    store = ScheduledTaskStore(":memory:")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    task = store.create_task(
        owner_user_id="u-alex",
        project_id="home",
        name="Reminder",
        prompt="Remind Alex",
        schedule_kind="once",
        run_at=(now - timedelta(seconds=1)).isoformat(),
        timezone_name="UTC",
        now=now - timedelta(minutes=1),
    )

    first = store.claim_due_occurrences(worker_id="worker-1", now=now, lease_seconds=60)
    second = store.claim_due_occurrences(worker_id="worker-2", now=now, lease_seconds=60)

    assert len(first) == 1
    assert first[0].task.scheduled_task_id == task.scheduled_task_id
    assert second == []


def test_scheduled_occurrence_expired_lease_is_reclaimed() -> None:
    store = ScheduledTaskStore(":memory:")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    store.create_task(
        owner_user_id="u-alex",
        project_id="home",
        name="Reminder",
        prompt="Remind Alex",
        schedule_kind="once",
        run_at=(now - timedelta(seconds=1)).isoformat(),
        timezone_name="UTC",
        now=now - timedelta(minutes=1),
    )

    first = store.claim_due_occurrences(worker_id="worker-1", now=now, lease_seconds=1)
    reclaimed = store.claim_expired_occurrences(
        worker_id="worker-2",
        now=now + timedelta(seconds=2),
        lease_seconds=60,
    )

    assert len(first) == 1
    assert len(reclaimed) == 1
    assert reclaimed[0].occurrence_key == first[0].occurrence_key


def test_daemon_processes_host_queue_without_tui() -> None:
    runtime = build_runtime_host(
        schedule_store=ScheduledTaskStore(":memory:"),
        inference=_respond_router(),
    )
    daemon = V2Daemon(runtime)

    runtime.channel.queue_message(prompt="hello", user="local", project_id="home")
    result = daemon.run_once()

    assert result.queued_message_id is not None


def test_daemon_stop_is_idempotent(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_V2_SOCKET_PATH", str(tmp_path / "v2-daemon.sock"))
    runtime = build_runtime_host(schedule_store=ScheduledTaskStore(":memory:"))
    daemon = V2Daemon(runtime)

    daemon.stop()
    daemon.stop()


def test_daemon_marks_scheduled_occurrence_delivered_after_outbox_delivery() -> None:
    store = ScheduledTaskStore(":memory:")
    runtime = build_runtime_host(schedule_store=store)
    daemon = V2Daemon(runtime)
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    task = store.create_task(
        owner_user_id="u-alex",
        project_id="home",
        name="Reminder",
        prompt="Remind Alex",
        schedule_kind="once",
        run_at=(now - timedelta(seconds=1)).isoformat(),
        timezone_name="UTC",
        now=now - timedelta(minutes=1),
    )
    occurrence = store.claim_due_occurrences(worker_id="worker-1", now=now)[0]
    outbound = runtime.outbox.enqueue(
        address=ChannelAddress(
            integration_id="telegram-home",
            provider_key="telegram",
            channel_target="123",
            alphonse_user_id="u-alex",
        ),
        message="Reminder",
        metadata={"occurrence_key": occurrence.occurrence_key},
    )

    daemon._on_outbox_delivered(outbound)

    executions = store.list_executions(scheduled_task_id=task.scheduled_task_id)
    assert executions[0].status == "delivered"
    assert executions[0].response_outbox_id == outbound.outbox_message_id
    assert store.get_task(task.scheduled_task_id).status == "completed"


def test_scheduled_processing_failure_marks_task_failed_and_notifies_owner() -> None:
    store = ScheduledTaskStore(":memory:")
    runtime = build_runtime_host(schedule_store=store)
    daemon = V2Daemon(runtime)
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    task = store.create_task(
        owner_user_id="u-alex",
        project_id="home",
        name="Turn on study air",
        prompt="Turn on the air",
        schedule_kind="once",
        run_at=(now - timedelta(seconds=1)).isoformat(),
        timezone_name="UTC",
        origin_channel={"integration_id": "telegram-home", "provider_key": "telegram", "channel_target": "123", "alphonse_user_id": "u-alex"},
        now=now - timedelta(minutes=1),
    )
    occurrence = store.claim_due_occurrences(worker_id="worker-1", now=now)[0]
    store.mark_occurrence_enqueued(occurrence.occurrence_key, worker_id="worker-1", message_id="scheduled:test")
    store.mark_occurrence_processing_failed(occurrence.occurrence_key, error="openai_codex_auth_required")
    daemon._notify_scheduled_task_failure(
        {"scheduled_task_id": task.scheduled_task_id, "channel": task.origin_channel},
        error="openai_codex_auth_required",
    )

    assert store.get_task(task.scheduled_task_id).status == "failed"
    assert store.list_executions(scheduled_task_id=task.scheduled_task_id)[0].status == "failed"
    notification = runtime.outbox.list()[0]
    assert notification.kind == "scheduled_task_failed"
    assert "signed in again" in notification.message
