from __future__ import annotations

from datetime import datetime, timedelta, timezone

from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.daemon import V2Daemon
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
    assert store.get_task(task.scheduled_task_id).status == "completed"
    assert runtime.queue.size() == 1


def test_daemon_processes_host_queue_without_tui() -> None:
    runtime = build_runtime_host(
        schedule_store=ScheduledTaskStore(":memory:"),
        inference=_respond_router(),
    )
    daemon = V2Daemon(runtime)

    runtime.channel.queue_message(prompt="hello", user="local")
    result = daemon.run_once()

    assert result.queued_message_id is not None
