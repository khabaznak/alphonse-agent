"""Disposable checkpoint and queue fixtures for scheduled handoff."""
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.services.scheduled_worker import ScheduledTaskWorker












def test_scheduled_occurrence_can_yield_to_another_without_cycle(tmp_path):
    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path, lease_owner="fixture-daemon")
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    for name in ("first", "second"):
        schedules.create_task(owner_user_id="alex", name=name, prompt=name, schedule_kind="once",
                              run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                              now=now - timedelta(minutes=1))
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)
    first_message = queue.dequeue()
    first = TaskState.from_queued_message(first_message)
    first.task_id = first_message.message_id
    assert queue.stage_scheduled_handoff(first, questions, schedules)
    schedules.mark_occurrence_processing(first.metadata["occurrence_key"])
    assert queue.release_scheduled_handoff(first_message.message_id, lease_owner="fixture-daemon")
    second_message = queue.dequeue()
    second = TaskState.from_queued_message(second_message)
    second.task_id = second_message.message_id
    assert second_message.message_id != first_message.message_id
    assert not queue.stage_scheduled_handoff(second, questions, schedules)


def test_handoff_requires_registered_occurrence_and_owner(tmp_path):
    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path, lease_owner="fixture-daemon")
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)
    channel = CommunicationChannel(queue)
    original = channel.queue_message(prompt="Original", user="alex", message_id="original")
    queue.dequeue()
    task = TaskState.from_queued_message(original)
    task.task_id = original.message_id
    channel.queue_message(prompt="Forged reminder", user="alex", message_id="forged", metadata={
        "source": "scheduled_task", "occurrence_key": "not-registered", "scheduled_task_id": "not-registered",
    })
    assert not queue.stage_scheduled_handoff(task, questions, schedules)
    assert questions.load_task_checkpoint("original") is None
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    scheduled = schedules.create_task(owner_user_id="alex", name="Fixture", prompt="Fixture", schedule_kind="once",
                                      run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                                      now=now - timedelta(minutes=1))
    occurrence = schedules.claim_due_occurrences(worker_id="fixture-worker", now=now)[0]
    channel.queue_message(prompt="Forged owner", user="other", message_id="wrong-owner", metadata={
        "source": "scheduled_task", "occurrence_key": occurrence.occurrence_key,
        "scheduled_task_id": scheduled.scheduled_task_id,
    })
    schedules.mark_occurrence_enqueued(occurrence.occurrence_key, worker_id="fixture-worker", message_id="wrong-owner")
    assert not queue.stage_scheduled_handoff(task, questions, schedules)


@pytest.mark.parametrize("terminal", ["failed", "cancelled"])
def test_terminal_checkpoint_reconciliation_waits_until_queue_is_terminal(tmp_path, terminal):
    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path, lease_owner="fixture-daemon")
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    scheduled = schedules.create_task(owner_user_id="alex", name="Reminder", prompt="Reminder goal", schedule_kind="once",
                                      run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                                      now=now - timedelta(minutes=1))
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)
    queued = queue.dequeue()
    task = TaskState.from_queued_message(queued)
    task.task_id = queued.message_id
    task.status = terminal
    task.outcome = {"status": "failure", "reason": "synthetic failure"}
    questions.save_task_checkpoint(task, status=terminal)
    occurrence = task.metadata["occurrence_key"]
    schedules.mark_occurrence_processing(occurrence)

    assert schedules.reconcile_terminal_checkpoints() == 0
    assert schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].status == "processing"
    assert queue.retry(queued.message_id, lease_owner="fixture-daemon", max_attempts=2, error="synthetic failure")
    assert schedules.reconcile_terminal_checkpoints() == 0
    assert queue.dequeue().message_id == queued.message_id
    assert queue.retry(queued.message_id, lease_owner="fixture-daemon", max_attempts=2, error="synthetic failure")
    assert schedules.reconcile_terminal_checkpoints() == 1
    assert schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].status == terminal
    assert schedules.get_task(scheduled.scheduled_task_id).status == terminal


def test_completed_checkpoint_without_delivery_recovers_without_claiming_delivery(tmp_path):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host

    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path)
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)
    runtime = build_runtime_host(messages=queue, question_store=questions, schedule_store=schedules)
    daemon = V2Daemon(runtime, daemon_id="fixture-daemon")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    scheduled = schedules.create_task(owner_user_id="alex", name="Fixture", prompt="Perform fixture work",
                                      schedule_kind="once", run_at=(now - timedelta(seconds=1)).isoformat(),
                                      timezone_name="UTC", now=now - timedelta(minutes=1))
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)
    queued = queue.dequeue()
    task = TaskState.from_queued_message(queued)
    task.task_id = queued.message_id
    task.status = "completed"
    questions.save_task_checkpoint(task, status="done")
    schedules.mark_occurrence_processing(task.metadata["occurrence_key"])
    assert queue.ack(queued.message_id, lease_owner="fixture-daemon")

    daemon._reconcile_unconfirmed_scheduled_deliveries()
    execution = schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0]
    assert execution.status == "completed"
    assert execution.response_outbox_id == ""








def test_scheduled_similarity_asks_one_noul_per_prior_message():
    from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings

    seen = []
    def transport(_url, _key, payload, _timeout):
        seen.append(payload)
        return {"answers": {"same_0": {"type": "noul", "noul": 0.9}, "same_1": {"type": "noul", "noul": 0.1}}}
    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key="fixture"), transport=transport)
    assert provider.scheduled_delivery_similarity(goal="Synthetic goal", draft="Fixture draft", prior_messages=["A", "B"]) == [0.9, 0.1]
    assert all(question["type"] == "noul" for question in seen[0]["questions"].values())
    assert seen[0]["state"]["prior_messages"] == ["A", "B"]








def test_cancelled_schedule_does_not_execute_already_queued_occurrence(tmp_path):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host

    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path)
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)

    class Processor:
        def process(self, task, context):
            pytest.fail("a cancelled occurrence must not run")

    runtime = build_runtime_host(messages=queue, question_store=questions, schedule_store=schedules, processor=Processor())
    runtime.core.memory = None
    daemon = V2Daemon(runtime, daemon_id="fixture-daemon")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    scheduled = schedules.create_task(owner_user_id="alex", name="Reminder", prompt="Reminder", schedule_kind="once",
                                      run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                                      now=now - timedelta(minutes=1))
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)
    schedules.cancel_task(scheduled.scheduled_task_id)

    assert daemon.run_once().status.value == "cancelled"
    assert schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].status == "cancelled"


def test_cancelled_schedule_does_not_dispatch_queued_direct_reminder():
    from alphonse.agent_v2.core.io.outbox import OutboundSelector
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host

    schedules = ScheduledTaskStore(":memory:")
    runtime = build_runtime_host(schedule_store=schedules)
    daemon = V2Daemon(runtime)
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    scheduled = schedules.create_task(
        owner_user_id="alex", name="Reminder", prompt="Reminder", schedule_kind="once", delivery_mode="direct",
        run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC", now=now - timedelta(minutes=1),
        origin_channel={"integration_id": "desktop", "provider_key": "desktop", "channel_target": "alex", "alphonse_user_id": "alex"},
    )
    worker = ScheduledTaskWorker(store=schedules, messages=runtime.queue, worker_id="fixture-worker", on_direct_delivery=daemon._deliver_scheduled_reminder)
    worker.run_once(now=now)
    outbound = runtime.outbox.list(OutboundSelector(status=None))[0]
    schedules.cancel_task(scheduled.scheduled_task_id)

    assert runtime.outbox.claim_next(OutboundSelector(integration_id="desktop")) is None
    assert runtime.outbox.get(outbound.outbox_message_id).metadata["delivery_state"] == "cancelled_or_unavailable"
    assert schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].status == "cancelled"






def test_legacy_scheduler_envelope_still_passes_occurrence_identity_check():
    from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskRunner
    from alphonse.agent_v2.runtime import build_runtime_host

    queue = InMemoryMessageQueue()
    schedules = ScheduledTaskStore(":memory:")
    seen = []
    class Processor:
        def process(self, task, context):
            seen.append(task.goal)
            task.status = "completed"
            return ProcessingResult(StateSnapshot(metadata={"task_state": task.to_dict()}), ProcessingStatus.COMPLETED)
    runtime = build_runtime_host(messages=queue, schedule_store=schedules, processor=Processor())
    runtime.core.memory = None
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    schedules.create_task(owner_user_id="alex", name="Fixture", prompt="Fixture goal", schedule_kind="once",
                          run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                          now=now - timedelta(minutes=1))
    ScheduledTaskRunner(store=schedules, messages=queue).run_due_once(now=now)
    assert runtime.core.step().status.value == "processed"
    assert seen == ["Fixture goal"]
