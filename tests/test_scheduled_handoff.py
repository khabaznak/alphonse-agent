"""Disposable checkpoint and queue fixtures for scheduled handoff."""
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import PhasePlan, PhaseSubgoal, PhaseStatus, TacticalState, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor, ScheduledHandoff
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.services.scheduled_worker import ScheduledTaskWorker
from test_agent_v3_phase_executor import _accept_successful_actions, _selector, _tool


def test_tool_finishes_then_checkpoint_yields_and_resumes_after_reminder(tmp_path):
    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path, lease_owner="fixture-daemon")
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)
    channel = CommunicationChannel(queue)
    original = channel.queue_message(prompt="Do the original task", user="alex", message_id="original")
    queue.dequeue()
    task = TaskState.from_queued_message(original)
    task.task_id = original.message_id
    task.acceptance_criteria_md = "- [ ] Finish the original task"
    task.ensure_acceptance_contract()
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    schedules.create_task(
        owner_user_id="alex", name="Synthetic reminder", prompt="Independent scheduled work",
        schedule_kind="once", run_at=(now - timedelta(seconds=1)).isoformat(),
        timezone_name="UTC", now=now - timedelta(minutes=1),
    )
    worker = ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker")
    calls = []
    def first(_arguments):
        calls.append("first")
        worker.run_once(now=now)
        return {"found": True}
    def second(_arguments):
        calls.append("second")
        return {"done": True}
    registry = InMemoryToolRegistry()
    registry.register(_tool("native.first", first, read_only=True))
    registry.register(_tool("native.second", second, read_only=True))
    phase = PhasePlan("fixture-phase", "Original work", (
        PhaseSubgoal("first", "Find", "first", allowed_capabilities=("native.first",)),
        PhaseSubgoal("second", "Finish", "second", depends_on=("first",), allowed_capabilities=("native.second",)),
    ), authorized_capabilities=("native.first", "native.second"))
    state = new_tactical_state(phase)
    state.revealed_tool_ids = ["native.first"]
    context = CoreLoopContext(messages=queue, tools=registry, question_store=questions,
                              schedule_store=schedules, system_one=_accept_successful_actions(), scheduled_handoff_enabled=True)
    executor = PhaseExecutor(action_selector=_selector([
        {"tool_id": "native.first", "arguments": {}}, {"tool_id": "native.second", "arguments": {}},
    ]))

    with pytest.raises(ScheduledHandoff):
        executor.run(task, state, context)
    assert calls == ["first"]
    assert queue.release_scheduled_handoff(original.message_id, lease_owner="fixture-daemon")

    restarted = SQLiteMessageQueue(path, lease_owner="fixture-daemon")
    reminder = restarted.dequeue()
    assert reminder.message.metadata["source"] == "scheduled_task"
    assert restarted.ack(reminder.message_id, lease_owner="fixture-daemon")
    resumed_message = restarted.dequeue()
    assert resumed_message.message_id == original.message_id
    resumed = questions.load_task_checkpoint(restarted.handoff_task_id(original.message_id, "alex"))
    assert resumed.goal == task.goal
    assert resumed.acceptance_contract == task.acceptance_contract
    resumed_state = TacticalState.from_dict(resumed.hierarchical_state)
    assert resumed_state.completed_subgoal_ids == ["first"]
    assert resumed_state.evidence.entries[0]["result"] == {"found": True}
    context.messages = restarted
    outcome = PhaseExecutor(action_selector=_selector([
        {"tool_id": "native.second", "arguments": {}},
    ])).run(resumed, resumed_state, context)
    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert calls == ["first", "second"]


def test_undeliverable_direct_reminder_does_not_strand_active_work():
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue

    queue = InMemoryMessageQueue()
    schedules = ScheduledTaskStore(":memory:")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    scheduled = schedules.create_task(owner_user_id="alex", name="Reminder", prompt="Reminder", schedule_kind="once",
                                      delivery_mode="direct", run_at=(now - timedelta(seconds=1)).isoformat(),
                                      timezone_name="UTC", now=now - timedelta(minutes=1))
    def fail_delivery(_occurrence):
        raise RuntimeError("synthetic destination unavailable")
    worker = ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker", on_direct_delivery=fail_delivery)
    calls = []
    def first(_arguments):
        calls.append("first")
        worker.run_once(now=now)
        return {"found": True}
    def second(_arguments):
        calls.append("second")
        return {"done": True}
    registry = InMemoryToolRegistry()
    registry.register(_tool("native.first", first, read_only=True))
    registry.register(_tool("native.second", second, read_only=True))
    phase = PhasePlan("fixture", "Original work", (
        PhaseSubgoal("first", "Find", "first", allowed_capabilities=("native.first",)),
        PhaseSubgoal("second", "Finish", "second", depends_on=("first",), allowed_capabilities=("native.second",)),
    ), authorized_capabilities=("native.first", "native.second"))
    state = new_tactical_state(phase)
    state.revealed_tool_ids = ["native.first"]
    outcome = PhaseExecutor(action_selector=_selector([
        {"tool_id": "native.first", "arguments": {}}, {"tool_id": "native.second", "arguments": {}},
    ])).run(TaskState(user="alex", goal="Original"), state,
            CoreLoopContext(messages=queue, tools=registry, system_one=_accept_successful_actions()))

    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert calls == ["first", "second"]
    assert schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].status == "retry_wait"


@pytest.mark.parametrize("reminder_result", ["parked", "failed", "cancelled"])
def test_daemon_resumes_original_when_reminder_does_not_complete(tmp_path, reminder_result):
    from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host

    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path)
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)
    seen = []

    class Processor:
        def process(self, task, context):
            seen.append((task.goal, task.metadata.get("saved_evidence")))
            if task.metadata.get("source") == "scheduled_task":
                task.status = {"parked": "waiting_user", "failed": "failed", "cancelled": "cancelled"}[reminder_result]
                questions.save_task_checkpoint(task, status=task.status)
                return ProcessingResult(
                    StateSnapshot(metadata={"task_state": task.to_dict()}), ProcessingStatus(reminder_result),
                    error="v3_task_failed:synthetic" if reminder_result == "failed" else None,
                )
            if not task.metadata.get("saved_evidence"):
                task.metadata["saved_evidence"] = "fixture evidence"
                assert queue.stage_scheduled_handoff(task, questions, schedules)
                return ProcessingResult(StateSnapshot(metadata={"task_state": task.to_dict()}), ProcessingStatus.YIELDED)
            task.status = "completed"
            return ProcessingResult(StateSnapshot(metadata={"task_state": task.to_dict()}), ProcessingStatus.COMPLETED)

    runtime = build_runtime_host(messages=queue, question_store=questions, schedule_store=schedules, processor=Processor())
    runtime.core.memory = None
    daemon = V2Daemon(runtime, daemon_id="fixture-daemon")
    runtime.channel.queue_message(prompt="Original goal", user="alex", message_id="original")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    schedules.create_task(owner_user_id="alex", name="Reminder", prompt="Reminder goal", schedule_kind="once",
                          run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                          now=now - timedelta(minutes=1))
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)

    assert daemon.run_once().status.value == "yielded"
    assert daemon.run_once().status.value == reminder_result
    assert daemon.run_once().status.value == "processed"
    assert seen == [("Original goal", None), ("Reminder goal", None), ("Original goal", "fixture evidence")]


def test_cancellation_at_handoff_cancels_checkpoint_without_resuming(tmp_path):
    from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host

    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path)
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)

    class Processor:
        def process(self, task, context):
            assert queue.stage_scheduled_handoff(task, questions, schedules)
            daemon.kill_switch.request_cancel()
            return ProcessingResult(StateSnapshot(metadata={"task_state": task.to_dict()}), ProcessingStatus.YIELDED)

    runtime = build_runtime_host(messages=queue, question_store=questions, schedule_store=schedules, processor=Processor())
    runtime.core.memory = None
    daemon = V2Daemon(runtime, daemon_id="fixture-daemon")
    runtime.channel.queue_message(prompt="Original goal", user="alex", message_id="original")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    schedules.create_task(owner_user_id="alex", name="Reminder", prompt="Reminder goal", schedule_kind="once",
                          run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                          now=now - timedelta(minutes=1))
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)

    assert daemon.run_once().status.value == "cancelled"
    assert questions.load_task_checkpoint("original").status == "cancelled"
    assert queue.handoff_task_id("original", "alex") == ""
    assert queue.peek().message.metadata["source"] == "scheduled_task"


def test_crash_before_lease_release_keeps_checkpoint_and_can_yield_again(tmp_path):
    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path, lease_owner="first-daemon")
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)
    queued = CommunicationChannel(queue).queue_message(prompt="Original goal", user="alex", message_id="original")
    queue.dequeue()
    task = TaskState.from_queued_message(queued)
    task.task_id = "original"
    task.metadata["saved_evidence"] = "before crash"
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    schedules.create_task(owner_user_id="alex", name="Reminder", prompt="Reminder goal", schedule_kind="once",
                          run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                          now=now - timedelta(minutes=1))
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)
    assert queue.stage_scheduled_handoff(task, questions, schedules)

    restarted = SQLiteMessageQueue(path, lease_owner="new-daemon")
    assert restarted.reclaim_expired(now=datetime.now(timezone.utc) + timedelta(minutes=2)) == 1
    recovered_message = restarted.dequeue()
    restored = questions.load_task_checkpoint(restarted.handoff_task_id(recovered_message.message_id, "alex"))
    assert restored.metadata["saved_evidence"] == "before crash"
    assert restarted.stage_scheduled_handoff(restored, questions, schedules)
    assert restarted.release_scheduled_handoff("original", lease_owner="new-daemon")
    assert restarted.peek().message.metadata["source"] == "scheduled_task"


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


def test_late_provider_confirmation_requeues_only_matching_unconfirmed_task(tmp_path):
    from alphonse.agent_v2.core.io.channels import ChannelAddress
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore

    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path, lease_owner="fixture-daemon")
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)
    outbox = SQLiteOutboundStore(path)
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    scheduled = schedules.create_task(owner_user_id="alex", name="Fixture", prompt="Fixture", schedule_kind="once",
                                      run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                                      now=now - timedelta(minutes=1))
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)
    queued = queue.dequeue()
    task = TaskState.from_queued_message(queued)
    task.task_id = queued.message_id
    outbound = outbox.enqueue(
        address=ChannelAddress(integration_id="fixture", provider_key="fixture", channel_target="target"),
        message="Synthetic", audience_user_id="recipient", task_id=task.task_id,
        metadata={"occurrence_key": task.metadata["occurrence_key"], "scheduled_delivery_evidence": True},
    )
    task.status = "failed"
    task.outcome = {"status": "failure", "reason": "scheduled_delivery_outcome_unavailable"}
    task.metadata["v3_unconfirmed_delivery_outbox_id"] = outbound.outbox_message_id
    questions.save_task_checkpoint(task, status="failed")
    assert queue.retry(queued.message_id, lease_owner="fixture-daemon", max_attempts=1, error="synthetic")
    assert schedules.mark_occurrence_delivery_unconfirmed(task.metadata["occurrence_key"], response_outbox_id=outbound.outbox_message_id)
    assert not queue.resume_confirmed_scheduled_delivery(outbound.outbox_message_id, questions)
    outbox.mark_delivered(outbound.outbox_message_id, provider_message_id="fixture-provider")
    assert queue.resume_confirmed_scheduled_delivery(outbound.outbox_message_id, questions)
    assert not queue.resume_confirmed_scheduled_delivery(outbound.outbox_message_id, questions)
    assert queue.peek().message_id == queued.message_id
    assert questions.load_task_checkpoint(task.task_id).status == "running"
    assert schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].status == "processing"


def test_resumed_scheduled_delivery_with_unknown_outcome_is_not_sent_again():
    from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalAction
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue

    phase = PhasePlan("fixture", "Send reminder", (
        PhaseSubgoal("send", "Send", "delivery", allowed_capabilities=("native.deliver_message",)),
    ), authorized_capabilities=("native.deliver_message",))
    state = new_tactical_state(phase)
    state.status = PhaseStatus.RUNNING
    state.actions.append(TacticalAction("pending-send", "send", "native.deliver_message", {"recipient": "alex", "message": "Reminder"}))
    task = TaskState(user="alex", metadata={"source": "scheduled_task", "occurrence_key": "fixture"})
    executor = PhaseExecutor(action_selector=lambda *_: pytest.fail("ambiguous delivery must not be dispatched again"))

    outcome = executor.run(task, state, CoreLoopContext(messages=InMemoryMessageQueue()))

    assert outcome.blockers == ("scheduled_delivery_outcome_unavailable",)
    assert task.metadata["v3_unconfirmed_delivery_action_id"] == "pending-send"


@pytest.mark.parametrize("probability,blocked", [(0.05, False), (0.5, True), (0.95, True)])
def test_semantic_repeat_check_is_bounded_and_uncertainty_stops_dispatch(probability, blocked):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalAction
    from alphonse.agent_v2.core.intelligence.v3.executor import _scheduled_delivery_may_repeat
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue

    phase = PhasePlan("next-phase", "Send reminder", (
        PhaseSubgoal("send", "Send", "delivery", allowed_capabilities=("native.deliver_message",)),
    ), authorized_capabilities=("native.deliver_message",))
    state = new_tactical_state(phase)
    state.evidence.append({
        "evidence_ref": "earlier", "phase_id": "prior-phase", "subgoal_id": "send",
        "tool_id": "native.deliver_message", "status": "success",
        "arguments": {"recipient": "A", "message": "Take the fixture item"},
        "result": {"status": "queued", "recipient_user_id": "person-1", "outbox_message_id": "outbox-1"},
    })
    task = TaskState(user="alex", goal="Send a reminder", metadata={"source": "scheduled_task"})
    action = TacticalAction("next", "send", "native.deliver_message", {"recipient": "Alias", "message": "Remember the fixture item"})
    jev = SimpleNamespace(settings=SimpleNamespace(no_threshold=0.2),
                          scheduled_delivery_similarity=lambda **kwargs: [probability])
    context = CoreLoopContext(messages=InMemoryMessageQueue(), system_one=jev,
                              delivery_sink=lambda event: {"status": "resolved", "recipient_user_id": "person-1"})
    assert _scheduled_delivery_may_repeat(task, state, action, context) is blocked
    context.system_one = None
    assert _scheduled_delivery_may_repeat(task, state, action, context)
    context.delivery_sink = lambda event: {"status": "resolved", "recipient_user_id": "person-2"}
    assert not _scheduled_delivery_may_repeat(task, state, action, context)


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


def test_executor_does_not_dispatch_semantically_repeated_scheduled_message():
    from types import SimpleNamespace
    from alphonse.agent_v2.core.intelligence.v3 import MutationScope, SideEffectClass
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue

    calls = []
    registry = InMemoryToolRegistry()
    registry.register(_tool("native.deliver_message", lambda args: calls.append(args) or {"status": "queued"}, read_only=False))
    phase = PhasePlan("next", "Remind", (
        PhaseSubgoal("send", "Send", "delivery", allowed_capabilities=("native.deliver_message",),
                     allowed_side_effects=(SideEffectClass.EXTERNAL_REVERSIBLE,)),
    ), authorized_capabilities=("native.deliver_message",), mutation_scope=MutationScope(allow_external_effects=True))
    state = new_tactical_state(phase)
    state.revealed_tool_ids = ["native.deliver_message"]
    state.evidence.append({
        "evidence_ref": "earlier", "phase_id": "previous", "subgoal_id": "send",
        "tool_id": "native.deliver_message", "status": "success",
        "arguments": {"recipient": "A", "message": "Take the fixture item"},
        "result": {"status": "queued", "recipient_user_id": "person-1", "outbox_message_id": "outbox-1"},
    })
    task = TaskState(user="alex", goal="Send a reminder", metadata={"source": "scheduled_task"})
    jev = SimpleNamespace(settings=SimpleNamespace(no_threshold=0.2), scheduled_delivery_similarity=lambda **kwargs: [0.95])
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=jev,
                              delivery_sink=lambda event: {"status": "resolved", "recipient_user_id": "person-1"})
    executor = PhaseExecutor(action_selector=_selector([
        {"tool_id": "native.deliver_message", "arguments": {"recipient": "Alias", "message": "Remember the fixture item"}},
    ]))
    outcome = executor.run(task, state, context)
    assert outcome.blockers == ("scheduled_delivery_repetition_unresolved",)
    assert calls == []


def test_real_question_answer_keeps_scheduled_occurrence_provenance(tmp_path):
    from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot
    from alphonse.agent_v2.core.io.outbox import OutboundSelector
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host

    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path)
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)

    class Processor:
        def process(self, task, context):
            if not task.metadata.get("question_interrupt"):
                question = questions.create_question(task=task, question="Choose a fixture value")
                self.question_id = question.question_id
                return ProcessingResult(StateSnapshot(metadata={"task_state": task.to_dict()}), ProcessingStatus.PARKED)
            task.status = "completed"
            task.metadata["prepared_user_response"] = {"message": "Synthetic result"}
            questions.save_task_checkpoint(task, status="done")
            return ProcessingResult(StateSnapshot(metadata={"task_state": task.to_dict()}), ProcessingStatus.COMPLETED)

    processor = Processor()
    runtime = build_runtime_host(messages=queue, question_store=questions, schedule_store=schedules, processor=processor)
    runtime.core.memory = None
    daemon = V2Daemon(runtime, daemon_id="fixture-daemon")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    scheduled = schedules.create_task(
        owner_user_id="alex", name="Reminder", prompt="Reminder goal", schedule_kind="once",
        run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC", now=now - timedelta(minutes=1),
        origin_channel={"integration_id": "desktop", "provider_key": "desktop", "channel_target": "alex", "alphonse_user_id": "alex"},
    )
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)
    assert daemon.run_once().status.value == "parked"
    assert not questions.route_answer(respondent_user_id="other", question_id=processor.question_id, text="A").handled
    answer = questions.route_answer(respondent_user_id="alex", question_id=processor.question_id, text="A")
    runtime.channel.queue_message(
        prompt="A", user="alex", correlation_id=answer.resumed_task.correlation_id,
        metadata={"task_state": answer.resumed_task.to_dict(), "answered_question_id": processor.question_id,
                  "routing_disposition": "correlated_response"},
    )
    assert daemon.run_once().status.value == "processed"
    execution = schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0]
    assert execution.status == "response_pending"
    assert runtime.outbox.get(execution.response_outbox_id).message == "Synthetic result"
    assert runtime.outbox.claim_next(OutboundSelector(integration_id="desktop"), lease_owner="desktop:fixture")
    assert daemon.acknowledge_desktop_delivery(client_id="fixture", outbox_message_id=execution.response_outbox_id)
    assert ScheduledTaskStore(path).list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].status == "delivered"
    assert SQLiteQuestionStore(path).load_task_checkpoint(answer.resumed_task.task_id).status == "completed"


def test_cancelled_schedule_closes_parked_question_on_reconciliation(tmp_path):
    from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host

    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path)
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)
    class Processor:
        def process(self, task, context):
            self.question = questions.create_question(task=task, question="Choose a fixture value")
            return ProcessingResult(StateSnapshot(metadata={"task_state": task.to_dict()}), ProcessingStatus.PARKED)
    processor = Processor()
    runtime = build_runtime_host(messages=queue, question_store=questions, schedule_store=schedules, processor=processor)
    runtime.core.memory = None
    daemon = V2Daemon(runtime, daemon_id="fixture-daemon")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    scheduled = schedules.create_task(owner_user_id="alex", name="Fixture", prompt="Fixture", schedule_kind="once",
                                      run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                                      now=now - timedelta(minutes=1))
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)
    assert daemon.run_once().status.value == "parked"
    schedules.cancel_task(scheduled.scheduled_task_id)
    daemon._last_unconfirmed_delivery_reconciliation = None
    daemon._reconcile_unconfirmed_scheduled_deliveries()
    assert questions.get_question(processor.question.question_id).status == "cancelled"
    assert not questions.route_answer(respondent_user_id="alex", question_id=processor.question.question_id, text="A").handled
    assert schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].status == "cancelled"


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


def test_waiting_for_provider_receipt_releases_processor_and_resumes(tmp_path):
    from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor, _await_scheduled_delivery
    from alphonse.agent_v2.core.io.channels import ChannelAddress
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host

    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path)
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)
    outbox = SQLiteOutboundStore(path)
    seen = []
    class Processor(HierarchicalCAPDProcessor):
        def _process(self, task, context):
            seen.append(task.goal)
            if task.metadata.get("source") == "scheduled_task":
                waiting = task.metadata.get("v3_scheduled_delivery_wait")
                if not waiting:
                    outbound = outbox.enqueue(
                        address=ChannelAddress(integration_id="fixture", provider_key="fixture", channel_target="target"),
                        message="Synthetic message", task_id=task.task_id, audience_user_id="recipient",
                        metadata={"occurrence_key": task.metadata["occurrence_key"], "scheduled_delivery_evidence": True},
                    )
                    self.outbox_id = outbound.outbox_message_id
                    waiting = {"outbox_message_id": self.outbox_id, "recipient_user_id": "recipient"}
                assert _await_scheduled_delivery(task, context, waiting)
            task.status = "completed"
            questions.save_task_checkpoint(task, status="done")
            return ProcessingResult(StateSnapshot(metadata={"task_state": task.to_dict()}), ProcessingStatus.COMPLETED)
    processor = Processor()
    runtime = build_runtime_host(messages=queue, question_store=questions, schedule_store=schedules, outbox=outbox, processor=processor)
    runtime.core.memory = None
    daemon = V2Daemon(runtime, daemon_id="fixture-daemon")
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    scheduled = schedules.create_task(owner_user_id="alex", name="Fixture", prompt="Scheduled goal", schedule_kind="once",
                                      run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC",
                                      now=now - timedelta(minutes=1))
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)
    runtime.channel.queue_message(prompt="Other goal", user="alex", message_id="other")

    assert daemon.run_once().status.value == "yielded"
    assert schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].status == "response_pending"
    assert daemon.run_once().status.value == "processed"
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET next_attempt_at=? WHERE status='retry_wait'", ("2000-01-01T00:00:00+00:00",))
    assert daemon.run_once().status.value == "yielded"
    assert len(outbox.list_for_occurrence(schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].occurrence_key)) == 1
    outbox.mark_delivered(processor.outbox_id, provider_message_id="fixture-provider")
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET next_attempt_at=? WHERE status='retry_wait'", ("2000-01-01T00:00:00+00:00",))
    assert daemon.run_once().status.value == "processed"
    assert seen == ["Scheduled goal", "Other goal", "Scheduled goal", "Scheduled goal"]
    assert schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].status == "delivered"
    assert len(outbox.list_for_occurrence(schedules.list_executions(scheduled_task_id=scheduled.scheduled_task_id)[0].occurrence_key)) == 1


def test_real_tactical_delivery_waits_and_reuses_the_sent_outbox(tmp_path):
    from alphonse.agent_v2.core.intelligence.v3 import MutationScope, SideEffectClass
    from alphonse.agent_v2.core.intelligence.v3.processor import _await_scheduled_delivery
    from alphonse.agent_v2.core.io import IntegrationIdentity, SQLiteCommunicationThreadStore, V2IdentityResolver
    from alphonse.agent_v2.core.io.communication import CommunicationRouter
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore, build_outbox_delivery_sink
    from alphonse.agent_v2.core.tools.registry.native.deliver_message import build_deliver_message_tool_definition
    from alphonse.agent_v2.users import V2UserStore

    path = tmp_path / "fixture.sqlite3"
    queue = SQLiteMessageQueue(path, lease_owner="fixture-daemon")
    questions = SQLiteQuestionStore(path)
    schedules = ScheduledTaskStore(path)
    outbox = SQLiteOutboundStore(path)
    users = V2UserStore(tmp_path / "users.sqlite3")
    owner = users.onboard(display_name="Fixture owner", users_root=tmp_path / "profiles")
    recipient = users.create_user(display_name="Fixture recipient")
    users.bind_address(user_id=recipient.user_id, integration_id="fixture", provider_key="telegram", provider_user_id="target", channel_target="target")
    resolver = V2IdentityResolver((IntegrationIdentity("fixture", "telegram"),), user_store=users)
    router = CommunicationRouter(users=users, resolver=resolver, outbox=outbox, threads=SQLiteCommunicationThreadStore(path))
    sink = build_outbox_delivery_sink(outbox=outbox, identity_resolver=resolver, communication_router=router)
    now = datetime(2026, 7, 10, 12, 30, tzinfo=timezone.utc)
    schedules.create_task(owner_user_id=owner.user_id, name="Fixture", prompt="Send a fixture message", schedule_kind="once",
                          run_at=(now - timedelta(seconds=1)).isoformat(), timezone_name="UTC", now=now - timedelta(minutes=1),
                          origin_channel={"integration_id": "desktop", "provider_key": "tui", "channel_target": owner.user_id, "alphonse_user_id": owner.user_id})
    ScheduledTaskWorker(store=schedules, messages=queue, worker_id="fixture-worker").run_once(now=now)
    queued = queue.dequeue()
    task = TaskState.from_queued_message(queued)
    task.task_id = queued.message_id
    registry = InMemoryToolRegistry()
    registry.register(build_deliver_message_tool_definition())
    phase = PhasePlan("phase", "Send", (PhaseSubgoal(
        "send", "Send the fixture", "delivery", allowed_capabilities=("native.deliver_message",),
        allowed_side_effects=(SideEffectClass.EXTERNAL_REVERSIBLE,),
    ),), authorized_capabilities=("native.deliver_message",), mutation_scope=MutationScope(allow_external_effects=True))
    state = new_tactical_state(phase)
    state.revealed_tool_ids = ["native.deliver_message"]
    context = CoreLoopContext(messages=queue, tools=registry, question_store=questions, schedule_store=schedules,
                              delivery_sink=sink, system_one=_accept_successful_actions(), scheduled_handoff_enabled=True)
    executor = PhaseExecutor(action_selector=_selector([{"tool_id": "native.deliver_message", "arguments": {
        "recipient": "Fixture recipient", "message": "Synthetic reminder",
    }}]))

    with pytest.raises(ScheduledHandoff):
        executor.run(task, state, context)
    assert len(outbox.list_for_occurrence(task.metadata["occurrence_key"])) == 1
    message_id = task.metadata["v3_scheduled_delivery_wait"]["outbox_message_id"]
    assert queue.release_scheduled_handoff(queued.message_id, lease_owner="fixture-daemon")
    outbox.mark_delivered(message_id, provider_message_id="fixture-provider")
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET next_attempt_at=? WHERE message_id=?", ("2000-01-01T00:00:00+00:00", queued.message_id))
    queue.dequeue()
    restored = questions.load_task_checkpoint(queue.handoff_task_id(queued.message_id, owner.user_id))
    assert _await_scheduled_delivery(restored, context, restored.metadata["v3_scheduled_delivery_wait"])
    outcome = PhaseExecutor(action_selector=lambda *_: pytest.fail("saved action should be reused")).run(
        restored, TacticalState.from_dict(restored.hierarchical_state), context,
    )
    assert outcome.status == PhaseStatus.PHASE_COMPLETE
    assert len(outbox.list_for_occurrence(task.metadata["occurrence_key"])) == 1


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
