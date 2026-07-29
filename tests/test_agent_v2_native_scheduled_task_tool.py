from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.intelligence.pdca.nodes import do_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node import _render_tool_call_plan_prompt
from alphonse.agent_v2.core.intelligence.pdca.nodes.plan_node import plan_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskRunner
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.scheduled_tasks import compute_next_run_at
from alphonse.agent_v2.core.tools.registry.native import SCHEDULED_TASK_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import SCHEDULED_TASK_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native import execute_scheduled_task
from alphonse.agent_v2.interfaces.tui import _latest_tool_result_response


def test_store_creates_one_off_task_and_lists_by_owner_and_project() -> None:
    store = ScheduledTaskStore()
    run_at = "2026-07-10T09:00:00-06:00"

    task = store.create_task(
        owner_user_id="alex",
        project_id="alpha",
        name="Morning reminder",
        description="Check status",
        prompt="Remind me to check status.",
        schedule_kind="once",
        run_at=run_at,
        timezone_name="America/Mexico_City",
    )

    assert task.status == "active"
    assert task.next_run_at == "2026-07-10T15:00:00+00:00"
    assert store.get_task(task.scheduled_task_id) == task
    assert store.list_tasks(owner_user_id="alex", project_id="alpha") == [task]
    assert store.list_tasks(owner_user_id="gaby") == []


def test_store_creates_rrule_task_and_computes_next_run() -> None:
    store = ScheduledTaskStore()
    now = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)

    task = store.create_task(
        owner_user_id="alex",
        name="Daily report",
        prompt="Prepare the daily report.",
        schedule_kind="rrule",
        rrule="FREQ=DAILY;BYHOUR=9;BYMINUTE=0",
        dtstart="2026-07-09T09:00:00-06:00",
        timezone_name="America/Mexico_City",
        now=now,
    )

    assert task.schedule["kind"] == "rrule"
    assert task.next_run_at == "2026-07-09T15:00:00+00:00"


def test_store_pause_resume_cancel_complete_and_execution_history() -> None:
    store = ScheduledTaskStore()
    task = store.create_task(
        owner_user_id="alex",
        name="One",
        prompt="Do it.",
        schedule_kind="once",
        run_at="2026-07-10T09:00:00+00:00",
    )

    paused = store.pause_task(task.scheduled_task_id)
    resumed = store.resume_task(task.scheduled_task_id, now=datetime(2026, 7, 9, tzinfo=timezone.utc))
    execution = store.record_execution(
        scheduled_task_id=task.scheduled_task_id,
        run_id="run-1",
        status="queued",
        queued_message_id="message-1",
    )
    completed = store.complete_task(task.scheduled_task_id, last_run_at="2026-07-10T09:00:00+00:00")
    cancelled = store.cancel_task(task.scheduled_task_id)

    assert paused.status == "paused"
    assert paused.next_run_at is None
    assert resumed.status == "active"
    assert resumed.next_run_at == "2026-07-10T09:00:00+00:00"
    assert execution.queued_message_id == "message-1"
    assert store.list_executions(scheduled_task_id=task.scheduled_task_id)[0].run_id == "run-1"
    assert completed.status == "completed"
    assert cancelled.status == "cancelled"


def test_scheduled_execution_project_migration_backfills_parent_idempotently(tmp_path) -> None:
    db_path = tmp_path / "schedules.sqlite3"
    store = ScheduledTaskStore(db_path)
    task = store.create_task(
        owner_user_id="alex",
        project_id="alpha",
        name="Migrate",
        prompt="Run",
        schedule_kind="once",
        run_at="2026-07-10T09:00:00+00:00",
    )
    store.record_execution(scheduled_task_id=task.scheduled_task_id, run_id="run-migrate", status="queued")
    with sqlite3.connect(db_path) as conn:
        conn.execute("DROP INDEX idx_v2_scheduled_task_executions_project")
        conn.execute("ALTER TABLE v2_scheduled_task_executions DROP COLUMN project_id")

    migrated = ScheduledTaskStore(db_path)
    restarted = ScheduledTaskStore(db_path)

    assert migrated.list_executions(scheduled_task_id=task.scheduled_task_id)[0].project_id == "alpha"
    assert restarted.list_executions(scheduled_task_id=task.scheduled_task_id)[0].project_id == "alpha"


def test_store_updates_only_editable_tasks_and_permanently_deletes_execution_history() -> None:
    store = ScheduledTaskStore()
    task = store.create_task(
        owner_user_id="alex",
        name="Original",
        prompt="Original prompt.",
        schedule_kind="once",
        run_at="2026-07-10T09:00:00+00:00",
    )
    store.record_execution(scheduled_task_id=task.scheduled_task_id, run_id="run-1", status="queued")

    updated = store.update_task(task.scheduled_task_id, name="Updated", prompt="Updated prompt.")

    assert updated.name == "Updated"
    assert updated.prompt == "Updated prompt."
    assert updated.schedule == task.schedule
    with pytest.raises(ValueError, match="scheduled_task_not_active"):
        store.pause_task(store.pause_task(task.scheduled_task_id).scheduled_task_id)
    store.cancel_task(task.scheduled_task_id)
    with pytest.raises(ValueError, match="scheduled_task_not_editable"):
        store.update_task(task.scheduled_task_id, name="No", prompt="No.")

    assert store.delete_task(task.scheduled_task_id) is True
    assert store.get_task(task.scheduled_task_id) is None
    assert store.list_executions(scheduled_task_id=task.scheduled_task_id) == []


def test_native_scheduled_task_tool_registers_and_uses_context_owner_project() -> None:
    store = ScheduledTaskStore()
    task = TaskState(user="alex", project_id="alpha")
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), schedule_store=store)

    result = execute_scheduled_task(
        {
            "name": "Check oven",
            "prompt": "Remind Alex to check the oven.",
            "schedule_kind": "once",
            "run_at": "2026-07-10T09:00:00+00:00",
            "timezone": "UTC",
        },
        context=context,
    )
    stored = store.get_task(result["scheduled_task_id"])
    descriptor = build_native_tool_registry().get(SCHEDULED_TASK_TOOL_NAME)

    assert descriptor is not None
    assert descriptor.tool_id == SCHEDULED_TASK_TOOL_ID
    assert result["next_run_at"] == "2026-07-10T09:00:00+00:00"
    assert result["project_id"] == "alpha"
    assert stored is not None
    assert stored.owner_user_id == "alex"
    assert stored.project_id == "alpha"


def test_native_scheduled_task_uses_the_configured_user_timezone_by_default() -> None:
    store = ScheduledTaskStore()
    context = ToolExecutionContext(
        task=TaskState(user="alex"),
        messages=InMemoryMessageQueue(),
        schedule_store=store,
        user_timezone_provider=lambda _user_id: "America/Mexico_City",
    )

    result = execute_scheduled_task(
        {"name": "Evening reminder", "prompt": "Reminder", "schedule_kind": "once", "run_at": "2026-07-10T20:30:00"},
        context=context,
    )

    stored = store.get_task(result["scheduled_task_id"])
    assert stored is not None
    assert stored.timezone == "America/Mexico_City"
    assert stored.next_run_at == "2026-07-11T02:30:00+00:00"


def test_native_scheduled_task_tool_rejects_missing_required_fields() -> None:
    context = ToolExecutionContext(
        task=TaskState(user="alex"),
        messages=InMemoryMessageQueue(),
        schedule_store=ScheduledTaskStore(),
    )

    with pytest.raises(ValueError, match="scheduled_task_prompt_required"):
        execute_scheduled_task(
            {"name": "Missing", "prompt": "", "schedule_kind": "once", "run_at": "2026-07-10T09:00:00+00:00"},
            context=context,
        )


def test_do_node_records_scheduled_task_result_in_execution_result_not_metadata() -> None:
    store = ScheduledTaskStore()
    task = TaskState(user="alex", project_id="alpha")
    task.append_plan_call(
        {
            "id": "call-1",
            "tool_id": SCHEDULED_TASK_TOOL_ID,
            "tool_name": SCHEDULED_TASK_TOOL_NAME,
            "arguments": {
                "name": "Check oven",
                "prompt": "Remind Alex to check the oven.",
                "schedule_kind": "once",
                "run_at": "2026-07-10T09:00:00+00:00",
            },
            "internal_state": "Scheduling the reminder.",
        }
    )

    do_node(
        task,
        context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=build_native_tool_registry(), schedule_store=store),
    )

    execution = json.loads(task.plan_json)[0]["execution"]
    assert execution["status"] == "success"
    assert execution["result"]["scheduled_task_id"]
    assert "scheduled_task_id" not in task.metadata


def test_runner_queues_due_one_off_task_and_marks_completed() -> None:
    store = ScheduledTaskStore()
    queue = InMemoryMessageQueue()
    task = store.create_task(
        owner_user_id="alex",
        project_id="alpha",
        name="Due",
        prompt="Do the due thing.",
        schedule_kind="once",
        run_at="2026-07-09T10:00:00+00:00",
    )

    outcomes = ScheduledTaskRunner(store=store, messages=queue).run_due_once(
        now=datetime(2026, 7, 9, 10, 1, tzinfo=timezone.utc)
    )
    queued = queue.peek()
    updated = store.get_task(task.scheduled_task_id)

    assert len(outcomes) == 1
    assert outcomes[0]["status"] == "queued"
    assert queued is not None
    assert queued.message.prompt == "Do the due thing."
    assert queued.message.user == "alex"
    assert queued.message.project_id == "alpha"
    assert queued.message.metadata["source"] == "scheduled_task"
    assert queued.message.metadata["scheduled_task_id"] == task.scheduled_task_id
    assert queued.message.metadata["project_id"] == "alpha"
    assert store.list_executions(scheduled_task_id=task.scheduled_task_id)[0].project_id == "alpha"
    assert updated is not None
    assert updated.status == "completed"
    assert updated.next_run_at is None


def test_runner_queues_due_rrule_task_and_advances_next_run() -> None:
    store = ScheduledTaskStore()
    queue = InMemoryMessageQueue()
    task = store.create_task(
        owner_user_id="alex",
        name="Daily",
        prompt="Run daily.",
        schedule_kind="rrule",
        rrule="FREQ=DAILY;BYHOUR=10;BYMINUTE=0",
        dtstart="2026-07-09T10:00:00+00:00",
        timezone_name="UTC",
        now=datetime(2026, 7, 9, 9, 0, tzinfo=timezone.utc),
    )

    outcomes = ScheduledTaskRunner(store=store, messages=queue).run_due_once(
        now=datetime(2026, 7, 9, 10, 0, tzinfo=timezone.utc)
    )
    updated = store.get_task(task.scheduled_task_id)

    assert len(outcomes) == 1
    assert queue.size() == 1
    assert updated is not None
    assert updated.status == "active"
    assert updated.next_run_at == "2026-07-10T10:00:00+00:00"


def test_runner_ignores_not_due_and_inactive_tasks() -> None:
    store = ScheduledTaskStore()
    queue = InMemoryMessageQueue()
    future = store.create_task(
        owner_user_id="alex",
        name="Future",
        prompt="Later.",
        schedule_kind="once",
        run_at="2026-07-10T10:00:00+00:00",
    )
    paused = store.create_task(
        owner_user_id="alex",
        name="Paused",
        prompt="No.",
        schedule_kind="once",
        run_at="2026-07-09T10:00:00+00:00",
        enabled=False,
    )
    store.cancel_task(paused.scheduled_task_id)

    outcomes = ScheduledTaskRunner(store=store, messages=queue).run_due_once(
        now=datetime(2026, 7, 9, 10, 1, tzinfo=timezone.utc)
    )

    assert outcomes == []
    assert queue.size() == 0
    assert store.get_task(future.scheduled_task_id).status == "active"  # type: ignore[union-attr]


def test_tool_planning_prompt_exposes_scheduled_task_tool() -> None:
    descriptor = build_native_tool_registry().get(SCHEDULED_TASK_TOOL_NAME)
    assert descriptor is not None

    prompt = _render_tool_call_plan_prompt(
        TaskState(goal="Remind me tomorrow", acceptance_criteria_md="1.- [ ] Reminder scheduled"),
        (descriptor,),
    )

    assert SCHEDULED_TASK_TOOL_ID in prompt
    assert "future activity" in prompt
    assert "Do not pass natural-language dates" in prompt


def test_stub_inference_can_select_scheduled_task_tool() -> None:
    router = InferenceRouter(
        provider=StubInferenceProvider(
            markdown_by_purpose={
                InferencePurpose.ACCEPTANCE_CRITERIA: "1.- [ ] Reminder scheduled",
                InferencePurpose.CRITERIA_REVIEW: "1.- [x] Reminder scheduled",
            },
            tool_call={
                "tool_id": SCHEDULED_TASK_TOOL_ID,
                "tool_name": SCHEDULED_TASK_TOOL_NAME,
                "arguments": {
                    "name": "Check oven",
                    "prompt": "Remind Alex to check the oven.",
                    "schedule_kind": "once",
                    "run_at": "2026-07-10T09:00:00+00:00",
                },
                "internal_state": "Scheduling the reminder.",
            },
        ),
        default_profile=ModelProfile(provider="stub", model="stub", profile_id="stub"),
    )
    store = ScheduledTaskStore()
    task = TaskState(user="alex", goal="Remind me", acceptance_criteria_md="1.- [ ] Reminder scheduled")

    plan_node(
        task,
        context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=build_native_tool_registry(), inference=router),
    )
    do_node(
        task,
        context=CoreLoopContext(messages=InMemoryMessageQueue(), tools=build_native_tool_registry(), schedule_store=store),
    )

    execution = json.loads(task.plan_json)[0]["execution"]
    assert execution["status"] == "success"
    assert store.list_tasks(owner_user_id="alex")


def test_tui_renders_scheduled_task_result() -> None:
    task = TaskState()
    task.append_plan_call(
        {
            "id": "call-1",
            "tool_id": SCHEDULED_TASK_TOOL_ID,
            "tool_name": SCHEDULED_TASK_TOOL_NAME,
            "arguments": {},
            "internal_state": "Scheduling.",
        }
    )
    task.record_plan_call_success(
        "call-1",
        {
            "scheduled_task_id": "scheduled_task_1",
            "name": "Check oven",
            "next_run_at": "2026-07-10T09:00:00+00:00",
        },
    )

    assert _latest_tool_result_response(task.to_dict()) == 'Scheduled "Check oven" for 2026-07-10T09:00:00+00:00.'


def test_compute_next_run_at_returns_none_for_exhausted_rrule() -> None:
    assert (
        compute_next_run_at(
            schedule={
                "kind": "rrule",
                "rrule": "FREQ=DAILY;COUNT=1",
                "dtstart": "2026-07-09T10:00:00+00:00",
            },
            timezone_name="UTC",
            after=datetime(2026, 7, 10, tzinfo=timezone.utc),
        )
        is None
    )
