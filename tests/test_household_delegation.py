import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.evaluation.household_delegation import HouseholdFixture, score, run_case
from alphonse.agent_v2.system_one import SystemOneUnavailableError


CORPUS = json.loads((Path(__file__).parent / "fixtures/household_delegation/v1.json").read_text())


def make_fixture(tmp_path, case_id):
    case = next(item for item in CORPUS["cases"] if item["id"] == case_id)
    return HouseholdFixture(tmp_path, case, CORPUS, SimpleNamespace(calls=[]), None)


def test_fixture_blocks_shell_and_never_loads_household_state(tmp_path, monkeypatch):
    monkeypatch.setenv("ALPHONSE_V2_DB_PATH", "/nonexistent/production-must-not-open.sqlite")
    fixture = make_fixture(tmp_path, "sensor-recovery")
    invoke = ToolInvocationService(context=fixture.context, task=fixture.task)
    for command in ("touch /tmp/household-escape", "household-sensor --help; env", "cat ~/.alphonse/state.sqlite"):
        with pytest.raises(PermissionError, match="allowlist"):
            invoke.execute_or_raise("native.bash", {"command": command})
    assert fixture.skills.list_skills() == []
    assert fixture.schedules.list_tasks() == []


def test_oracle_rejects_plausible_answer_without_historical_evidence(tmp_path):
    fixture = make_fixture(tmp_path, "anniversary-recall")
    fixture.task.status = "completed"
    fixture.task.metadata["prepared_user_response"] = {"message": "Casa del Lago in Chapultepec."}
    result = score(fixture.case, fixture, fixture.task, error=None)
    assert result["checks"]["answer_fragments_present"]
    assert not result["checks"]["necessary_history_delivered"]
    assert not result["mechanical_pass"]


def test_real_native_schedule_for_requester_does_not_satisfy_mother_delegation(tmp_path):
    fixture = make_fixture(tmp_path, "mother-medication-reminder")
    invoke = ToolInvocationService(context=fixture.context, task=fixture.task)
    result = invoke.execute_or_raise("native.scheduled_task", {
        "name": "Elena medication", "prompt": "Medicine A", "description": "Medicine A",
        "schedule_kind": "rrule", "rrule": "FREQ=DAILY;BYHOUR=8;BYMINUTE=0;BYSECOND=0",
        "dtstart": "2026-10-19T08:00:00-06:00", "timezone": "America/Mexico_City", "delivery_mode": "direct",
    })
    invoke.execute_or_raise("native.scheduled_task_delivery", {
        "operation": "add_delivery_channel", "provider_key": "telegram", "scheduled_task_id": result["scheduled_task_id"],
    })
    fixture.task.status = "completed"
    fixture.task.metadata["prepared_user_response"] = {"message": "I scheduled Elena's reminder."}
    scored = score(fixture.case, fixture, fixture.task, error=None)
    assert not scored["checks"]["correct_recipient_schedule"]
    assert not scored["mechanical_pass"]
    assert fixture.schedules.list_tasks()[0].origin_channel["channel_target"] == "fixture-admin-chat"


def schedule_for_mother(fixture):
    return ToolInvocationService(context=fixture.context, task=fixture.task).execute_or_raise("native.scheduled_task", {
        "name": "Elena medication", "prompt": "Medicine A", "description": "Medicine A",
        "schedule_kind": "rrule", "rrule": "FREQ=DAILY;BYHOUR=8;BYMINUTE=0;BYSECOND=0",
        "dtstart": "2026-10-19T08:00:00-06:00", "timezone": "America/Mexico_City",
        "recipient_user_id": "fixture-mother", "recipient_provider_key": "telegram",
    })


def test_recipient_reminder_is_delivered_once_to_mother_and_records_her_history(tmp_path):
    from datetime import datetime
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from alphonse.agent_v2.daemon import V2Daemon
    fixture = make_fixture(tmp_path, "mother-medication-reminder")
    first = schedule_for_mother(fixture)
    second = schedule_for_mother(fixture)
    assert first["scheduled_task_id"] == second["scheduled_task_id"]
    scheduled = fixture.schedules.get_task(first["scheduled_task_id"])
    assert scheduled.owner_user_id == "fixture-admin"
    assert scheduled.delivery_mode == "direct"
    assert scheduled.origin_channel["channel_target"] == "fixture-admin-chat"
    occurrence = fixture.schedules.claim_due_occurrences(worker_id="fixture-worker", now=datetime.fromisoformat(scheduled.next_run_at))[0]
    outbox = SQLiteOutboundStore()
    daemon = SimpleNamespace(runtime=SimpleNamespace(identity_resolver=fixture.context.identity_resolver,
        outbox=outbox, conversation_store=fixture.store))
    one = V2Daemon._deliver_scheduled_reminder(daemon, occurrence)
    two = V2Daemon._deliver_scheduled_reminder(daemon, occurrence)
    assert one == two
    assert len(outbox.list()) == 1
    message = outbox.list()[0]
    assert message.channel_target == "fixture-mother-chat"
    assert message.audience_user_id == "fixture-mother"
    assert message.message == "Reminder: Medicine A"
    assert fixture.store.list_recent(owner_user_id="fixture-mother")[-1].content == message.message


def test_recipient_becoming_inactive_never_falls_back_to_requester(tmp_path):
    from datetime import datetime
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from alphonse.agent_v2.daemon import V2Daemon
    fixture = make_fixture(tmp_path, "mother-medication-reminder")
    result = schedule_for_mother(fixture)
    scheduled = fixture.schedules.get_task(result["scheduled_task_id"])
    fixture.users.update_user("fixture-mother", is_active=False)
    occurrence = fixture.schedules.claim_due_occurrences(worker_id="fixture-worker", now=datetime.fromisoformat(scheduled.next_run_at))[0]
    outbox = SQLiteOutboundStore()
    daemon = SimpleNamespace(runtime=SimpleNamespace(identity_resolver=fixture.context.identity_resolver,
        outbox=outbox, conversation_store=fixture.store))
    with pytest.raises(RuntimeError, match="recipient_unresolved"):
        V2Daemon._deliver_scheduled_reminder(daemon, occurrence)
    assert outbox.list() == []


def test_unknown_recipient_creates_no_reminder(tmp_path):
    fixture = make_fixture(tmp_path, "mother-medication-reminder")
    with pytest.raises(ValueError, match="recipient_unresolved"):
        ToolInvocationService(context=fixture.context, task=fixture.task).execute_or_raise("native.scheduled_task", {
            "name": "Reminder", "prompt": "Medicine A", "schedule_kind": "once", "run_at": "2026-10-19T08:00:00-06:00",
            "recipient_user_id": "unknown-member",
        })
    assert fixture.schedules.list_tasks() == []


def test_native_skill_install_reads_back_the_approved_definition_and_is_idempotent(tmp_path):
    fixture = make_fixture(tmp_path, "sensor-learned-route")
    invoke = ToolInvocationService(context=fixture.context, task=fixture.task)
    arguments = {"name": "study-sensor", "description": "Read the study sensor using a known working route.",
                 "instructions": "Read fresh data with household-sensor read --route backup."}
    first = invoke.execute_or_raise("native.skill_install", arguments)
    repeated = invoke.execute_or_raise("native.skill_install", arguments)
    assert first["created"] and not repeated["created"]
    assert repeated["instructions"] == arguments["instructions"]
    assert repeated["verification"]["status"] == "verified"
    assert len(fixture.skills.list_skills()) == 1
    with pytest.raises(ValueError, match="existing_definition_mismatch"):
        invoke.execute_or_raise("native.skill_install", {**arguments, "instructions": "Different instructions"})
    assert fixture.skills.list_skills()[0].instructions == arguments["instructions"]


def test_authenticated_approval_becomes_typed_evidence_in_the_resumed_v3_phase(tmp_path, monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3 import PhasePlan, PhaseSubgoal, new_tactical_state
    from alphonse.agent_v2.core.intelligence.v3 import processor
    fixture = make_fixture(tmp_path, "sensor-learned-route")
    phase = PhasePlan("approval", "Obtain approval", (PhaseSubgoal("approve", "Obtain approval", "approval"),))
    fixture.task.hierarchical_state = new_tactical_state(phase).to_dict()
    interrupt = ToolInvocationService(context=fixture.context, task=fixture.task).execute_or_raise("native.ask_question", {
        "question": "Approve this complete fixture skill draft?", "question_kind": "yes_no"})["question_interrupt"]
    answer = fixture.context.question_store.route_answer(respondent_user_id="fixture-admin",
        question_id=interrupt["question_id"], payload={"answer": True})
    assert answer.resumed_task is not None
    receipt = answer.resumed_task.metadata["v3_resumption_evidence"][0]
    assert receipt["kind"] == "authenticated_question_answer"
    assert receipt["result"]["answered_by"] == "fixture-admin"
    assert receipt["result"]["answer"] == {"answer": True}
    next_phase = PhasePlan("install", "Install the approved skill", (PhaseSubgoal("install", "Install skill", "skill"),))
    monkeypatch.setattr(processor, "plan_phase", lambda *args: next_phase)
    resumed = processor.HierarchicalCAPDProcessor._new_state(answer.resumed_task, fixture.context)
    imported = next(item for item in resumed.evidence.entries if item["evidence_ref"] == receipt["evidence_ref"])
    assert imported["status"] == "success"
    assert imported["result"] == receipt["result"]
    assert imported["source"] == "prior_task_evidence"
    from alphonse.agent_v2.core.intelligence.v3.outer import _invariant_violations
    assert _invariant_violations(answer.resumed_task, resumed) == []
    duplicate = fixture.context.question_store.route_answer(respondent_user_id="fixture-admin",
        question_id=interrupt["question_id"], payload={"answer": True})
    assert not duplicate.handled


def test_recipient_delivery_failure_notifies_the_requester(tmp_path):
    from datetime import datetime
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.services.scheduled_worker import ScheduledTaskWorker
    fixture = make_fixture(tmp_path, "mother-medication-reminder")
    result = schedule_for_mother(fixture)
    scheduled = fixture.schedules.get_task(result["scheduled_task_id"])
    fixture.users.update_user("fixture-mother", is_active=False)
    outbox = SQLiteOutboundStore()
    daemon = SimpleNamespace(runtime=SimpleNamespace(identity_resolver=fixture.context.identity_resolver,
        outbox=outbox, conversation_store=fixture.store, schedule_store=fixture.schedules))
    worker = ScheduledTaskWorker(store=fixture.schedules, messages=fixture.context.messages, max_attempts=1,
        on_direct_delivery=lambda occurrence: V2Daemon._deliver_scheduled_reminder(daemon, occurrence),
        on_failure=lambda occurrence, error: V2Daemon._notify_scheduled_task_failure(daemon, {
            "scheduled_task_id": occurrence.task.scheduled_task_id, "channel": occurrence.task.origin_channel}, error=error))
    worker.run_once(now=datetime.fromisoformat(scheduled.next_run_at))
    messages = outbox.list()
    assert len(messages) == 1
    assert messages[0].kind == "scheduled_task_failed"
    assert messages[0].channel_target == "fixture-admin-chat"


def fixture_daemon(fixture):
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host
    return V2Daemon(build_runtime_host(user_store=fixture.users, schedule_store=fixture.schedules,
        conversation_store=fixture.store, identity_resolver=fixture.context.identity_resolver,
        skill_store=fixture.skills, tools=fixture.context.tools, question_store=fixture.context.question_store))


def test_actual_daemon_scheduler_wires_terminal_failure_notification(tmp_path):
    from datetime import datetime
    fixture = make_fixture(tmp_path, "mother-medication-reminder")
    result = schedule_for_mother(fixture)
    scheduled = fixture.schedules.get_task(result["scheduled_task_id"])
    daemon = fixture_daemon(fixture)
    assert daemon.scheduler.on_failure is not None
    daemon.scheduler.max_attempts = 1
    fixture.users.update_user("fixture-mother", is_active=False)
    daemon.scheduler.run_once(now=datetime.fromisoformat(scheduled.next_run_at))
    notifications = [item for item in daemon.runtime.outbox.list() if item.kind == "scheduled_task_failed"]
    assert len(notifications) == 1
    assert notifications[0].channel_target == "fixture-admin-chat"


def test_terminal_telegram_failure_notifies_requester_once(tmp_path):
    from datetime import datetime
    fixture = make_fixture(tmp_path, "mother-medication-reminder")
    result = schedule_for_mother(fixture)
    scheduled = fixture.schedules.get_task(result["scheduled_task_id"])
    daemon = fixture_daemon(fixture)
    daemon.scheduler.run_once(now=datetime.fromisoformat(scheduled.next_run_at))
    reminder = next(item for item in daemon.runtime.outbox.list() if item.kind == "scheduled_reminder")
    daemon._on_outbox_failed(reminder, "fixture_telegram_unavailable")
    daemon._on_outbox_failed(reminder, "fixture_telegram_unavailable")
    notifications = [item for item in daemon.runtime.outbox.list() if item.kind == "scheduled_task_failed"]
    assert len(notifications) == 1
    assert notifications[0].channel_target == "fixture-admin-chat"
    assert notifications[0].audience_user_id == "fixture-admin"


def test_inference_failure_preserves_completed_schedule_and_failed_checkpoint(tmp_path):
    from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
    from alphonse.agent_v2.evaluation.deterministic_adapters import _ReplayJev
    fixture = make_fixture(tmp_path, "mother-medication-reminder")
    result = schedule_for_mother(fixture)

    class FailedInference:
        def generate_json(self, request):
            raise RuntimeError("PRIVATE_PROVIDER_DIAGNOSTIC")

        def generate_markdown(self, request):
            raise RuntimeError("PRIVATE_PROVIDER_DIAGNOSTIC")

    fixture.task.metadata["v3_admission"] = {"completed": True, "route": "task"}
    fixture.context.inference = FailedInference()
    fixture.context.system_one = _ReplayJev()
    outcome = HierarchicalCAPDProcessor().process(fixture.task, fixture.context)
    assert outcome.status.value == "failed"
    assert fixture.task.status == "failed"
    assert fixture.context.question_store.load_task_checkpoint(fixture.task.task_id).status == "failed"
    assert len(fixture.schedules.list_tasks()) == 1
    assert fixture.schedules.get_task(result["scheduled_task_id"]).delivery_mode == "direct"
    message = fixture.task.metadata["v3_failure_response"]["message"]
    assert "not been fully verified" in message
    assert "PRIVATE_PROVIDER_DIAGNOSTIC" not in message + str(outcome.error)


def test_success_label_and_temperature_text_do_not_substitute_for_fresh_read(tmp_path):
    fixture = make_fixture(tmp_path, "sensor-recovery")
    fixture.task.status = "completed"
    fixture.task.metadata["prepared_user_response"] = {"message": "The temperature is 24 C."}
    result = score(fixture.case, fixture, fixture.task, error=None)
    assert not result["checks"]["fresh_read_obtained"]
    assert not result["mechanical_pass"]


@pytest.mark.parametrize("message", [
    "Both sensor routes are unavailable. I could not get a fresh reading.",
    "I couldn’t get a fresh study temperature. Both routes failed to return a measurement.",
])
def test_unavailable_routes_cannot_be_scored_as_recovered(tmp_path, message):
    fixture = make_fixture(tmp_path, "sensor-visible-failure")
    for route in ("primary", "backup"):
        assert fixture.bash({"command": "household-sensor read --route " + route})["exit_code"] == 1
    fixture.task.metadata["v3_failure_response"] = {"message": message}
    result = score(fixture.case, fixture, fixture.task, error=None)
    assert result["checks"]["unresolved_explanation_present"]
    assert result["checks"]["alternative_attempted"]
    assert result["human_review"] == "pending"


def test_runtime_service_error_is_retained_without_fabricating_an_answer(monkeypatch):
    class Unavailable:
        def classify_task_admission(self, **kwargs):
            raise SystemOneUnavailableError("fixture_offline")

    monkeypatch.setenv("ALPHONSE_V2_DB_PATH", "/nonexistent/production-must-not-open.sqlite")
    case = CORPUS["cases"][0]
    report = run_case(case, CORPUS, SimpleNamespace(calls=[]), Unavailable(), candidate=False)
    assert report["runtime_error"] is None
    assert report["task_status"] == "failed"
    assert report["score"]["answer"]
    assert not report["score"]["mechanical_pass"]
