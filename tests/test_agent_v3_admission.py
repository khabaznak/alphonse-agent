from __future__ import annotations

import pytest
import json

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferenceResult
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.intelligence.v3.context import prepare_context_curation
from alphonse.agent_v2.core.io import SQLiteOutboundStore, build_outbox_delivery_sink, channel_metadata
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.agent_config import AgentConfigPromptLoader, AgentConfigStore, GLOBAL_CONTEXT_FILE
from alphonse.agent_v2.system_one import SystemOneAdmissionDecision
from alphonse.agent_v2.system_one import SystemOneUnavailableError


def _new_task(prompt: str = "Please inspect the project") -> TaskState:
    return TaskState(
        task_id="message-1",
        message_id="message-1",
        goal=prompt,
        user="alex",
        project_id="home",
        metadata={
            "routing_disposition": "pdca_task",
            "channel": channel_metadata(
                integration_id="desktop",
                provider_key="tui",
                channel_target="alex",
                alphonse_user_id="alex",
                provider_message_id="inbound-1",
            ),
        },
    )


class _AdmissionJev:
    def __init__(self, decision: SystemOneAdmissionDecision) -> None:
        self.decision = decision
        self.calls = 0
        self.context_candidates = ()
        self.recent_conversation = ""
        self.acknowledgement_safe = True
        self.acknowledgements = []

    def review_task_acknowledgement(self, *, message, acknowledgement):
        self.acknowledgements.append((message, acknowledgement))
        return self.acknowledgement_safe

    def classify_task_admission(self, *, message: str, recent_conversation: str = "", context_candidates=()) -> SystemOneAdmissionDecision:
        assert message
        self.calls += 1
        self.context_candidates = tuple(context_candidates)
        self.recent_conversation = recent_conversation
        return self.decision


class _DirectInference:
    def __init__(self, response: str) -> None:
        self.response = response
        self.requests = []

    def generate_markdown(self, request):
        self.requests.append(request)
        return InferenceResult(content=self.response, model_profile=request.model_profile)


def test_new_text_only_message_responds_without_entering_plan() -> None:
    task = _new_task("Hello Alphonse")
    jev = _AdmissionJev(SystemOneAdmissionDecision(False, 0.04, True, model="test-jev"))
    inference = _DirectInference("Hello Alex! How can I help?")

    result = HierarchicalCAPDProcessor().process(
        task,
        CoreLoopContext(messages=InMemoryMessageQueue(), inference=inference, system_one=jev),
    )

    assert result.status.value == "completed"
    assert task.metadata["v3_admission"]["route"] == "direct_response"
    assert task.metadata["prepared_user_response"]["message"] == "Hello Alex! How can I help?"
    assert task.metadata.get("v3_phase_history", []) == []
    assert len(inference.requests) == 1


def test_direct_reply_receives_cross_project_history_global_and_selected_project_context(tmp_path) -> None:
    task = _new_task("And what should I do next?")
    conversations = SQLiteConversationStore()
    conversations.record(
        owner_user_id="alex", project_id="another-project", role="user",
        content="A clinician recommended a week of rest.", source="test", source_message_id="prior-1",
    )
    config = AgentConfigStore(tmp_path / "agent-config")
    config.save(GLOBAL_CONTEXT_FILE, "Household language preference: Spanish.")
    project = type("Project", (), {"config_path": str(tmp_path / "project_config.json"), "project_id": "home", "name": "Recovery"})()
    (tmp_path / "project_config.json").write_text(json.dumps({"schema_version": 1, "scope": "Goal: return to activity gradually."}), encoding="utf-8")
    projects = type("Projects", (), {
        "get_project": lambda self, project_id, **kwargs: project if project_id == "home" else None,
        "list_visible_projects": lambda self, user_id: [project],
    })()
    inference = _DirectInference("Descansa y sigue las indicaciones recibidas.")
    context = CoreLoopContext(
        messages=InMemoryMessageQueue(), inference=inference,
        system_one=_AdmissionJev(SystemOneAdmissionDecision(False, 0.04, True, model="test-jev", selected_context_ids=("project:home",))),
        prompts=AgentConfigPromptLoader.from_store(config), project_store=projects,
        conversation_store=conversations,
    )

    HierarchicalCAPDProcessor().process(task, context)

    prompt = inference.requests[0].prompt
    assert "A clinician recommended a week of rest." in prompt
    assert "Household language preference: Spanish." in prompt
    assert "Goal: return to activity gradually." in prompt


def test_context_curation_separates_invariants_from_optional_candidates(tmp_path) -> None:
    config = AgentConfigStore(tmp_path / "agent-config")
    config.save(GLOBAL_CONTEXT_FILE, """# Global Context
## Family definition
- Member A is a parent of Member B.
## Household location and setup
- Household is in a generic town.
## Household norms
- The household prefers shared meals.
## Privacy and sharing boundaries
- Follow member privacy settings.
## Please remove
- REMOVAL_SENTINEL
""")
    selected_root = tmp_path / "selected"
    optional_root = tmp_path / "optional"
    selected_root.mkdir()
    optional_root.mkdir()
    (selected_root / "project_config.json").write_text(json.dumps({"schema_version": 1, "scope": "Selected goal context"}), encoding="utf-8")
    (optional_root / "project_config.json").write_text(json.dumps({"schema_version": 1, "scope": "Optional project context"}), encoding="utf-8")
    selected_project = type("Project", (), {
        "project_id": "selected", "name": "Selected", "description": "Current effort",
        "config_path": str(selected_root / "project_config.json"),
    })()
    optional_project = type("Project", (), {
        "project_id": "optional", "name": "Optional", "description": "Another active effort",
        "config_path": str(optional_root / "project_config.json"),
    })()
    project_store = type("Projects", (), {
        "get_project": lambda self, project_id, **kwargs: selected_project if project_id == "selected" else None,
        "list_visible_projects": lambda self, user_id: [selected_project, optional_project],
    })()
    task = _new_task("A request")
    task.project_id = "selected"

    invariant, candidates = prepare_context_curation(
        task,
        CoreLoopContext(
            messages=InMemoryMessageQueue(), prompts=AgentConfigPromptLoader.from_store(config),
            project_store=project_store,
        ),
    )

    assert "Member A is a parent" in invariant
    assert "Selected goal context" not in invariant
    assert "Selected goal context" in next(item["content"] for item in candidates if item["id"] == "project:selected")
    assert "The household prefers shared meals" not in invariant
    assert "Optional project context" not in invariant
    assert "REMOVAL_SENTINEL" not in invariant
    assert {item["id"] for item in candidates} == {"global:household-norms", "project:selected", "project:optional"}


def test_direct_reply_uses_only_the_context_candidates_selected_at_admission(tmp_path) -> None:
    task = _new_task("What should we cook this week?")
    config = AgentConfigStore(tmp_path / "agent-config")
    config.save(GLOBAL_CONTEXT_FILE, """# Global Context
## Family definition
- Member A and Member B are household members.
## Household norms
- The household prefers shared meals.
## Context maintenance
- Keep context files current.
## Please remove
- REMOVAL_SENTINEL
""")
    jev = _AdmissionJev(SystemOneAdmissionDecision(
        False, 0.03, True, model="test-jev",
        selected_context_ids=("global:household-norms",),
        context_probabilities={"global:household-norms": 0.94, "global:context-maintenance": 0.11},
    ))
    inference = _DirectInference("Let's plan some meals.")

    HierarchicalCAPDProcessor().process(
        task,
        CoreLoopContext(
            messages=InMemoryMessageQueue(), inference=inference, system_one=jev,
            prompts=AgentConfigPromptLoader.from_store(config),
        ),
    )

    prompt = inference.requests[0].prompt
    assert "The household prefers shared meals." in prompt
    assert "Keep context files current." not in prompt
    assert "REMOVAL_SENTINEL" not in prompt
    assert task.metadata["v3_context_selection"]["selected_context_ids"] == ["global:household-norms"]
    assert jev.calls == 1
    assert "global:household-norms" in {item["id"] for item in jev.context_candidates}


def test_new_task_is_acknowledged_once_and_continues_to_plan() -> None:
    task = _new_task()
    jev = _AdmissionJev(SystemOneAdmissionDecision(True, 0.97, True, model="test-jev"))
    inference = _DirectInference("Claro, revisaré el proyecto y buscaré la causa.")
    deliveries = []

    def deliver(event):
        deliveries.append(event)
        return {"status": "queued", "outbox_message_id": "ack-1"}

    context = CoreLoopContext(
        messages=InMemoryMessageQueue(), inference=inference, system_one=jev, delivery_sink=deliver,
    )

    assert HierarchicalCAPDProcessor._admit_initial_human_task(task, context) is False
    assert HierarchicalCAPDProcessor._admit_initial_human_task(task, context) is False

    assert jev.calls == 1
    assert len(deliveries) == 1
    assert deliveries[0]["event_type"] == "task.acknowledge"
    assert deliveries[0]["message"] == "Claro, revisaré el proyecto y buscaré la causa."
    assert task.metadata["v3_admission"]["route"] == "task"
    assert task.metadata["v3_admission"]["acknowledgement"]["outbox_message_id"] == "ack-1"
    assert "prepared_user_response" not in task.metadata
    assert len(jev.acknowledgements) == 1


@pytest.mark.parametrize("candidate", ["Done, I've updated the record.", "Listo, ya quedó actualizado.", "All set!"])
def test_premature_done_acknowledgement_is_suppressed_without_stopping_work(candidate):
    task = _new_task("Update the record")
    jev = _AdmissionJev(SystemOneAdmissionDecision(True, 0.97, True))
    jev.acknowledgement_safe = False
    outbox = SQLiteOutboundStore()
    conversations = SQLiteConversationStore()
    context = CoreLoopContext(
        messages=InMemoryMessageQueue(), system_one=jev, inference=_DirectInference(candidate),
        delivery_sink=build_outbox_delivery_sink(outbox=outbox, conversation_store=conversations),
    )
    assert HierarchicalCAPDProcessor._admit_initial_human_task(task, context) is False
    assert task.metadata["v3_admission"]["acknowledgement"] == {"status": "review_rejected"}
    assert task.status != "completed"
    assert not outbox.list()
    assert not conversations.list_recent(owner_user_id="alex")
    assert "prepared_user_response" not in task.metadata


@pytest.mark.parametrize("review", [None, lambda **kwargs: (_ for _ in ()).throw(RuntimeError("offline")), lambda **kwargs: None])
def test_unavailable_or_ambiguous_acknowledgement_review_does_not_block_execution(review):
    task = _new_task()
    jev = _AdmissionJev(SystemOneAdmissionDecision(True, 0.97, True))
    jev.review_task_acknowledgement = review
    deliveries = []
    context = CoreLoopContext(messages=InMemoryMessageQueue(), system_one=jev,
        inference=_DirectInference("Done"), delivery_sink=lambda event: deliveries.append(event))
    assert HierarchicalCAPDProcessor._admit_initial_human_task(task, context) is False
    assert deliveries == []
    assert task.metadata["v3_admission"]["route"] == "task"


def test_cancellation_during_acknowledgement_review_prevents_delivery():
    task = _new_task()
    cancelled = {"value": False}
    jev = _AdmissionJev(SystemOneAdmissionDecision(True, 0.97, True))
    def review(**kwargs):
        cancelled["value"] = True
        return True
    jev.review_task_acknowledgement = review
    deliveries = []
    result = HierarchicalCAPDProcessor().process(task, CoreLoopContext(
        messages=InMemoryMessageQueue(), system_one=jev, inference=_DirectInference("I'll check."),
        cancellation_checker=lambda: cancelled["value"], delivery_sink=lambda event: deliveries.append(event)))
    assert result.status.value == "cancelled"
    assert deliveries == []


def test_ambiguous_admission_continues_but_missing_jev_fails_closed() -> None:
    system_one = _AdmissionJev(SystemOneAdmissionDecision(False, 0.5, False, model="test-jev"))
    task = _new_task()
    deliveries = []
    context = CoreLoopContext(
        messages=InMemoryMessageQueue(), inference=_DirectInference("I’ll review that request now."), system_one=system_one,
        delivery_sink=lambda event: deliveries.append(event) or {"status": "queued"},
    )

    assert HierarchicalCAPDProcessor._admit_initial_human_task(task, context) is False
    assert task.metadata["v3_admission"]["route"] == "task"
    assert len(deliveries) == 1

    with pytest.raises(SystemOneUnavailableError, match="classifier_unavailable"):
        task = _new_task()
        HierarchicalCAPDProcessor._admit_initial_human_task(task, CoreLoopContext(messages=InMemoryMessageQueue()))


def test_acknowledgement_generation_failure_does_not_block_plan_or_send_a_template() -> None:
    task = _new_task()
    jev = _AdmissionJev(SystemOneAdmissionDecision(True, 0.97, True, model="test-jev"))
    deliveries = []

    assert HierarchicalCAPDProcessor._admit_initial_human_task(
        task,
        CoreLoopContext(
            messages=InMemoryMessageQueue(), system_one=jev,
            delivery_sink=lambda event: deliveries.append(event) or {"status": "queued"},
        ),
    ) is False

    assert deliveries == []
    assert task.metadata["v3_admission"]["acknowledgement"] == {"status": "generation_unavailable"}


def test_steering_message_bypasses_one_time_admission() -> None:
    task = _new_task()
    task.metadata["routing_disposition"] = "steering"
    jev = _AdmissionJev(SystemOneAdmissionDecision(False, 0.01, True))

    assert HierarchicalCAPDProcessor._admit_initial_human_task(
        task, CoreLoopContext(messages=InMemoryMessageQueue(), system_one=jev),
    ) is False
    assert jev.calls == 0
    assert "v3_admission" not in task.metadata


def test_acknowledgement_outbox_delivery_is_idempotent_and_recorded() -> None:
    outbox = SQLiteOutboundStore()
    conversations = SQLiteConversationStore()
    sink = build_outbox_delivery_sink(outbox=outbox, conversation_store=conversations)
    task = _new_task()
    event = {
        "event_type": "task.acknowledge",
        "task": task.to_dict(),
        "message": "Revisaré la solicitud de temperatura ahora.",
        "idempotency_key": "v3-early-ack:message-1",
    }

    first = sink(event)
    second = sink(event)

    assert first["outbox_message_id"] == second["outbox_message_id"]
    assert first["integration_id"] == "desktop"
    assert first["channel_target"] == "alex"
    assert outbox.status_counts()["pending"] == 1
    timeline = conversations.list(owner_user_id="alex", project_id="home")
    assert [item.content for item in timeline] == ["Revisaré la solicitud de temperatura ahora."]


@pytest.mark.parametrize("status", ["pending", "running", "waiting_user", "failed", "cancelled", "completed"])
def test_final_response_delivery_requires_completed_task_even_when_prepared(status):
    from alphonse.agent_v2.core.core import StateSnapshot
    from alphonse.agent_v2.core.io import project_snapshot_to_outbox
    task = _new_task()
    task.status = status
    task.metadata["prepared_user_response"] = {"message": "The record was updated."}
    outbox = SQLiteOutboundStore()
    result = project_snapshot_to_outbox(snapshot=StateSnapshot(metadata={"task_state": task.to_dict()}), outbox=outbox)
    assert bool(result) is (status == "completed")
    assert len(outbox.list()) == (1 if status == "completed" else 0)


def test_parked_task_does_not_announce_prepared_completion_through_daemon():
    from alphonse.agent_v2.core.core import ProcessingResult, ProcessingStatus, StateSnapshot
    from alphonse.agent_v2.core.state import reset_state
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.daemon import V2Daemon
    reset_state()
    runtime = build_runtime_host()
    runtime.conversation_store.set_daily_ledger_projector(None)
    runtime.core.memory = None
    class ParkedProcessor:
        def process(self, task, context):
            task.status = "waiting_user"
            task.metadata["prepared_user_response"] = {"message": "Done, updated the record."}
            return ProcessingResult(status=ProcessingStatus.PARKED,
                snapshot=StateSnapshot(metadata={"task_state": task.to_dict()}))
    runtime.core.intelligence = ParkedProcessor()
    runtime.channel.queue_message(prompt="Update the record", user="alex", integration_id="desktop",
        provider_key="desktop", channel_target="alex")
    daemon = V2Daemon(runtime)
    assert daemon.run_once().status.value == "parked"
    assert runtime.outbox.list() == []
    assert not [event for event in runtime.conversation_store.list_recent(owner_user_id="alex") if event.role == "assistant"]



def test_queued_work_notice_keeps_its_identity_without_claiming_completion(tmp_path):
    from alphonse.agent_v2.core.messages import CommunicationChannel, SQLiteMessageQueue
    from alphonse.agent_v2.core.projects import ProjectStore
    from alphonse.agent_v2.services.project_sessions import ProjectInboundRouter, SQLiteProjectSessionStore
    queue = SQLiteMessageQueue(tmp_path / "queue.sqlite3")
    outbox = SQLiteOutboundStore()
    active = {}
    router = ProjectInboundRouter(channel=CommunicationChannel(queue), outbox=outbox,
        projects=ProjectStore(tmp_path / "projects.sqlite3"),
        sessions=SQLiteProjectSessionStore(tmp_path / "sessions.sqlite3"),
        managed_root=lambda user: tmp_path / "projects" / user, active_task_lookup=lambda: active)
    router.ingest(prompt="First", user="alex", integration_id="desktop", provider_key="desktop", channel_target="alex")
    first = queue.claim_next()
    active.update({"user": "alex", "project_id": first.message.project_id, "routing_disposition": "pdca_task"})
    routed = router.ingest(prompt="Second", user="gaby", integration_id="desktop", provider_key="desktop", channel_target="gaby")
    notice = outbox.list()[-1]
    assert notice.kind == "queue_waiting_notice"
    assert notice.task_id == routed.queued.message_id
    assert notice.audience_user_id == "gaby" and notice.channel_target == "gaby"
    assert "queued" in notice.message and "after 1 task ahead" in notice.message
    assert "complete" not in notice.message.lower()
