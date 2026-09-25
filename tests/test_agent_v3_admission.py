from __future__ import annotations

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferenceResult
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.io import SQLiteOutboundStore, build_outbox_delivery_sink, channel_metadata
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.system_one import SystemOneAdmissionDecision


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

    def classify_task_admission(self, *, message: str) -> SystemOneAdmissionDecision:
        assert message
        self.calls += 1
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


def test_ambiguous_or_unavailable_admission_falls_back_to_acknowledged_task() -> None:
    for system_one in (
        _AdmissionJev(SystemOneAdmissionDecision(False, 0.5, False, model="test-jev")),
        None,
    ):
        task = _new_task()
        deliveries = []
        context = CoreLoopContext(
            messages=InMemoryMessageQueue(), inference=_DirectInference("I’ll review that request now."), system_one=system_one,
            delivery_sink=lambda event: deliveries.append(event) or {"status": "queued"},
        )

        assert HierarchicalCAPDProcessor._admit_initial_human_task(task, context) is False
        assert task.metadata["v3_admission"]["route"] == "task"
        assert len(deliveries) == 1


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
