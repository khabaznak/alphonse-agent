from __future__ import annotations

from alphonse.agent_v2.core.core import CoreActivityEvent
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.core.intelligence import PDCAIntelligenceProcessor
from alphonse.agent_v2.interfaces.tui import build_tui_runtime
from alphonse.agent_v2.interfaces.tui import format_activity_message
from alphonse.agent_v2.interfaces.tui import process_tui_queue_once
from alphonse.agent_v2.interfaces.tui import queue_tui_input
from alphonse.agent_v2.interfaces.tui import submit_tui_input
from alphonse.agent_v2.interfaces.tui import TuiProcessorCoordinator


def test_tui_runtime_factory_wires_core_services() -> None:
    runtime = build_tui_runtime(user="alex")

    assert runtime.user == "alex"
    assert runtime.queue.size() == 0
    assert runtime.core.messages is runtime.queue
    assert runtime.channel.messages is runtime.queue
    assert isinstance(runtime.processor, PDCAIntelligenceProcessor)
    assert runtime.core.inference is not None


def test_submitting_input_queues_steps_and_updates_visible_state() -> None:
    runtime = build_tui_runtime(user="alex", inference=_respond_router())

    result = submit_tui_input(runtime, "hello")

    assert result.queued
    assert result.step_status == LoopStepStatus.PROCESSED
    assert runtime.queue.size() == 0
    assert runtime.visible_state.snapshot().current_work == "hello"
    assert result.response == "Hello, Alex."
    assert runtime.visible_state.snapshot().metadata["check_verdict"] == "mission_success"


def test_queue_tui_input_queues_without_processing() -> None:
    runtime = build_tui_runtime(user="alex", inference=_respond_router())

    result = queue_tui_input(runtime, "hello")

    assert result.queued
    assert result.step_status is None
    assert result.response == ""
    assert runtime.queue.size() == 1
    assert runtime.visible_state.snapshot().current_work is None


def test_queue_tui_input_allows_multiple_messages_while_processor_reserved() -> None:
    runtime = build_tui_runtime(user="alex", inference=_respond_router())
    coordinator = TuiProcessorCoordinator(runtime)

    assert coordinator.reserve_processing() is True
    first = queue_tui_input(runtime, "hello")
    second = queue_tui_input(runtime, "one more thing")

    assert first.queued
    assert second.queued
    assert runtime.queue.size() == 2
    assert coordinator.reserve_processing() is False


def test_processor_coordinator_runs_queue_until_idle_and_releases_reservation() -> None:
    runtime = build_tui_runtime(user="alex", inference=_respond_router())
    coordinator = TuiProcessorCoordinator(runtime)
    queue_tui_input(runtime, "hello")

    assert coordinator.reserve_processing() is True
    results = coordinator.process_until_idle()

    assert len(results) == 1
    assert results[0].step_status == LoopStepStatus.PROCESSED
    assert results[0].response == "Hello, Alex."
    assert runtime.queue.size() == 0
    assert coordinator.is_processing is False


def test_process_tui_queue_once_displays_response_after_queue_only_submit() -> None:
    runtime = build_tui_runtime(user="alex", inference=_respond_router())
    queue_tui_input(runtime, "hello")

    result = process_tui_queue_once(runtime)

    assert result.step_status == LoopStepStatus.PROCESSED
    assert result.response == "Hello, Alex."


def test_capd_activity_events_include_phase_signifiers_and_plan_internal_state() -> None:
    runtime = build_tui_runtime(user="alex", inference=_respond_router())
    events: list[CoreActivityEvent] = []
    runtime.core.activity_sink = events.append

    submit_tui_input(runtime, "hello")

    labels = [event.label for event in events]
    messages = [event.message for event in events]
    assert "deliberating" in labels
    assert "deciding" in labels
    assert "thinking" in labels
    assert "working" in labels
    assert "Answering the greeting." in messages


def test_format_activity_message_renders_label_and_message() -> None:
    event = CoreActivityEvent(
        phase=ImprovementPhase.PLAN,
        label="thinking",
        message="Answering the greeting.",
    )

    assert format_activity_message(event) == "thinking - Answering the greeting."


def test_submitting_shell_prompt_displays_bash_stdout() -> None:
    runtime = build_tui_runtime(user="alex", inference=_bash_router())

    result = submit_tui_input(runtime, "ls -latr")

    assert result.queued
    assert result.step_status == LoopStepStatus.PROCESSED
    assert result.response == "hello"
    assert runtime.visible_state.snapshot().metadata["check_verdict"] == "mission_success"


def test_submit_stop_exits_without_queueing_message() -> None:
    runtime = build_tui_runtime(user="alex")

    result = submit_tui_input(runtime, "/stop")

    assert result.should_exit
    assert not result.queued
    assert runtime.queue.size() == 0


def test_queue_stop_exits_without_queueing_message() -> None:
    runtime = build_tui_runtime(user="alex")

    result = queue_tui_input(runtime, "/stop")

    assert result.should_exit
    assert not result.queued
    assert runtime.queue.size() == 0


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


def _bash_router() -> InferenceRouter:
    return InferenceRouter(
        provider=StubInferenceProvider(
            markdown_by_purpose={
                InferencePurpose.ACCEPTANCE_CRITERIA: "1.- [ ] Greeting was printed",
                InferencePurpose.CRITERIA_REVIEW: "1.- [x] Greeting was printed",
            },
            tool_call={
                "tool_id": "native.bash",
                "tool_name": "bash",
                "arguments": {"command": "printf hello"},
                "internal_state": "Printing a greeting.",
            },
        ),
        default_profile=ModelProfile(provider="stub", model="stub", profile_id="stub"),
    )
