from __future__ import annotations

from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.core.intelligence import PDCAIntelligenceProcessor
from alphonse.agent_v2.interfaces.tui import build_tui_runtime
from alphonse.agent_v2.interfaces.tui import submit_tui_input


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
