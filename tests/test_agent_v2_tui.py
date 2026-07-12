from __future__ import annotations

from alphonse.agent import identity
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent_v2.core.core import CoreActivityEvent
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.inference import InferencePurpose
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import StubInferenceProvider
from alphonse.agent_v2.core.intelligence import PDCAIntelligenceProcessor
from alphonse.agent_v2.integrations import IntegrationDescriptor
from alphonse.agent_v2.integrations import IntegrationRegistry
from alphonse.agent_v2.interfaces.tui import build_tui_runtime
from alphonse.agent_v2.interfaces.tui import format_current_project_status
from alphonse.agent_v2.interfaces.tui import format_daemon_connection_status
from alphonse.agent_v2.interfaces.tui import format_daemon_active_work
from alphonse.agent_v2.interfaces.tui import format_activity_message
from alphonse.agent_v2.interfaces.tui import format_activity_status_line
from alphonse.agent_v2.interfaces.tui import format_global_activity_state
from alphonse.agent_v2.interfaces.tui import format_global_activity_status
from alphonse.agent_v2.interfaces.tui import format_inference_status
from alphonse.agent_v2.interfaces.tui import format_inference_settings_status
from alphonse.agent_v2.interfaces.tui import inference_status_parts
from alphonse.agent_v2.interfaces.tui import format_slash_command_suggestions
from alphonse.agent_v2.interfaces.tui import matching_slash_commands
from alphonse.agent_v2.interfaces.tui import process_tui_queue_once
from alphonse.agent_v2.interfaces.tui import queue_tui_input
from alphonse.agent_v2.interfaces.tui import list_integration_options
from alphonse.agent_v2.interfaces.tui import save_telegram_integration_config
from alphonse.agent_v2.interfaces.tui import start_enabled_integration_runtimes
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
    delivered = runtime.outbox.list()
    assert len(delivered) == 1
    assert delivered[0].integration_id == "tui"
    assert delivered[0].channel_target == "alex"
    assert delivered[0].status == "delivered"


def test_process_tui_queue_once_processes_integration_message_for_canonical_user() -> None:
    runtime = build_tui_runtime(user="local", inference=_respond_router())
    runtime.channel.queue_message(
        prompt="hello",
        user="u-alex",
        integration_id="telegram-home",
        provider_key="telegram",
        provider_user_id="8553589429",
        channel_target="8553589429",
        provider_message_id="5",
    )

    result = process_tui_queue_once(runtime)

    assert result.step_status == LoopStepStatus.PROCESSED
    assert result.response == ""
    pending = runtime.outbox.list()
    assert len(pending) == 1
    assert pending[0].integration_id == "telegram-home"
    assert pending[0].channel_target == "8553589429"
    assert pending[0].status == "pending"


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
    assert all(event.message_id for event in events)
    assert all(event.user == "alex" for event in events)
    assert all(event.integration_id == "tui" for event in events)
    assert all(event.channel_target == "alex" for event in events)


def test_format_activity_message_renders_label_and_message() -> None:
    event = CoreActivityEvent(
        phase=ImprovementPhase.PLAN,
        label="thinking",
        message="Answering the greeting.",
    )

    assert format_activity_message(event) == "thinking - Answering the greeting."


def test_format_activity_status_line_renders_morphing_internal_state() -> None:
    event = CoreActivityEvent(
        phase=ImprovementPhase.PLAN,
        label="thinking",
        message="Choosing the next tool call.",
    )

    assert format_activity_status_line(event) == "Thinking: Choosing the next tool call."


def test_format_activity_status_line_falls_back_to_phase() -> None:
    event = CoreActivityEvent(
        phase=ImprovementPhase.CHECK,
        label="",
        message="",
    )

    assert format_activity_status_line(event) == "Check"


def test_global_activity_status_is_safe_and_reports_terminal_states() -> None:
    event = CoreActivityEvent(
        phase=ImprovementPhase.PLAN,
        label="thinking",
        message="Sensitive task detail",
        user="other-user",
    )

    assert format_global_activity_status(event, 1) == "Working: Planning /"
    assert format_global_activity_state({"state": "waiting"}) == "Waiting: user input"
    assert format_global_activity_state({"state": "error"}, error="token limit reached") == "Error: model token limit"
    assert format_global_activity_state({"state": "idle"}) == "Idle: ready"


def test_format_slash_command_suggestions_filters_matches() -> None:
    assert "> /integrations - Configure optional integrations" in format_slash_command_suggestions("/")
    assert "> /project-context - Edit the active project context" in format_slash_command_suggestions("/project-c")
    assert "/project - Select or create a project" not in format_slash_command_suggestions("/project-c")
    assert format_slash_command_suggestions(" /project") == ""


def test_format_slash_command_suggestions_marks_selected_row() -> None:
    rendered = format_slash_command_suggestions("/", selected_index=1)

    assert "  /integrations - Configure optional integrations" in rendered
    assert "> /project - Select or create a project" in rendered
    assert [command for command, _ in matching_slash_commands("/project")] == ["/project", "/project-context"]


def test_queue_integrations_command_opens_command_flow() -> None:
    runtime = build_tui_runtime(user="alex")

    result = queue_tui_input(runtime, "/integrations")

    assert result.command == "integrations"
    assert result.queued is False


def test_queue_model_commands_open_local_command_flows() -> None:
    runtime = build_tui_runtime(user="alex")

    assert queue_tui_input(runtime, "/model").command == "model"
    assert queue_tui_input(runtime, "/model-provider").command == "model-provider"
    assert queue_tui_input(runtime, "/agent-config").command == "agent-config"
    assert queue_tui_input(runtime, "/scheduled-tasks").command == "scheduled-tasks"


def test_tui_integration_options_and_telegram_config_save() -> None:
    runtime = build_tui_runtime(user="alex")

    assert list_integration_options(runtime) == [("Telegram - disabled (telegram-home)", "telegram")]
    record = save_telegram_integration_config(
        runtime,
        integration_id="telegram-home",
        display_name="Telegram Home",
        bot_token="token",
        poll_interval_sec="2.5",
        allowed_chat_ids="123, -456",
        enabled=True,
    )

    assert record.enabled is True
    assert record.config["owner_user_id"] == "alex"
    assert record.config["poll_interval_sec"] == 2.5
    assert record.config["allowed_chat_ids"] == ["123", "-456"]
    assert record.secrets["bot_token"] == "token"
    assert list_integration_options(runtime) == [("Telegram - enabled (telegram-home)", "telegram")]


def test_telegram_config_save_maps_current_tui_user(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    runtime = build_tui_runtime(user="u-alex")

    record = save_telegram_integration_config(
        runtime,
        integration_id="telegram-home",
        display_name="Telegram Home",
        bot_token="token",
        telegram_user_id="123",
        enabled=True,
    )

    assert record.config["telegram_user_id"] == "123"
    assert identity.resolve_user_id(service_id=2, service_user_id="123") == "u-alex"
    assert identity.resolve_service_user_id(user_id="u-alex", service_id=2) == "123"


def test_disabling_telegram_config_keeps_identity_mapping(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    runtime = build_tui_runtime(user="u-alex")
    save_telegram_integration_config(
        runtime,
        integration_id="telegram-home",
        display_name="Telegram Home",
        bot_token="token",
        telegram_user_id="123",
        enabled=True,
    )

    save_telegram_integration_config(
        runtime,
        integration_id="telegram-home",
        display_name="Telegram Home",
        enabled=False,
    )

    assert identity.resolve_user_id(service_id=2, service_user_id="123") == "u-alex"


def test_no_enabled_integrations_keeps_tui_runtime_without_background_runtimes() -> None:
    runtime = build_tui_runtime(user="alex")

    started = start_enabled_integration_runtimes(runtime)

    assert started == []
    assert runtime.integration_runtimes == []


def test_start_enabled_integrations_passes_message_wake_callback() -> None:
    captured: dict[str, object] = {}

    class FakeIntegrationRuntime:
        def __init__(self) -> None:
            self.started = False

        def start(self) -> None:
            self.started = True

    def fake_runtime_factory(**kwargs):
        captured.update(kwargs)
        return FakeIntegrationRuntime()

    runtime = build_tui_runtime(
        user="alex",
        integration_registry=IntegrationRegistry(
            (
                IntegrationDescriptor(
                    provider_key="fake",
                    display_name="Fake",
                    description="Fake integration",
                    default_integration_id="fake-home",
                    runtime_factory=fake_runtime_factory,
                ),
            )
        ),
    )
    runtime.integration_store.upsert(
        integration_id="fake-home",
        provider_key="fake",
        display_name="Fake",
        enabled=True,
        config={},
        secrets={},
    )
    wakes: list[str] = []

    started = start_enabled_integration_runtimes(runtime, on_message_queued=lambda: wakes.append("wake"))

    assert len(started) == 1
    assert started[0].started is True
    assert captured["record"].integration_id == "fake-home"
    captured["on_message_queued"]()
    assert wakes == ["wake"]


def test_sidebar_status_helpers_render_project_and_inference(tmp_path) -> None:
    runtime = build_tui_runtime(user="alex", inference=_respond_router())
    project = runtime.project_store.create_project(
        name="Alpha",
        root_path=str(tmp_path / "alpha"),
        owner_user_id="alex",
    )

    assert format_current_project_status(runtime) == "None"
    runtime.active_project_id = project.project_id
    assert format_current_project_status(runtime) == "Alpha"
    assert format_inference_status(runtime) == "stub / stub"
    assert format_inference_settings_status({"provider_key": "openai_codex", "model_id": ""}) == "openai_codex / default"
    assert inference_status_parts({"provider_key": "openai_codex", "model_id": "gpt-5.5"}) == ("openai_codex", "gpt-5.5")


def test_daemon_connection_status_distinguishes_external_embedded_and_disconnected() -> None:
    assert format_daemon_connection_status("connected") == "connected"
    assert format_daemon_connection_status("embedded") == "connected (embedded)"
    assert format_daemon_connection_status("disconnected") == "disconnected"
    assert format_daemon_connection_status("unexpected") == "disconnected"


def test_daemon_active_work_formatting_collapses_whitespace() -> None:
    assert format_daemon_active_work({"user": "alex", "prompt": "  Reply\nnow  "}) == "alex: Reply now"
    assert format_daemon_active_work({}) == ""


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
