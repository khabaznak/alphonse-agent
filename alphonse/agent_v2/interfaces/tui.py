"""Textual native interface for Alphonse v2."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

from alphonse.agent_v2.core.core import AlphonseCore
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.core import CoreActivityEvent
from alphonse.agent_v2.core.core import CoreUiEvent
from alphonse.agent_v2.core.core import IntelligenceProcessor
from alphonse.agent_v2.core.core import LoopStepResult
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.core import MemoryRecord
from alphonse.agent_v2.core.core import PromptFile
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolRegistry
from alphonse.agent_v2.core.inference import InferenceRouter
from alphonse.agent_v2.core.inference import ModelProfile
from alphonse.agent_v2.core.inference import OpenAICodexProvider
from alphonse.agent_v2.core.intelligence import PDCAIntelligenceProcessor
from alphonse.agent_v2.core.io import OutboundSelector
from alphonse.agent_v2.core.io import IntegrationIdentity
from alphonse.agent_v2.core.io import SQLiteOutboundStore
from alphonse.agent_v2.core.io import V2IdentityResolver
from alphonse.agent_v2.core.io import build_outbox_delivery_sink
from alphonse.agent_v2.core.io import channel_address_from_metadata
from alphonse.agent_v2.core.io import project_snapshot_to_outbox
from alphonse.agent_v2.core.io import resolve_provider_user_mapping
from alphonse.agent_v2.core.io import upsert_provider_user_mapping
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages import MessageSelector
from alphonse.agent_v2.core.messages import SQLiteMessageQueue
from alphonse.agent_v2.core.projects import ProjectRecord
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.state import get_state
from alphonse.agent_v2.core.state import reset_state
from alphonse.agent_v2.core.tools.registry.native import BASH_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import RESPOND_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import SCHEDULED_TASK_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.integrations import IntegrationConfigRecord
from alphonse.agent_v2.integrations import IntegrationRegistry
from alphonse.agent_v2.integrations import SQLiteIntegrationStore
from alphonse.agent_v2.integrations import build_default_integration_registry
from alphonse.agent_v2.integrations.presence import PresenceProjector
from alphonse.agent_v2.integrations.presence import TuiPresenceAdapter
from alphonse.agent_v2.runtime import InMemoryInternalState
from alphonse.agent_v2.runtime import NullMemory
from alphonse.agent_v2.runtime import NullPromptLoader
from alphonse.agent_v2.runtime import V2RuntimeHost
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonClient

TuiRuntime = V2RuntimeHost

LOCAL_STOP_COMMANDS = {"/exit", "/quit", "/stop"}
TUI_SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/integrations", "Configure optional integrations"),
    ("/project", "Select or create a project"),
    ("/project-context", "Edit the active project context"),
    ("/stop", "Stop Alphonse"),
    ("/exit", "Exit the TUI"),
    ("/quit", "Exit the TUI"),
)


@dataclass(frozen=True)
class TuiSubmissionResult:
    prompt: str
    response: str
    event: str
    should_exit: bool = False
    queued: bool = False
    queued_message_id: str | None = None
    step_status: LoopStepStatus | None = None
    command: str = ""


@dataclass(frozen=True)
class TuiProcessingResult:
    response: str
    event: str
    step_status: LoopStepStatus
    queued_message_id: str | None = None


class TuiProcessorCoordinator:
    """Keeps TUI queueing separate from one-at-a-time CAPD processing."""

    def __init__(self, runtime: TuiRuntime) -> None:
        self.runtime = runtime
        self._lock = Lock()
        self._processing = False

    @property
    def is_processing(self) -> bool:
        with self._lock:
            return self._processing

    def reserve_processing(self) -> bool:
        with self._lock:
            if self._processing:
                return False
            self._processing = True
            return True

    def process_until_idle(self) -> list[TuiProcessingResult]:
        results: list[TuiProcessingResult] = []
        try:
            while True:
                result = process_tui_queue_once(self.runtime)
                if result.step_status == LoopStepStatus.EMPTY:
                    break
                results.append(result)
                if result.step_status in {LoopStepStatus.BUSY, LoopStepStatus.STOPPED}:
                    break
        finally:
            with self._lock:
                self._processing = False
        return results


class NullToolRegistry:
    def register(self, tool: ToolDescriptor) -> None:
        _ = tool

    def get(self, name: str) -> ToolDescriptor | None:
        _ = name
        return None

    def list(self) -> tuple[ToolDescriptor, ...]:
        return ()

    def execute(self, tool_id: str, arguments: dict[str, Any], execution_context: Any | None = None) -> Any:
        _ = tool_id
        _ = arguments
        _ = execution_context
        raise KeyError("tool_not_found")


def build_tui_runtime(
    *,
    user: str = "local",
    inference: InferenceRouter | None = None,
    tools: ToolRegistry | None = None,
    processor: IntelligenceProcessor | None = None,
    question_store: SQLiteQuestionStore | None = None,
    project_store: ProjectStore | None = None,
    schedule_store: ScheduledTaskStore | None = None,
    outbox: SQLiteOutboundStore | None = None,
    identity_resolver: V2IdentityResolver | None = None,
    integration_store: SQLiteIntegrationStore | None = None,
    integration_registry: IntegrationRegistry | None = None,
) -> TuiRuntime:
    return build_runtime_host(
        user=user,
        inference=inference,
        tools=tools,
        processor=processor,
        question_store=question_store,
        project_store=project_store,
        schedule_store=schedule_store,
        outbox=outbox,
        identity_resolver=identity_resolver,
        integration_store=integration_store,
        integration_registry=integration_registry,
    )


def build_default_tui_inference_router() -> InferenceRouter:
    """Build the default live inference router for the native TUI."""
    return InferenceRouter(
        provider=OpenAICodexProvider(),
        default_profile=ModelProfile(
            provider="openai_codex",
            model=os.getenv("OPENAI_CODEX_MODEL", ""),
            profile_id="chatgpt-plus-codex",
            supports_tool_calling=False,
            supports_structured_output=False,
            supports_json_mode=True,
            cost_tier="subscription",
        ),
    )


def submit_tui_input(runtime: TuiRuntime, prompt: str) -> TuiSubmissionResult:
    """Queue input and process it synchronously.

    This helper is useful for tests and non-interactive use. The Textual app uses
    queue_tui_input plus a background worker so the UI can keep accepting input.
    """
    queued = queue_tui_input(runtime, prompt)
    if not queued.queued:
        return queued

    processed = process_tui_queue_once(runtime)
    return TuiSubmissionResult(
        prompt=queued.prompt,
        response=processed.response,
        event=f"{queued.event}; {processed.event}",
        queued=True,
        queued_message_id=queued.queued_message_id,
        step_status=processed.step_status,
    )


def queue_tui_input(runtime: TuiRuntime, prompt: str) -> TuiSubmissionResult:
    raw_prompt = str(prompt or "")
    prompt_value = raw_prompt.strip()
    if not prompt_value:
        return TuiSubmissionResult(prompt="", response="", event="empty input ignored")

    if prompt_value in LOCAL_STOP_COMMANDS:
        runtime.core.request_stop()
        return TuiSubmissionResult(
            prompt=prompt_value,
            response="Stopping Alphonse.",
            event="stop requested",
            should_exit=True,
        )

    command = detect_tui_slash_command(raw_prompt)
    if command in {"project", "project-context", "integrations"}:
        return TuiSubmissionResult(
            prompt=prompt_value,
            response="",
            event=f"command=/{command}",
            command=command,
        )

    pending_result = _route_pending_question_answer(runtime, prompt_value)
    if pending_result is not None:
        return pending_result

    queued = runtime.channel.queue_message(
        prompt=prompt_value,
        user=runtime.user,
        project_id=runtime.active_project_id,
    )
    command_event = ""
    if queued.message.metadata.get("is_command"):
        command_event = f" command=/{queued.message.metadata.get('command')}"
    return TuiSubmissionResult(
        prompt=prompt_value,
        response="",
        event=f"queued {queued.message_id}{command_event}",
        queued=True,
        queued_message_id=queued.message_id,
    )


def detect_tui_slash_command(prompt: str) -> str:
    raw_prompt = str(prompt or "")
    if not raw_prompt.startswith("/"):
        return ""
    command_body = raw_prompt[1:]
    if not command_body or command_body[0].isspace():
        return ""
    return command_body.split(maxsplit=1)[0]


def matching_slash_commands(prompt: str) -> list[tuple[str, str]]:
    raw_prompt = str(prompt or "")
    if not raw_prompt.startswith("/"):
        return []
    typed = raw_prompt.split(maxsplit=1)[0]
    return [(command, description) for command, description in TUI_SLASH_COMMANDS if command.startswith(typed)]


def build_identity_resolver_from_integrations(store: SQLiteIntegrationStore) -> V2IdentityResolver:
    identities = [IntegrationIdentity("tui", "tui")]
    identities.extend(
        IntegrationIdentity(record.integration_id, record.provider_key)
        for record in store.list_enabled()
    )
    return V2IdentityResolver(tuple(identities))


def refresh_tui_identity_resolver(runtime: TuiRuntime) -> None:
    runtime.identity_resolver = build_identity_resolver_from_integrations(runtime.integration_store)
    runtime.core.delivery_sink = build_outbox_delivery_sink(
        outbox=runtime.outbox,
        identity_resolver=runtime.identity_resolver,
    )


def start_enabled_integration_runtimes(
    runtime: TuiRuntime,
    *,
    on_message_queued: Callable[[], None] | None = None,
) -> list[Any]:
    stop_integration_runtimes(runtime)
    refresh_tui_identity_resolver(runtime)
    started: list[Any] = []
    for record in runtime.integration_store.list_enabled():
        descriptor = runtime.integration_registry.get(record.provider_key)
        if descriptor is None or descriptor.runtime_factory is None:
            continue
        try:
            integration_runtime = descriptor.runtime_factory(
                record=record,
                channel=runtime.channel,
                outbox=runtime.outbox,
                identity_resolver=runtime.identity_resolver,
                owner_user_id=runtime.user,
                on_message_queued=on_message_queued,
                presence_projector=runtime.presence_projector,
            )
            integration_runtime.start()
        except Exception:
            continue
        presence_adapter = getattr(integration_runtime, "presence_adapter", None)
        if presence_adapter is not None:
            runtime.presence_projector.register(record.integration_id, presence_adapter)
        started.append(integration_runtime)
    runtime.integration_runtimes = started
    return started


def stop_integration_runtimes(runtime: TuiRuntime) -> None:
    for integration_runtime in list(runtime.integration_runtimes):
        integration_id = str(getattr(integration_runtime, "integration_id", "") or "").strip()
        if integration_id:
            runtime.presence_projector.unregister(integration_id)
        stop = getattr(integration_runtime, "stop", None)
        if callable(stop):
            stop()
    runtime.integration_runtimes = []


def list_integration_options(runtime: TuiRuntime) -> list[tuple[str, str]]:
    options: list[tuple[str, str]] = []
    for descriptor in runtime.integration_registry.list():
        record = runtime.integration_store.get_by_provider(descriptor.provider_key)
        status = "enabled" if record is not None and record.enabled else "disabled"
        integration_id = record.integration_id if record is not None else descriptor.default_integration_id
        options.append((f"{descriptor.display_name} - {status} ({integration_id})", descriptor.provider_key))
    return options


def save_telegram_integration_config(
    runtime: TuiRuntime,
    *,
    integration_id: str,
    display_name: str,
    bot_token: str = "",
    poll_interval_sec: str | float = "1.0",
    allowed_chat_ids: str = "",
    telegram_user_id: str = "",
    enabled: bool = False,
    remove_token: bool = False,
    presence_enabled: bool = True,
) -> IntegrationConfigRecord:
    existing = runtime.integration_store.get(integration_id) or runtime.integration_store.get_by_provider("telegram")
    secrets = dict(existing.secrets) if existing is not None else {}
    token = str(bot_token or "").strip()
    if remove_token:
        secrets.pop("bot_token", None)
    elif token:
        secrets["bot_token"] = token
    if enabled and not str(secrets.get("bot_token") or "").strip():
        raise ValueError("telegram_bot_token_required")
    provider_user_id = str(telegram_user_id or "").strip()
    if provider_user_id:
        upsert_provider_user_mapping(
            alphonse_user_id=runtime.user,
            provider_key="telegram",
            provider_user_id=provider_user_id,
            display_name=runtime.user,
            is_active=True,
        )
    record = runtime.integration_store.upsert(
        integration_id=str(integration_id or "").strip(),
        provider_key="telegram",
        display_name=str(display_name or "").strip() or "Telegram",
        enabled=enabled,
        config={
            "poll_interval_sec": _coerce_poll_interval(poll_interval_sec),
            "allowed_chat_ids": _parse_chat_ids(allowed_chat_ids),
            "owner_user_id": runtime.user,
            "telegram_user_id": provider_user_id,
            "presence_enabled": presence_enabled,
        },
        secrets=secrets,
    )
    refresh_tui_identity_resolver(runtime)
    return record


def _parse_chat_ids(value: str) -> list[str]:
    return [entry.strip() for entry in str(value or "").split(",") if entry.strip()]


def _coerce_poll_interval(value: str | float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 1.0
    return parsed if parsed > 0 else 1.0


def format_slash_command_suggestions(prompt: str, *, selected_index: int = 0) -> str:
    matches = matching_slash_commands(prompt)
    if not matches:
        return ""
    selected = min(max(0, selected_index), len(matches) - 1)
    lines = ["Commands"]
    lines.extend(
        f"{'> ' if index == selected else '  '}{command} - {description}"
        for index, (command, description) in enumerate(matches)
    )
    return "\n".join(lines)


def select_tui_project(runtime: TuiRuntime, project_id: str) -> ProjectRecord:
    project = runtime.project_store.get_project(project_id, requester_user_id=runtime.user)
    if project is None:
        raise KeyError(f"project_not_found: {project_id}")
    runtime.active_project_id = project.project_id
    return project


def create_tui_project(
    runtime: TuiRuntime,
    *,
    name: str,
    description: str,
    root_path: str,
    visibility: str = "private",
) -> ProjectRecord:
    project = runtime.project_store.create_project(
        name=name,
        description=description,
        root_path=root_path,
        visibility=visibility,  # type: ignore[arg-type]
        owner_user_id=runtime.user,
    )
    runtime.active_project_id = project.project_id
    return project


def save_tui_project_context(runtime: TuiRuntime, content: str) -> ProjectRecord:
    if not runtime.active_project_id:
        raise RuntimeError("active_project_required")
    return runtime.project_store.write_project_context(
        runtime.active_project_id,
        content,
        requester_user_id=runtime.user,
    )


def process_tui_queue_once(runtime: TuiRuntime) -> TuiProcessingResult:
    queued = runtime.queue.peek()
    with runtime.presence_projector.processing(queued):
        step = runtime.core.step()
        if step.status in {
            LoopStepStatus.PROCESSED,
            LoopStepStatus.PARKED,
            LoopStepStatus.WAITING,
            LoopStepStatus.FAILED,
        }:
            runtime.presence_projector.finish(
                failed=step.status == LoopStepStatus.FAILED,
                waiting=step.status in {LoopStepStatus.PARKED, LoopStepStatus.WAITING},
            )
    snapshot = runtime.visible_state.snapshot()
    if step.status in {LoopStepStatus.PROCESSED, LoopStepStatus.PARKED, LoopStepStatus.WAITING}:
        project_snapshot_to_outbox(snapshot=snapshot, outbox=runtime.outbox)
    response = _drain_tui_outbox(runtime)
    if not response and _snapshot_origin_is_tui(snapshot):
        response = _response_from_snapshot(snapshot, step)
    return TuiProcessingResult(
        response=response,
        event=f"step={step.status.value}",
        step_status=step.status,
        queued_message_id=step.queued_message_id,
    )


def _drain_tui_outbox(runtime: TuiRuntime) -> str:
    selector = OutboundSelector(integration_id="tui", channel_target=runtime.user, status="pending")
    messages: list[str] = []
    while True:
        outbound = runtime.outbox.claim_next(selector)
        if outbound is None:
            break
        messages.append(outbound.message)
        runtime.outbox.mark_delivered(outbound.outbox_message_id)
    return "\n".join(message for message in messages if message)


def _snapshot_origin_is_tui(snapshot: StateSnapshot) -> bool:
    metadata = snapshot.metadata or {}
    task_state = metadata.get("task_state")
    if not isinstance(task_state, dict):
        return True
    task_metadata = task_state.get("metadata") if isinstance(task_state.get("metadata"), dict) else {}
    origin = channel_address_from_metadata(task_metadata)
    return origin is None or origin.integration_id == "tui"


def _response_from_snapshot(snapshot: StateSnapshot, step: Any) -> str:
    if step.status == LoopStepStatus.FAILED:
        return f"CAPD failed: {step.error or 'unknown error'}"

    metadata = snapshot.metadata or {}
    if "response" in metadata:
        return str(metadata.get("response") or "")

    question = metadata.get("question_interrupt")
    if isinstance(question, dict):
        return _render_question_interrupt(question)

    task_state = metadata.get("task_state")
    if isinstance(task_state, dict):
        verdict = str(metadata.get("check_verdict") or task_state.get("check_verdict") or "unknown")
        status = str(metadata.get("status") or task_state.get("status") or "unknown")
        route = str(metadata.get("act_route") or task_state.get("metadata", {}).get("act_route") or "unknown")
        planned = metadata.get("planned_tool_call")
        tool_result_response = _latest_tool_result_response(task_state)
        if tool_result_response:
            return tool_result_response
        if verdict == "mission_success":
            if isinstance(planned, dict):
                tool_name = str(planned.get("tool_name") or planned.get("tool_id") or "tool")
                return f"Done. CAPD completed the task using {tool_name}."
            return "Done. CAPD completed the task."
        if verdict == "mission_failed":
            outcome = metadata.get("outcome")
            if isinstance(outcome, dict) and outcome.get("reason"):
                return f"CAPD could not complete the task: {outcome['reason']}"
            return "CAPD could not complete the task."
        if isinstance(planned, dict):
            tool_name = str(planned.get("tool_name") or planned.get("tool_id") or "tool")
            return f"CAPD status={status}; verdict={verdict}; route={route}; planned {tool_name}."
        return f"CAPD status={status}; verdict={verdict}; route={route}."

    if snapshot.current_work:
        return f"CAPD processed: {snapshot.current_work}"
    return ""


def _route_pending_question_answer(runtime: TuiRuntime, prompt_value: str) -> TuiSubmissionResult | None:
    pending = runtime.question_store.list_pending_for_respondent(runtime.user)
    if not pending:
        return None
    result = runtime.question_store.route_answer(respondent_user_id=runtime.user, text=prompt_value)
    if result.ambiguous or result.invalid:
        return TuiSubmissionResult(
            prompt=prompt_value,
            response=result.message,
            event="question answer rejected",
        )
    if not result.handled or result.resumed_task is None:
        return None
    runtime.ui_events.append(
        CoreUiEvent(
            event_type="question_interrupt_resolved",
            payload={
                "question": result.question.to_dict() if result.question is not None else None,
                "answer": result.answer,
            },
        )
    )
    queued = runtime.channel.queue_message(
        prompt=prompt_value,
        user=runtime.user,
        project_id=result.resumed_task.project_id,
        correlation_id=result.resumed_task.correlation_id,
        metadata={"task_state": result.resumed_task.to_dict(), "answered_question_id": result.question.question_id if result.question else ""},
    )
    return TuiSubmissionResult(
        prompt=prompt_value,
        response="",
        event=f"answered question; queued {queued.message_id}",
        queued=True,
        queued_message_id=queued.message_id,
    )


def _render_question_interrupt(question: dict[str, Any]) -> str:
    message = str(question.get("message") or "").strip()
    kind = str(question.get("kind") or "open_text").strip()
    choices = question.get("choices") if isinstance(question.get("choices"), list) else []
    if kind == "yes_no":
        return f"{message}\n[yes/no]"
    if kind == "single_choice":
        rendered_choices = "\n".join(
            f"{index + 1}. {str(choice.get('label') or choice.get('id') or '').strip()}"
            for index, choice in enumerate(choices)
            if isinstance(choice, dict)
        )
        if rendered_choices:
            return f"{message}\n{rendered_choices}"
    return message


def _latest_tool_result_response(task_state: dict[str, Any]) -> str:
    plan = _json_list_or_empty(task_state.get("plan_json"))
    for call in reversed(plan):
        if not isinstance(call, dict):
            continue
        execution = call.get("execution")
        if not isinstance(execution, dict):
            continue
        status = str(execution.get("status") or "").strip()
        tool_id = str(call.get("tool_id") or "").strip()
        if status == "exception":
            exception = str(execution.get("exception") or "").strip()
            if exception:
                return f"{tool_id or 'tool'} failed: {exception}"
            return f"{tool_id or 'tool'} failed."
        if status != "success":
            continue
        result = execution.get("result")
        if not isinstance(result, dict):
            continue
        if tool_id == RESPOND_TOOL_ID:
            return str(result.get("message") or "").strip()
        if tool_id == BASH_TOOL_ID:
            stdout = str(result.get("stdout") or "").strip()
            stderr = str(result.get("stderr") or "").strip()
            if stdout:
                return stdout
            if stderr:
                return stderr
        if tool_id == SCHEDULED_TASK_TOOL_ID:
            name = str(result.get("name") or "Scheduled task").strip()
            next_run_at = str(result.get("next_run_at") or "").strip()
            if next_run_at:
                return f'Scheduled "{name}" for {next_run_at}.'
            return f'Scheduled "{name}".'
    return ""


def _json_list_or_empty(value: Any) -> list[Any]:
    if not isinstance(value, str) or not value.strip() or value.strip() == "- (none)":
        return []
    try:
        parsed = json.loads(value)
    except ValueError:
        return []
    return parsed if isinstance(parsed, list) else []


def format_activity_message(event: CoreActivityEvent) -> str:
    message = str(event.message or "").strip()
    if message:
        return f"{event.label} - {message}"
    return event.label


def format_activity_status_line(event: CoreActivityEvent) -> str:
    label = str(event.label or "").strip() or event.phase.value
    message = str(event.message or "").strip()
    prefix = label.capitalize()
    if message:
        return f"{prefix}: {message}"
    return prefix


def format_inference_status(runtime: TuiRuntime) -> str:
    inference = runtime.core.inference
    if inference is None:
        return "None"
    profile = inference.default_profile
    provider = str(profile.provider or "").strip() or "-"
    model = str(profile.model or "").strip() or "-"
    return f"{provider} / {model}"


def format_current_project_status(runtime: TuiRuntime) -> str:
    if not runtime.active_project_id:
        return "None"
    project = runtime.project_store.get_project(runtime.active_project_id, requester_user_id=runtime.user)
    if project is None:
        return "None"
    return project.name


def main() -> None:
    app_cls = _build_textual_app_class()
    app_cls().run()


def _build_textual_app_class() -> type[Any]:
    try:
        from textual.app import App, ComposeResult
        from textual.screen import ModalScreen
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Button, Footer, Header, Input, RichLog, Select, Static, TextArea
        from rich.text import Text
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Textual is required for the Alphonse v2 TUI. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

    class ProjectPickerScreen(ModalScreen[str | None]):
        def __init__(self, runtime: TuiRuntime) -> None:
            super().__init__()
            self.runtime = runtime

        def compose(self) -> ComposeResult:
            projects = self.runtime.project_store.list_visible_projects(self.runtime.user)
            options = [(f"{project.name} - {project.root_path}", project.project_id) for project in projects]
            options.append(("New Project", "__new__"))
            with Vertical(id="project-dialog"):
                yield Static("Project", classes="dialog-title")
                yield Select(options=options, id="project-select", allow_blank=False)
                with Horizontal(classes="dialog-actions"):
                    yield Button("Open", id="open-project", variant="primary")
                    yield Button("Cancel", id="cancel-project")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel-project":
                self.dismiss(None)
                return
            select = self.query_one("#project-select", Select)
            value = str(select.value or "")
            if value:
                self.dismiss(value)

    class NewProjectScreen(ModalScreen[ProjectRecord | None]):
        def __init__(self, runtime: TuiRuntime) -> None:
            super().__init__()
            self.runtime = runtime

        def on_mount(self) -> None:
            self.query_one("#project-name", Input).focus()

        def compose(self) -> ComposeResult:
            with Vertical(id="project-dialog"):
                yield Static("New Project", classes="dialog-title")
                yield Input(placeholder="Name", id="project-name")
                yield Input(placeholder="Description", id="project-description")
                yield Input(placeholder="Directory path", id="project-path")
                yield Select(options=[("Private", "private"), ("Shared", "shared")], id="project-visibility", allow_blank=False)
                yield Static("", id="project-error")
                with Horizontal(classes="dialog-actions"):
                    yield Button("Create", id="create-project", variant="primary")
                    yield Button("Cancel", id="cancel-new-project")

        def on_input_submitted(self, event: Input.Submitted) -> None:
            event.stop()
            if event.input.id == "project-name":
                self.query_one("#project-description", Input).focus()
                return
            if event.input.id == "project-description":
                self.query_one("#project-path", Input).focus()
                return
            if event.input.id == "project-path":
                self.query_one("#create-project", Button).focus()

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel-new-project":
                self.dismiss(None)
                return
            try:
                project = create_tui_project(
                    self.runtime,
                    name=self.query_one("#project-name", Input).value,
                    description=self.query_one("#project-description", Input).value,
                    root_path=self.query_one("#project-path", Input).value,
                    visibility=str(self.query_one("#project-visibility", Select).value or "private"),
                )
            except Exception as exc:
                self.query_one("#project-error", Static).update(str(exc))
                return
            self.dismiss(project)

    class ProjectContextScreen(ModalScreen[bool]):
        def __init__(self, runtime: TuiRuntime) -> None:
            super().__init__()
            self.runtime = runtime

        def on_mount(self) -> None:
            self.query_one("#project-context-editor", TextArea).focus()

        def compose(self) -> ComposeResult:
            project = self.runtime.project_store.get_project(self.runtime.active_project_id, requester_user_id=self.runtime.user)
            content = ""
            title = "Project Context"
            if project is not None:
                title = f"Project Context - {project.name}"
                content = self.runtime.project_store.read_project_context(project.project_id, requester_user_id=self.runtime.user)
            with Vertical(id="project-context-dialog"):
                yield Static(title, classes="dialog-title")
                yield TextArea(content, id="project-context-editor")
                yield Static("", id="project-context-error")
                with Horizontal(classes="dialog-actions"):
                    yield Button("Save", id="save-project-context", variant="primary")
                    yield Button("Cancel", id="cancel-project-context")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel-project-context":
                self.dismiss(False)
                return
            try:
                save_tui_project_context(
                    self.runtime,
                    self.query_one("#project-context-editor", TextArea).text,
                )
            except Exception as exc:
                self.query_one("#project-context-error", Static).update(str(exc))
                return
            self.dismiss(True)

    class IntegrationsScreen(ModalScreen[str | None]):
        def __init__(self, runtime: TuiRuntime) -> None:
            super().__init__()
            self.runtime = runtime

        def compose(self) -> ComposeResult:
            options = list_integration_options(self.runtime)
            with Vertical(id="integration-dialog"):
                yield Static("Integrations", classes="dialog-title")
                yield Select(options=options, id="integration-select", allow_blank=False)
                with Horizontal(classes="dialog-actions"):
                    yield Button("Configure", id="configure-integration", variant="primary")
                    yield Button("Cancel", id="cancel-integration")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel-integration":
                self.dismiss(None)
                return
            select = self.query_one("#integration-select", Select)
            value = str(select.value or "")
            if value:
                self.dismiss(value)

    class TelegramConfigScreen(ModalScreen[bool]):
        def __init__(self, runtime: TuiRuntime) -> None:
            super().__init__()
            self.runtime = runtime
            self.record = runtime.integration_store.get_by_provider("telegram")

        def on_mount(self) -> None:
            self.query_one("#telegram-integration-id", Input).focus()

        def compose(self) -> ComposeResult:
            descriptor = self.runtime.integration_registry.get("telegram")
            integration_id = (
                self.record.integration_id
                if self.record is not None
                else (descriptor.default_integration_id if descriptor is not None else "telegram-home")
            )
            display_name = self.record.display_name if self.record is not None else "Telegram"
            config = self.record.config if self.record is not None else {}
            has_token = self.record is not None and bool(str(self.record.secrets.get("bot_token") or "").strip())
            allowed = ", ".join(str(item) for item in config.get("allowed_chat_ids", []) if str(item).strip())
            telegram_user_id = str(config.get("telegram_user_id") or "").strip()
            if not telegram_user_id:
                try:
                    mapped_user_id = resolve_provider_user_mapping(
                        alphonse_user_id=self.runtime.user,
                        provider_key="telegram",
                    )
                except Exception:
                    mapped_user_id = None
                telegram_user_id = str(mapped_user_id or "").strip()
            enabled = "yes" if self.record is not None and self.record.enabled else "no"
            presence_enabled = "yes" if config.get("presence_enabled", True) else "no"
            with Vertical(id="integration-dialog"):
                yield Static("Telegram", classes="dialog-title")
                yield Input(value=integration_id, placeholder="Integration id", id="telegram-integration-id")
                yield Input(value=display_name, placeholder="Display name", id="telegram-display-name")
                yield Input(placeholder="Bot token" + (" (saved)" if has_token else ""), id="telegram-bot-token")
                yield Input(value=telegram_user_id, placeholder="Telegram user id for this Alphonse user", id="telegram-user-id")
                yield Input(value=str(config.get("poll_interval_sec") or "1.0"), placeholder="Poll interval seconds", id="telegram-poll-interval")
                yield Input(value=allowed, placeholder="Allowed chat ids, comma separated", id="telegram-allowed-chat-ids")
                yield Select(options=[("Enabled", "yes"), ("Disabled", "no")], value=enabled, id="telegram-enabled", allow_blank=False)
                yield Select(options=[("Presence enabled", "yes"), ("Presence disabled", "no")], value=presence_enabled, id="telegram-presence-enabled", allow_blank=False)
                yield Static("", id="telegram-config-error")
                with Horizontal(classes="dialog-actions"):
                    yield Button("Save", id="save-telegram", variant="primary")
                    yield Button("Disable", id="disable-telegram")
                    yield Button("Remove Token", id="remove-telegram-token")
                    yield Button("Cancel", id="cancel-telegram")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel-telegram":
                self.dismiss(False)
                return
            integration_id = self.query_one("#telegram-integration-id", Input).value
            if event.button.id == "disable-telegram":
                try:
                    self.runtime.integration_store.set_enabled(integration_id, False)
                    refresh_tui_identity_resolver(self.runtime)
                except Exception as exc:
                    self.query_one("#telegram-config-error", Static).update(str(exc))
                    return
                self.dismiss(True)
                return
            try:
                save_telegram_integration_config(
                    self.runtime,
                    integration_id=integration_id,
                    display_name=self.query_one("#telegram-display-name", Input).value,
                    bot_token=self.query_one("#telegram-bot-token", Input).value,
                    poll_interval_sec=self.query_one("#telegram-poll-interval", Input).value,
                    allowed_chat_ids=self.query_one("#telegram-allowed-chat-ids", Input).value,
                    telegram_user_id=self.query_one("#telegram-user-id", Input).value,
                    enabled=str(self.query_one("#telegram-enabled", Select).value or "no") == "yes",
                    remove_token=event.button.id == "remove-telegram-token",
                    presence_enabled=str(self.query_one("#telegram-presence-enabled", Select).value or "yes") == "yes",
                )
            except Exception as exc:
                self.query_one("#telegram-config-error", Static).update(str(exc))
                return
            self.dismiss(True)

    class AlphonseTuiApp(App[None]):
        CSS = """
        Screen {
            layout: vertical;
        }

        #main {
            height: 1fr;
        }

        #chat-column {
            width: 2fr;
        }

        #side-column {
            width: 1fr;
            min-width: 28;
        }

        #chat {
            border: solid $primary;
        }

        #activity {
            border: solid $primary;
            height: 5;
            padding: 1;
            color: $text-muted;
        }

        #status {
            border: solid $secondary;
            height: 8;
            padding: 1;
        }

        #slash-commands {
            height: auto;
            max-height: 7;
            padding: 0 1;
            color: $text-muted;
            display: none;
        }

        #project-dialog {
            width: 70;
            height: auto;
            padding: 1;
            border: solid $primary;
            background: $surface;
        }

        #project-context-dialog {
            width: 80%;
            height: 80%;
            padding: 1;
            border: solid $primary;
            background: $surface;
        }

        #integration-dialog {
            width: 80;
            height: auto;
            padding: 1;
            border: solid $primary;
            background: $surface;
        }

        .dialog-title {
            text-style: bold;
            margin-bottom: 1;
        }

        .dialog-actions {
            height: auto;
            margin-top: 1;
        }
        """

        BINDINGS = [("ctrl+c", "quit", "Quit")]

        def __init__(self) -> None:
            super().__init__()
            integration_store = SQLiteIntegrationStore.default()
            self.daemon_client = V2DaemonClient()
            self.external_daemon = _daemon_is_available(self.daemon_client)
            self.runtime = build_runtime_host(
                user="local",
                project_store=ProjectStore.default(),
                schedule_store=ScheduledTaskStore.default(),
                outbox=SQLiteOutboundStore.default(),
                integration_store=integration_store,
                messages=SQLiteMessageQueue.default(),
            )
            self.daemon = None if self.external_daemon else V2Daemon(self.runtime)
            self.runtime.presence_projector.register(
                "tui",
                TuiPresenceAdapter(self._emit_presence_status),
            )
            if self.daemon is not None:
                self.daemon.start()
            self.processor = TuiProcessorCoordinator(self.runtime)
            self.external_queue_size = -1
            self.last_message_id = ""
            self.slash_command_matches: list[tuple[str, str]] = []
            self.slash_command_selected_index = 0

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                with Vertical(id="chat-column"):
                    yield RichLog(id="chat", wrap=True, highlight=True)
                    yield Static(id="slash-commands")
                    yield Input(placeholder="Message Alphonse...", id="prompt")
                with Vertical(id="side-column"):
                    yield Static(id="activity")
                    yield Static(id="status")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "Alphonse v2"
            self.query_one("#chat", RichLog).write(Text.from_markup("[bold]Alphonse v2[/bold] native TUI"))
            self.query_one("#activity", Static).update("Idle: ready")
            self._refresh_status()
            self.query_one("#prompt", Input).focus()
            self.set_interval(0.1, self._poll_daemon)

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id != "prompt":
                return
            self._refresh_slash_commands(event.value)

        def on_key(self, event: Any) -> None:
            if not self.slash_command_matches:
                return
            if getattr(self.focused, "id", "") != "prompt":
                return
            if event.key == "down":
                event.stop()
                self.slash_command_selected_index = min(
                    self.slash_command_selected_index + 1,
                    len(self.slash_command_matches) - 1,
                )
                self._refresh_slash_commands(self.query_one("#prompt", Input).value, reset_selection=False)
                return
            if event.key == "up":
                event.stop()
                self.slash_command_selected_index = max(self.slash_command_selected_index - 1, 0)
                self._refresh_slash_commands(self.query_one("#prompt", Input).value, reset_selection=False)
                return
            if event.key == "enter":
                event.stop()
                command = self.slash_command_matches[self.slash_command_selected_index][0]
                prompt = self.query_one("#prompt", Input)
                prompt.value = command
                self._refresh_slash_commands("")
                self._handle_prompt_submission(command)

        def on_input_submitted(self, event: Input.Submitted) -> None:
            prompt = event.value
            event.input.value = ""
            self._handle_prompt_submission(prompt)

        def _handle_prompt_submission(self, prompt: str) -> None:
            self._refresh_slash_commands("")
            command = detect_tui_slash_command(prompt)
            if command == "project":
                self.query_one("#chat", RichLog).write(Text.assemble((self.runtime.user, "bold cyan"), f": {prompt.strip()}"))
                self._open_project_picker()
                return
            if command == "project-context":
                self.query_one("#chat", RichLog).write(Text.assemble((self.runtime.user, "bold cyan"), f": {prompt.strip()}"))
                self._open_project_context_flow()
                return
            if command == "integrations":
                self.query_one("#chat", RichLog).write(Text.assemble((self.runtime.user, "bold cyan"), f": {prompt.strip()}"))
                self._open_integrations()
                return

            if str(prompt or "").strip() in LOCAL_STOP_COMMANDS:
                self.exit()
                return

            if self.external_daemon:
                prompt_value = str(prompt or "").strip()
                if not prompt_value:
                    return
                try:
                    queued = self.daemon_client.queue_message(
                        prompt=prompt_value,
                        user=self.runtime.user,
                        integration_id="tui",
                        provider_key="tui",
                        channel_target=self.runtime.user,
                    )
                except Exception as exc:
                    self.query_one("#activity", Static).update(f"Daemon unavailable: {exc}")
                    return
                submitted_prompt = prompt_value
                queued_message_id = str(queued.get("message_id") or "")
            else:
                result = queue_tui_input(self.runtime, prompt)
                if not result.prompt:
                    return
                submitted_prompt = result.prompt
                queued_message_id = result.queued_message_id
            chat = self.query_one("#chat", RichLog)
            chat.write(Text.assemble((self.runtime.user, "bold cyan"), f": {submitted_prompt}"))
            if queued_message_id:
                self.last_message_id = queued_message_id
            self._refresh_status()
            self._start_processor_if_needed()

        def _open_project_picker(self, *, then_context: bool = False) -> None:
            def _selected(value: str | None) -> None:
                if not value:
                    return
                if value == "__new__":
                    self.push_screen(NewProjectScreen(self.runtime), callback=_created)
                    return
                select_tui_project(self.runtime, value)
                self._refresh_status()
                if then_context:
                    self.push_screen(ProjectContextScreen(self.runtime), callback=lambda _: self._refresh_status())

            def _created(project: ProjectRecord | None) -> None:
                if project is None:
                    return
                self._refresh_status()
                if then_context:
                    self.push_screen(ProjectContextScreen(self.runtime), callback=lambda _: self._refresh_status())

            self.push_screen(ProjectPickerScreen(self.runtime), callback=_selected)

        def _open_project_context_flow(self) -> None:
            if not self.runtime.active_project_id:
                self._open_project_picker(then_context=True)
                return
            self.push_screen(ProjectContextScreen(self.runtime), callback=lambda _: self._refresh_status())

        def _open_integrations(self) -> None:
            def _selected(provider_key: str | None) -> None:
                if provider_key == "telegram":
                    self.push_screen(TelegramConfigScreen(self.runtime), callback=_updated)

            def _updated(updated: bool) -> None:
                if updated:
                    if self.external_daemon:
                        self.daemon_client.restart_integrations()
                    elif self.daemon is not None:
                        self.daemon.restart_integrations()
                    self._refresh_status()

            self.push_screen(IntegrationsScreen(self.runtime), callback=_selected)

        def on_unmount(self) -> None:
            if self.daemon is not None:
                self.daemon.stop()

        def _start_processor_if_needed(self) -> None:
            self._refresh_status()

        def _process_queue_worker(self) -> None:
            return

        def _emit_activity_from_worker(self, event: CoreActivityEvent) -> None:
            self.runtime.presence_projector.on_activity(event)
            self.call_from_thread(self._append_activity_event, event)

        def _emit_presence_status(self, status: str) -> None:
            self.call_from_thread(self._set_presence_status, status)

        def _set_presence_status(self, status: str) -> None:
            self.query_one("#activity", Static).update(status)

        def _wake_processor_from_integration(self) -> None:
            self.call_from_thread(self._refresh_status)

        def _poll_daemon(self) -> None:
            if self.external_daemon:
                try:
                    events = self.daemon_client.events()
                    for event in events:
                        self._append_activity_event(
                            CoreActivityEvent(
                                phase=ImprovementPhase(str(event.get("phase") or "act")),
                                label=str(event.get("label") or ""),
                                message=str(event.get("message") or ""),
                                speaker=str(event.get("speaker") or "Alphonse"),
                            )
                        )
                    status = self.daemon_client.status()
                    self.external_queue_size = int(status.get("queue_size") or 0)
                except Exception:
                    self.external_queue_size = -1
            else:
                while self.runtime.activity_events:
                    event = self.runtime.activity_events.pop(0)
                    self._append_activity_event(event)
            selector = OutboundSelector(integration_id="tui", channel_target=self.runtime.user, status="pending")
            chat = self.query_one("#chat", RichLog)
            while True:
                outbound = self.runtime.outbox.claim_next(selector)
                if outbound is None:
                    break
                if outbound.message:
                    chat.write(Text.assemble(("Alphonse", "bold green"), f": {outbound.message}"))
                self.runtime.outbox.mark_delivered(outbound.outbox_message_id)
            self._refresh_status()

        def _append_activity_event(self, event: CoreActivityEvent) -> None:
            self.query_one("#activity", Static).update(format_activity_status_line(event))

        def _apply_processing_results(self, results: list[TuiProcessingResult]) -> None:
            chat = self.query_one("#chat", RichLog)
            for result in results:
                if result.queued_message_id:
                    self.last_message_id = result.queued_message_id
                if result.response:
                    chat.write(Text.assemble(("Alphonse", "bold green"), f": {result.response}"))
            self._refresh_status()
            if self.runtime.queue.size(MessageSelector(user=self.runtime.user)) > 0:
                self._start_processor_if_needed()

        def _refresh_slash_commands(self, prompt: str, *, reset_selection: bool = True) -> None:
            suggestions = self.query_one("#slash-commands", Static)
            matches = matching_slash_commands(prompt)
            if reset_selection or not matches:
                self.slash_command_selected_index = 0
            self.slash_command_matches = matches
            if matches:
                self.slash_command_selected_index = min(self.slash_command_selected_index, len(matches) - 1)
            rendered = format_slash_command_suggestions(prompt, selected_index=self.slash_command_selected_index)
            if rendered:
                suggestions.update(rendered)
                suggestions.display = True
                return
            suggestions.update("")
            suggestions.display = False

        def _refresh_status(self) -> None:
            state = get_state()
            text = "\n".join(
                [
                    f"state: {state.key}",
                    f"processing: {self.processor.is_processing if not self.external_daemon else 'daemon'}",
                    f"user: {self.runtime.user}",
                    f"current project: {format_current_project_status(self.runtime)}",
                    f"inference: {format_inference_status(self.runtime)}",
                    f"queue: {self.external_queue_size if self.external_daemon else self.runtime.queue.size()}",
                    f"last message: {self.last_message_id or '-'}",
                ]
            )
            self.query_one("#status", Static).update(text)

    return AlphonseTuiApp


def _daemon_is_available(client: V2DaemonClient) -> bool:
    try:
        client.ping()
    except Exception:
        return False
    return True
