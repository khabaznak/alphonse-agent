"""Textual native interface for Alphonse v2."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from threading import Lock, Thread
from typing import Any, Callable

from alphonse.agent_v2.core.core import AlphonseCore
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.core import CoreActivityEvent
from alphonse.agent_v2.core.core import CoreUiEvent
from alphonse.agent_v2.core.core import IntelligenceProcessor
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.core import LoopStepResult
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.core import MemoryRecord
from alphonse.agent_v2.core.core import PromptFile
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.core import ToolRegistry
from alphonse.agent_v2.core.inference import InferenceRouter
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
from alphonse.agent_v2.inference_settings import CODEX_DEFAULT_MODEL
from alphonse.agent_v2.inference_settings import SQLiteInferenceSettingsStore
from alphonse.agent_v2.inference_settings import inference_provider_descriptors
from alphonse.agent_v2.inference_settings import provider_status
from alphonse.agent_v2.inference_settings import validate_and_save_inference_settings
from alphonse.agent_v2.agent_config import AgentConfigStore
from alphonse.agent_v2.runtime import InMemoryInternalState
from alphonse.agent_v2.runtime import NullMemory
from alphonse.agent_v2.runtime import NullPromptLoader
from alphonse.agent_v2.runtime import V2RuntimeHost
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.runtime import build_default_runtime_inference_router
from alphonse.agent_v2.runtime import refresh_runtime_inference
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
    ("/model-provider", "Select an inference provider"),
    ("/model", "Select and validate an inference model"),
    ("/agent-config", "Edit global agent configuration"),
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
    inference_settings_store: SQLiteInferenceSettingsStore | None = None,
    agent_config_store: AgentConfigStore | None = None,
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
        inference_settings_store=inference_settings_store,
        agent_config_store=agent_config_store,
    )


def build_default_tui_inference_router() -> InferenceRouter:
    """Build the default live inference router for the native TUI."""
    return build_default_runtime_inference_router()


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
    if command in {"project", "project-context", "integrations", "model-provider", "model", "agent-config"}:
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
    elif step.status == LoopStepStatus.FAILED:
        runtime.core.clear_failure()
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


def activity_spinner(index: int) -> str:
    return ("|", "/", "-", "\\")[max(0, int(index)) % 4]


def capd_activity_verb(event: CoreActivityEvent) -> str:
    return {
        ImprovementPhase.PLAN: "Planning",
        ImprovementPhase.DO: "Doing",
        ImprovementPhase.CHECK: "Checking",
        ImprovementPhase.ACT: "Acting",
    }.get(event.phase, "Working")


def format_global_activity_status(event: CoreActivityEvent, spinner_index: int = 0) -> str:
    """Show safe CAPD progress without exposing another user's task details."""
    return f"Working: {capd_activity_verb(event)} {activity_spinner(spinner_index)}"


def format_global_activity_state(activity: dict[str, Any] | None, *, error: str = "") -> str:
    state = str(activity.get("state") or "idle").strip().lower() if isinstance(activity, dict) else "idle"
    if state == "working":
        return "Working"
    if state == "waiting":
        return "Waiting: user input"
    if state == "error":
        return "Error: model token limit" if "token" in str(error).lower() else "Error: attention required"
    return "Idle: ready"


def format_inference_status(runtime: TuiRuntime) -> str:
    inference = runtime.core.inference
    if inference is None:
        return "None"
    profile = inference.default_profile
    provider = str(profile.provider or "").strip() or "-"
    model = str(profile.model or "").strip() or "-"
    return f"{provider} / {model}"


def format_inference_settings_status(settings: dict[str, Any] | None) -> str:
    provider, model = inference_status_parts(settings)
    return f"{provider} / {model}"


def inference_status_parts(
    settings: dict[str, Any] | None,
    *,
    runtime: TuiRuntime | None = None,
) -> tuple[str, str]:
    if isinstance(settings, dict):
        provider = str(settings.get("provider_key") or "").strip() or "-"
        model = str(settings.get("model_id") or "").strip() or "default"
        return provider, model
    if runtime is not None and runtime.core.inference is not None:
        profile = runtime.core.inference.default_profile
        return str(profile.provider or "").strip() or "-", str(profile.model or "").strip() or "default"
    return "-", "-"


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
        from textual.widgets import Button, Footer, Header, Input, OptionList, RichLog, Select, Static, TextArea
        from textual.widgets.option_list import Option
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

    class ModelProviderScreen(ModalScreen[str | None]):
        def __init__(self, providers: list[dict[str, Any]], selected_provider: str) -> None:
            super().__init__()
            self.providers = providers
            self.selected_provider = selected_provider

        def compose(self) -> ComposeResult:
            options = [
                (str(provider.get("display_name") or provider.get("provider_key") or "Provider"), str(provider.get("provider_key") or ""))
                for provider in self.providers
                if str(provider.get("provider_key") or "").strip()
            ]
            selected = self.selected_provider if any(value == self.selected_provider for _, value in options) else None
            with Vertical(id="model-dialog"):
                yield Static("Model Provider", classes="dialog-title")
                yield Select(options=options, value=selected, id="model-provider-select", allow_blank=False)
                yield Static("", id="model-provider-details")
                with Horizontal(classes="dialog-actions"):
                    yield Button("Continue", id="select-model-provider", variant="primary")
                    yield Button("Cancel", id="cancel-model-provider")

        def on_mount(self) -> None:
            self._render_details()

        def on_select_changed(self, event: Select.Changed) -> None:
            if event.select.id == "model-provider-select":
                self._render_details()

        def _render_details(self) -> None:
            selected = str(self.query_one("#model-provider-select", Select).value or "")
            provider = next((item for item in self.providers if item.get("provider_key") == selected), {})
            models = provider.get("models")
            count = len(models) if isinstance(models, list) else 0
            fetched_at = str(provider.get("catalog_fetched_at") or "")
            cli_version = str(provider.get("cli_version") or "")
            detail = str(provider.get("description") or "")
            detail += f"\n{count} selectable models"
            if fetched_at:
                detail += f" | catalog: {fetched_at}"
            if cli_version:
                detail += f" | CLI: {cli_version}"
            self.query_one("#model-provider-details", Static).update(detail)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel-model-provider":
                self.dismiss(None)
                return
            self.dismiss(str(self.query_one("#model-provider-select", Select).value or "") or None)

    class ModelSelectionScreen(ModalScreen[dict[str, Any] | None]):
        def __init__(self, settings: dict[str, Any], provider: dict[str, Any], save: Callable[[str, str], dict[str, Any]]) -> None:
            super().__init__()
            self.settings = settings
            self.provider = provider
            self.save = save

        def compose(self) -> ComposeResult:
            models = self.provider.get("models") if isinstance(self.provider.get("models"), list) else []
            options = [
                (str(item.get("display_name") or item.get("model_id") or "Model"), str(item.get("model_id") or ""))
                for item in models
                if isinstance(item, dict) and str(item.get("model_id") or "").strip()
            ]
            current = str(self.settings.get("model_id") or "") or CODEX_DEFAULT_MODEL
            if not any(value == current for _, value in options):
                options.insert(0, ("Codex default", CODEX_DEFAULT_MODEL))
                current = CODEX_DEFAULT_MODEL
            with Vertical(id="model-dialog"):
                yield Static("Inference Model", classes="dialog-title")
                yield Select(options=options, value=current, id="model-select", allow_blank=False)
                yield Static(self._details_for(current), id="model-details")
                yield Static("", id="model-error")
                with Horizontal(classes="dialog-actions"):
                    yield Button("Validate & Save", id="save-model", variant="primary")
                    yield Button("Cancel", id="cancel-model")

        def on_select_changed(self, event: Select.Changed) -> None:
            if event.select.id == "model-select":
                self.query_one("#model-details", Static).update(self._details_for(str(event.value or "")))

        def _details_for(self, model_id: str) -> str:
            for item in self.provider.get("models", []):
                if isinstance(item, dict) and str(item.get("model_id") or "") == model_id:
                    return str(item.get("description") or "")
            return "The installed Codex CLI will validate this choice before it becomes active."

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel-model":
                self.dismiss(None)
                return
            self.query_one("#save-model", Button).disabled = True
            self.query_one("#model-error", Static).update("Validating model with Codex...")
            provider_key = str(self.provider.get("provider_key") or "")
            model_id = str(self.query_one("#model-select", Select).value or "")
            Thread(target=self._save_in_background, args=(provider_key, model_id), daemon=True).start()

        def _save_in_background(self, provider_key: str, model_id: str) -> None:
            try:
                settings = self.save(provider_key, model_id)
            except Exception as exc:
                self.app.call_from_thread(self._save_failed, str(exc))
                return
            self.app.call_from_thread(self.dismiss, settings)

        def _save_failed(self, error: str) -> None:
            self.query_one("#save-model", Button).disabled = False
            self.query_one("#model-error", Static).update(error)

    class SlashCommandScreen(ModalScreen[str | None]):
        def compose(self) -> ComposeResult:
            with Vertical(id="slash-command-dialog"):
                yield Static("Commands", classes="dialog-title")
                yield Input(value="/", id="slash-command-filter")
                yield OptionList(id="slash-command-options")

        def on_mount(self) -> None:
            self._refresh_options()
            self.query_one("#slash-command-filter", Input).focus()

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id == "slash-command-filter":
                self._refresh_options()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            if event.input.id == "slash-command-filter":
                self._choose_highlighted()

        def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
            self.dismiss(str(event.option.id or "") or None)

        def on_key(self, event: Any) -> None:
            if event.key == "escape":
                event.stop()
                self.dismiss(None)
                return
            if event.key not in {"up", "down"}:
                return
            event.stop()
            options = self.query_one("#slash-command-options", OptionList)
            if options.option_count <= 0:
                return
            current = options.highlighted if options.highlighted is not None else 0
            options.highlighted = max(0, min(options.option_count - 1, current + (1 if event.key == "down" else -1)))
            options.scroll_to_highlight()

        def _refresh_options(self) -> None:
            typed = self.query_one("#slash-command-filter", Input).value
            options = self.query_one("#slash-command-options", OptionList)
            options.clear_options()
            options.add_options(
                Option(f"{command} - {description}", id=command)
                for command, description in matching_slash_commands(typed)
            )
            if options.option_count:
                options.highlighted = 0

        def _choose_highlighted(self) -> None:
            options = self.query_one("#slash-command-options", OptionList)
            if options.highlighted is None:
                return
            self.dismiss(str(options.get_option_at_index(options.highlighted).id or "") or None)

    class AgentConfigScreen(ModalScreen[str | None]):
        def __init__(self, documents: list[dict[str, Any]]) -> None:
            super().__init__()
            self.documents = documents

        def compose(self) -> ComposeResult:
            options = [
                (str(document.get("display_name") or document.get("file_name") or "Document"), str(document.get("file_name") or ""))
                for document in self.documents
                if str(document.get("file_name") or "").strip()
            ]
            with Vertical(id="agent-config-dialog"):
                yield Static("Agent Configuration", classes="dialog-title")
                yield Select(options=options, id="agent-config-select", allow_blank=False)
                with Horizontal(classes="dialog-actions"):
                    yield Button("Edit", id="edit-agent-config", variant="primary")
                    yield Button("Cancel", id="cancel-agent-config")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel-agent-config":
                self.dismiss(None)
                return
            self.dismiss(str(self.query_one("#agent-config-select", Select).value or "") or None)

    class AgentConfigEditorScreen(ModalScreen[bool]):
        def __init__(self, document: dict[str, Any], save: Callable[[str, str], dict[str, Any]]) -> None:
            super().__init__()
            self.document = document
            self.save = save

        def on_mount(self) -> None:
            self.query_one("#agent-config-editor", TextArea).focus()
            self.query_one("#close-agent-config-editor", Button).display = False

        def compose(self) -> ComposeResult:
            with Vertical(id="agent-config-editor-dialog"):
                yield Static(str(self.document.get("display_name") or "Agent Configuration"), classes="dialog-title")
                yield TextArea(str(self.document.get("content") or ""), id="agent-config-editor")
                yield Static("", id="agent-config-notice")
                with Horizontal(classes="dialog-actions", id="agent-config-actions"):
                    yield Button("Save", id="save-agent-config", variant="primary")
                    yield Button("Cancel", id="cancel-agent-config-editor")
                yield Button("Close", id="close-agent-config-editor")

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "cancel-agent-config-editor":
                self.dismiss(False)
                return
            if event.button.id == "close-agent-config-editor":
                self.dismiss(True)
                return
            try:
                self.save(
                    str(self.document.get("file_name") or ""),
                    self.query_one("#agent-config-editor", TextArea).text,
                )
            except Exception as exc:
                self.query_one("#agent-config-notice", Static).update(str(exc))
                return
            self.query_one("#agent-config-actions", Horizontal).display = False
            self.query_one("#close-agent-config-editor", Button).display = True
            self.query_one("#agent-config-notice", Static).update(
                "Saved. Restart the daemon for changes to take effect: alphonse stop, then alphonse start."
            )

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
            height: 13;
            padding: 1;
        }

        #slash-command-dialog {
            width: 80%;
            max-width: 80;
            height: 18;
            max-height: 80%;
            padding: 1;
            border: solid $primary;
            background: $surface;
        }

        #slash-command-options {
            height: 1fr;
        }

        #reply-activity {
            height: 1;
            padding: 0 1;
            color: $text-muted;
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

        #model-dialog {
            width: 80;
            height: auto;
            padding: 1;
            border: solid $primary;
            background: $surface;
        }

        #agent-config-dialog {
            width: 70;
            height: auto;
            padding: 1;
            border: solid $primary;
            background: $surface;
        }

        #agent-config-editor-dialog {
            width: 85%;
            height: 85%;
            padding: 1;
            border: solid $primary;
            background: $surface;
        }

        #agent-config-editor {
            height: 1fr;
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
            inference_settings_store = SQLiteInferenceSettingsStore.default()
            agent_config_store = AgentConfigStore.default()
            self.daemon_client = V2DaemonClient()
            self.external_daemon = _daemon_is_available(self.daemon_client)
            self.daemon_connection_status = "connected" if self.external_daemon else "embedded"
            self.runtime = build_runtime_host(
                user="local",
                project_store=ProjectStore.default(),
                schedule_store=ScheduledTaskStore.default(),
                outbox=SQLiteOutboundStore.default(),
                integration_store=integration_store,
                inference_settings_store=inference_settings_store,
                agent_config_store=agent_config_store,
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
            self.external_queue_counts: dict[str, int] = {}
            self.external_active_work: dict[str, Any] = {}
            self.external_inference_settings: dict[str, Any] = {}
            self.external_activity: dict[str, Any] = {"state": "idle"}
            self.external_processor_error = ""
            self.last_message_id = ""
            self.reply_message_id = ""
            self.reply_activity_verb = ""
            self.activity_hold_until = 0.0
            self.spinner_index = 0
            self.slash_command_matches: list[tuple[str, str]] = []
            self.slash_command_selected_index = 0

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                with Vertical(id="chat-column"):
                    yield RichLog(id="chat", wrap=True, highlight=True)
                    yield Static(id="reply-activity")
                    yield Input(placeholder="Message Alphonse...", id="prompt")
                with Vertical(id="side-column"):
                    yield Static(id="activity")
                    yield Static(id="status")
            yield Footer()

        def on_mount(self) -> None:
            self.title = "Alphonse v2"
            self.query_one("#chat", RichLog).write(Text.from_markup("[bold]Alphonse v2[/bold] native TUI"))
            self.query_one("#activity", Static).update("Idle: ready")
            self.query_one("#reply-activity", Static).update("")
            self._refresh_status()
            self.query_one("#prompt", Input).focus()
            self.set_interval(0.1, self._poll_daemon)
            self.set_interval(0.2, self._advance_activity_indicators)

        def on_input_changed(self, event: Input.Changed) -> None:
            if event.input.id != "prompt":
                return
            if event.value == "/":
                event.input.value = ""
                self._open_slash_command_palette()

        def on_key(self, event: Any) -> None:
            _ = event

        def on_input_submitted(self, event: Input.Submitted) -> None:
            prompt = event.value
            event.input.value = ""
            self._handle_prompt_submission(prompt)

        def _handle_prompt_submission(self, prompt: str) -> None:
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
            if command == "model-provider":
                self.query_one("#chat", RichLog).write(Text.assemble((self.runtime.user, "bold cyan"), f": {prompt.strip()}"))
                self._open_model_provider()
                return
            if command == "model":
                self.query_one("#chat", RichLog).write(Text.assemble((self.runtime.user, "bold cyan"), f": {prompt.strip()}"))
                self._open_model()
                return
            if command == "agent-config":
                self.query_one("#chat", RichLog).write(Text.assemble((self.runtime.user, "bold cyan"), f": {prompt.strip()}"))
                self._open_agent_config()
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
                self._begin_tui_reply_activity(queued_message_id)
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

        def _inference_settings(self) -> dict[str, Any]:
            if self.external_daemon:
                return dict(self.daemon_client.inference_settings().get("settings") or {})
            return self.runtime.inference_settings_store.get().to_dict()

        def _inference_providers(self) -> list[dict[str, Any]]:
            if self.external_daemon:
                payload = self.daemon_client.inference_providers().get("providers")
                return [dict(item) for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
            return [provider_status(item.provider_key) for item in inference_provider_descriptors()]

        def _inference_provider(self, provider_key: str) -> dict[str, Any]:
            if self.external_daemon:
                return dict(self.daemon_client.inference_models(provider_key))
            return provider_status(provider_key)

        def _save_inference_settings(self, provider_key: str, model_id: str) -> dict[str, Any]:
            if self.external_daemon:
                return dict(self.daemon_client.set_inference_settings(provider_key=provider_key, model_id=model_id).get("settings") or {})
            settings = validate_and_save_inference_settings(
                self.runtime.inference_settings_store,
                provider_key=provider_key,
                model_id=model_id,
            )
            refresh_runtime_inference(self.runtime, settings)
            return settings.to_dict()

        def _agent_config_documents(self) -> list[dict[str, Any]]:
            if self.external_daemon:
                payload = self.daemon_client.agent_config_documents().get("documents")
                return [dict(item) for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []
            return [document.to_dict(include_content=False) for document in self.runtime.agent_config_store.list_documents()]

        def _read_agent_config(self, file_name: str) -> dict[str, Any]:
            if self.external_daemon:
                return dict(self.daemon_client.read_agent_config(file_name).get("document") or {})
            return self.runtime.agent_config_store.read(file_name).to_dict()

        def _save_agent_config(self, file_name: str, content: str) -> dict[str, Any]:
            if self.external_daemon:
                return dict(self.daemon_client.save_agent_config(file_name=file_name, content=content).get("document") or {})
            return self.runtime.agent_config_store.save(file_name, content).to_dict()

        def _open_agent_config(self) -> None:
            try:
                documents = self._agent_config_documents()
            except Exception as exc:
                self.query_one("#activity", Static).update(f"Agent configuration unavailable: {exc}")
                return

            def _selected(file_name: str | None) -> None:
                if not file_name:
                    return
                try:
                    document = self._read_agent_config(file_name)
                except Exception as exc:
                    self.query_one("#activity", Static).update(f"Agent configuration unavailable: {exc}")
                    return
                self.push_screen(AgentConfigEditorScreen(document, self._save_agent_config))

            self.push_screen(AgentConfigScreen(documents), callback=_selected)

        def _open_model_provider(self) -> None:
            try:
                settings = self._inference_settings()
                providers = self._inference_providers()
            except Exception as exc:
                self.query_one("#activity", Static).update(f"Model settings unavailable: {exc}")
                return

            def _selected(provider_key: str | None) -> None:
                if provider_key:
                    self._open_model(provider_key=provider_key)

            self.push_screen(ModelProviderScreen(providers, str(settings.get("provider_key") or "")), callback=_selected)

        def _open_model(self, *, provider_key: str | None = None) -> None:
            try:
                settings = self._inference_settings()
                provider = self._inference_provider(provider_key or str(settings.get("provider_key") or "openai_codex"))
            except Exception as exc:
                self.query_one("#activity", Static).update(f"Model settings unavailable: {exc}")
                return
            self.push_screen(
                ModelSelectionScreen(settings, provider, self._save_inference_settings),
                callback=self._on_model_saved,
            )

        def _on_model_saved(self, settings: dict[str, Any] | None) -> None:
            if isinstance(settings, dict) and self.external_daemon:
                self.external_inference_settings = dict(settings)
            self._refresh_status()

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
                                task_id=str(event.get("task_id") or ""),
                                message_id=str(event.get("message_id") or ""),
                                user=str(event.get("user") or ""),
                                integration_id=str(event.get("integration_id") or ""),
                                channel_target=str(event.get("channel_target") or ""),
                            )
                        )
                    status = self.daemon_client.status()
                    self.external_queue_size = int(status.get("queue_size") or 0)
                    counts = status.get("inbound_counts")
                    self.external_queue_counts = dict(counts) if isinstance(counts, dict) else {}
                    active_work = status.get("active_work")
                    self.external_active_work = dict(active_work) if isinstance(active_work, dict) else {}
                    activity = status.get("activity")
                    self.external_activity = dict(activity) if isinstance(activity, dict) else {"state": "idle"}
                    self.external_processor_error = str(status.get("last_processor_error") or "")
                    settings = self.daemon_client.inference_settings().get("settings")
                    self.external_inference_settings = dict(settings) if isinstance(settings, dict) else {}
                    self.daemon_connection_status = "connected"
                except Exception:
                    self.external_queue_size = -1
                    self.external_queue_counts = {}
                    self.external_active_work = {}
                    self.external_inference_settings = {}
                    self.external_activity = {"state": "error"}
                    self.external_processor_error = "daemon connection lost"
                    self.daemon_connection_status = "disconnected"
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
                    self._clear_tui_reply_activity()
                self.runtime.outbox.mark_delivered(outbound.outbox_message_id)
            self._refresh_global_activity()
            self._refresh_status()

        def _append_activity_event(self, event: CoreActivityEvent) -> None:
            self.activity_hold_until = time.monotonic() + 0.8
            self.query_one("#activity", Static).update(format_global_activity_status(event, self.spinner_index))
            if self._is_tui_reply_event(event):
                self.reply_activity_verb = capd_activity_verb(event)
                self._render_tui_reply_activity()

        def _begin_tui_reply_activity(self, message_id: str) -> None:
            self.reply_message_id = str(message_id or "")
            self.reply_activity_verb = "Planning"
            self._render_tui_reply_activity()

        def _clear_tui_reply_activity(self) -> None:
            self.reply_message_id = ""
            self.reply_activity_verb = ""
            self.query_one("#reply-activity", Static).update("")

        def _is_tui_reply_event(self, event: CoreActivityEvent) -> bool:
            return bool(
                self.reply_message_id
                and event.message_id == self.reply_message_id
                and event.integration_id == "tui"
                and event.channel_target == self.runtime.user
            )

        def _render_tui_reply_activity(self) -> None:
            if self.reply_message_id:
                self.query_one("#reply-activity", Static).update(
                    f"Alphonse: {self.reply_activity_verb} {activity_spinner(self.spinner_index)}"
                )

        def _advance_activity_indicators(self) -> None:
            self.spinner_index += 1
            self._render_tui_reply_activity()
            self._refresh_global_activity()

        def _refresh_global_activity(self) -> None:
            active_work = self.external_active_work if self.external_daemon else self.daemon.active_work() if self.daemon else {}
            activity = self.external_activity if self.external_daemon else self.daemon.activity_status() if self.daemon else {"state": "idle"}
            if active_work:
                self.query_one("#activity", Static).update(f"Working {activity_spinner(self.spinner_index)}")
                return
            if time.monotonic() < self.activity_hold_until:
                return
            self.query_one("#activity", Static).update(
                format_global_activity_state(activity, error=self.external_processor_error)
            )

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

        def _open_slash_command_palette(self) -> None:
            self.push_screen(
                SlashCommandScreen(),
                callback=lambda command: self._handle_prompt_submission(command) if command else None,
            )

        def _refresh_status(self) -> None:
            state = get_state()
            active_work = self.external_active_work if self.external_daemon else self.daemon.active_work() if self.daemon else {}
            active_text = format_daemon_active_work(active_work)
            queue_text = (
                f"{self.external_queue_size} ready, {self.external_queue_counts.get('processing', 0)} processing"
                if self.external_daemon
                else str(self.runtime.queue.size())
            )
            inference_provider, inference_model = inference_status_parts(
                self.external_inference_settings if self.external_daemon else None,
                runtime=self.runtime if not self.external_daemon else None,
            )
            text = "\n".join(
                [
                    f"daemon: {format_daemon_connection_status(self.daemon_connection_status)}",
                    f"model provider: {inference_provider}",
                    f"model: {inference_model}",
                    f"state: {state.key}",
                    f"processing: {self.processor.is_processing if not self.external_daemon else 'daemon'}",
                    f"user: {self.runtime.user}",
                    f"current project: {format_current_project_status(self.runtime)}",
                    f"queue: {queue_text}",
                    f"active work: {active_text or '-'}",
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


def format_daemon_connection_status(status: str) -> str:
    """Render the TUI's current relationship with the daemon host."""
    normalized = str(status or "").strip().lower()
    if normalized == "embedded":
        return "connected (embedded)"
    if normalized == "connected":
        return "connected"
    return "disconnected"


def format_daemon_active_work(active_work: dict[str, Any] | None) -> str:
    if not isinstance(active_work, dict):
        return ""
    prompt = " ".join(str(active_work.get("prompt") or "").split())
    if not prompt:
        return ""
    user = str(active_work.get("user") or "unknown")
    return f"{user}: {prompt[:96]}"
