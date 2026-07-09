"""Textual native interface for Alphonse v2."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

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
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages import MessageSelector
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

LOCAL_STOP_COMMANDS = {"/exit", "/quit", "/stop"}
TUI_SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/project", "Select or create a project"),
    ("/project-context", "Edit the active project context"),
    ("/stop", "Stop Alphonse"),
    ("/exit", "Exit the TUI"),
    ("/quit", "Exit the TUI"),
)


@dataclass
class TuiRuntime:
    user: str
    queue: InMemoryMessageQueue
    channel: CommunicationChannel
    visible_state: "InMemoryInternalState"
    processor: IntelligenceProcessor
    core: AlphonseCore
    question_store: SQLiteQuestionStore
    project_store: ProjectStore
    schedule_store: ScheduledTaskStore
    active_project_id: str = ""
    ui_events: list[CoreUiEvent] = field(default_factory=list)


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


@dataclass
class InMemoryInternalState:
    value: StateSnapshot = field(default_factory=StateSnapshot)

    def update(self, snapshot: StateSnapshot) -> None:
        self.value = snapshot

    def snapshot(self) -> StateSnapshot:
        return self.value


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


class NullPromptLoader:
    def load(self, name: str) -> PromptFile:
        return PromptFile(name=name, content="")


class NullMemory:
    def write(self, record: MemoryRecord) -> None:
        _ = record

    def read(self, path: str) -> MemoryRecord | None:
        _ = path
        return None


def build_tui_runtime(
    *,
    user: str = "local",
    inference: InferenceRouter | None = None,
    tools: ToolRegistry | None = None,
    processor: IntelligenceProcessor | None = None,
    question_store: SQLiteQuestionStore | None = None,
    project_store: ProjectStore | None = None,
    schedule_store: ScheduledTaskStore | None = None,
) -> TuiRuntime:
    reset_state()
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    visible_state = InMemoryInternalState()
    processor = processor or PDCAIntelligenceProcessor()
    tools = tools or build_native_tool_registry()
    inference = inference or build_default_tui_inference_router()
    question_store = question_store or SQLiteQuestionStore()
    project_store = project_store or ProjectStore()
    schedule_store = schedule_store or ScheduledTaskStore()
    ui_events: list[CoreUiEvent] = []
    core = AlphonseCore(
        intelligence=processor,
        messages=queue,
        tools=tools,
        prompts=NullPromptLoader(),
        state=visible_state,
        memory=NullMemory(),
        inference=inference,
        ui_event_sink=ui_events.append,
        question_store=question_store,
        project_store=project_store,
        schedule_store=schedule_store,
    )
    return TuiRuntime(
        user=str(user or "local").strip() or "local",
        queue=queue,
        channel=channel,
        visible_state=visible_state,
        processor=processor,
        core=core,
        question_store=question_store,
        project_store=project_store,
        schedule_store=schedule_store,
        ui_events=ui_events,
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
    if command in {"project", "project-context"}:
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
    step = runtime.core.step(MessageSelector(user=runtime.user))
    snapshot = runtime.visible_state.snapshot()
    response = _response_from_snapshot(snapshot, step)
    return TuiProcessingResult(
        response=response,
        event=f"step={step.status.value}",
        step_status=step.status,
        queued_message_id=step.queued_message_id,
    )


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
            self.runtime = build_tui_runtime(
                project_store=ProjectStore.default(),
                schedule_store=ScheduledTaskStore.default(),
            )
            self.runtime.core.activity_sink = self._emit_activity_from_worker
            self.processor = TuiProcessorCoordinator(self.runtime)
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

            result = queue_tui_input(self.runtime, prompt)
            if not result.prompt:
                return
            chat = self.query_one("#chat", RichLog)
            chat.write(Text.assemble((self.runtime.user, "bold cyan"), f": {result.prompt}"))
            if result.queued_message_id:
                self.last_message_id = result.queued_message_id
            self._refresh_status()
            if result.should_exit:
                self.exit()
                return
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

        def _start_processor_if_needed(self) -> None:
            if not self.processor.reserve_processing():
                self.query_one("#activity", Static).update("Queued: processor already running")
                self._refresh_status()
                return
            self.run_worker(self._process_queue_worker, thread=True, exclusive=False)

        def _process_queue_worker(self) -> None:
            results = self.processor.process_until_idle()
            self.call_from_thread(self._apply_processing_results, results)

        def _emit_activity_from_worker(self, event: CoreActivityEvent) -> None:
            self.call_from_thread(self._append_activity_event, event)

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
                    f"processing: {self.processor.is_processing}",
                    f"user: {self.runtime.user}",
                    f"current project: {format_current_project_status(self.runtime)}",
                    f"inference: {format_inference_status(self.runtime)}",
                    f"queue: {self.runtime.queue.size()}",
                    f"last message: {self.last_message_id or '-'}",
                ]
            )
            self.query_one("#status", Static).update(text)

    return AlphonseTuiApp
