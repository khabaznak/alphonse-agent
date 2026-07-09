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
from alphonse.agent_v2.core.state import get_state
from alphonse.agent_v2.core.state import reset_state
from alphonse.agent_v2.core.tools.registry.native import BASH_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import RESPOND_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry

LOCAL_STOP_COMMANDS = {"/exit", "/quit", "/stop"}


@dataclass
class TuiRuntime:
    user: str
    queue: InMemoryMessageQueue
    channel: CommunicationChannel
    visible_state: "InMemoryInternalState"
    processor: IntelligenceProcessor
    core: AlphonseCore


@dataclass(frozen=True)
class TuiSubmissionResult:
    prompt: str
    response: str
    event: str
    should_exit: bool = False
    queued: bool = False
    queued_message_id: str | None = None
    step_status: LoopStepStatus | None = None


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

    def execute(self, tool_id: str, arguments: dict[str, Any]) -> Any:
        _ = tool_id
        _ = arguments
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
) -> TuiRuntime:
    reset_state()
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    visible_state = InMemoryInternalState()
    processor = processor or PDCAIntelligenceProcessor()
    tools = tools or build_native_tool_registry()
    inference = inference or build_default_tui_inference_router()
    core = AlphonseCore(
        intelligence=processor,
        messages=queue,
        tools=tools,
        prompts=NullPromptLoader(),
        state=visible_state,
        memory=NullMemory(),
        inference=inference,
    )
    return TuiRuntime(
        user=str(user or "local").strip() or "local",
        queue=queue,
        channel=channel,
        visible_state=visible_state,
        processor=processor,
        core=core,
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
    prompt_value = str(prompt or "").strip()
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

    queued = runtime.channel.queue_message(prompt=prompt_value, user=runtime.user)
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


def main() -> None:
    app_cls = _build_textual_app_class()
    app_cls().run()


def _build_textual_app_class() -> type[Any]:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Footer, Header, Input, Label, RichLog, Static
        from rich.text import Text
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Textual is required for the Alphonse v2 TUI. Install dependencies with `pip install -r requirements.txt`."
        ) from exc

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

        #chat, #events {
            border: solid $primary;
        }

        #status {
            border: solid $secondary;
            height: 8;
            padding: 1;
        }
        """

        BINDINGS = [("ctrl+c", "quit", "Quit")]

        def __init__(self) -> None:
            super().__init__()
            self.runtime = build_tui_runtime()
            self.runtime.core.activity_sink = self._emit_activity_from_worker
            self.processor = TuiProcessorCoordinator(self.runtime)
            self.last_message_id = ""

        def compose(self) -> ComposeResult:
            yield Header(show_clock=True)
            with Horizontal(id="main"):
                with Vertical(id="chat-column"):
                    yield RichLog(id="chat", wrap=True, highlight=True)
                    yield Input(placeholder="Message Alphonse...", id="prompt")
                with Vertical(id="side-column"):
                    yield Static(id="status")
                    yield RichLog(id="events", wrap=True, highlight=True)
            yield Footer()

        def on_mount(self) -> None:
            self.title = "Alphonse v2"
            self.query_one("#chat", RichLog).write(Text.from_markup("[bold]Alphonse v2[/bold] native TUI"))
            self.query_one("#events", RichLog).write("ready")
            self._refresh_status()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            prompt = event.value
            event.input.value = ""
            result = queue_tui_input(self.runtime, prompt)
            if not result.prompt:
                return
            chat = self.query_one("#chat", RichLog)
            events = self.query_one("#events", RichLog)
            chat.write(Text.assemble((self.runtime.user, "bold cyan"), f": {result.prompt}"))
            events.write(result.event)
            if result.queued_message_id:
                self.last_message_id = result.queued_message_id
            self._refresh_status()
            if result.should_exit:
                self.exit()
                return
            self._start_processor_if_needed()

        def _start_processor_if_needed(self) -> None:
            if not self.processor.reserve_processing():
                self.query_one("#events", RichLog).write("processor already running; message remains queued")
                self._refresh_status()
                return
            self.run_worker(self._process_queue_worker, thread=True, exclusive=False)

        def _process_queue_worker(self) -> None:
            results = self.processor.process_until_idle()
            self.call_from_thread(self._apply_processing_results, results)

        def _emit_activity_from_worker(self, event: CoreActivityEvent) -> None:
            self.call_from_thread(self._append_activity_event, event)

        def _append_activity_event(self, event: CoreActivityEvent) -> None:
            chat = self.query_one("#chat", RichLog)
            events = self.query_one("#events", RichLog)
            rendered = format_activity_message(event)
            chat.write(Text.assemble((event.speaker, "bold green"), f": {rendered}"))
            events.write(f"activity={event.phase.value}:{event.label}")

        def _apply_processing_results(self, results: list[TuiProcessingResult]) -> None:
            chat = self.query_one("#chat", RichLog)
            events = self.query_one("#events", RichLog)
            for result in results:
                events.write(result.event)
                if result.queued_message_id:
                    self.last_message_id = result.queued_message_id
                if result.response:
                    chat.write(Text.assemble(("Alphonse", "bold green"), f": {result.response}"))
            self._refresh_status()
            if self.runtime.queue.size(MessageSelector(user=self.runtime.user)) > 0:
                self._start_processor_if_needed()

        def _refresh_status(self) -> None:
            state = get_state()
            text = "\n".join(
                [
                    f"state: {state.key}",
                    f"processing: {self.processor.is_processing}",
                    f"user: {self.runtime.user}",
                    f"queue: {self.runtime.queue.size()}",
                    f"last message: {self.last_message_id or '-'}",
                ]
            )
            self.query_one("#status", Static).update(text)

    return AlphonseTuiApp
