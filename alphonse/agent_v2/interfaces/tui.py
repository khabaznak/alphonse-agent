"""Textual native interface for Alphonse v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphonse.agent_v2.core.core import AlphonseCore
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.core import MemoryRecord
from alphonse.agent_v2.core.core import PromptFile
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.core import ToolDescriptor
from alphonse.agent_v2.core.intelligence import BasicIntelligenceProcessor
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages import MessageSelector
from alphonse.agent_v2.core.state import get_state
from alphonse.agent_v2.core.state import reset_state

LOCAL_STOP_COMMANDS = {"/exit", "/quit", "/stop"}


@dataclass
class TuiRuntime:
    user: str
    queue: InMemoryMessageQueue
    channel: CommunicationChannel
    visible_state: "InMemoryInternalState"
    processor: BasicIntelligenceProcessor
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


def build_tui_runtime(*, user: str = "local") -> TuiRuntime:
    reset_state()
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    visible_state = InMemoryInternalState()
    processor = BasicIntelligenceProcessor()
    core = AlphonseCore(
        intelligence=processor,
        messages=queue,
        tools=NullToolRegistry(),
        prompts=NullPromptLoader(),
        state=visible_state,
        memory=NullMemory(),
    )
    return TuiRuntime(
        user=str(user or "local").strip() or "local",
        queue=queue,
        channel=channel,
        visible_state=visible_state,
        processor=processor,
        core=core,
    )


def submit_tui_input(runtime: TuiRuntime, prompt: str) -> TuiSubmissionResult:
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
    step = runtime.core.step(MessageSelector(user=runtime.user))
    snapshot = runtime.visible_state.snapshot()
    response = str(snapshot.metadata.get("response") or "")
    command_event = ""
    if queued.message.metadata.get("is_command"):
        command_event = f" command=/{queued.message.metadata.get('command')}"
    return TuiSubmissionResult(
        prompt=prompt_value,
        response=response,
        event=f"queued {queued.message_id}; step={step.status.value}{command_event}",
        queued=True,
        queued_message_id=queued.message_id,
        step_status=step.status,
    )


def main() -> None:
    app_cls = _build_textual_app_class()
    app_cls().run()


def _build_textual_app_class() -> type[Any]:
    try:
        from textual.app import App, ComposeResult
        from textual.containers import Horizontal, Vertical
        from textual.widgets import Footer, Header, Input, Label, RichLog, Static
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
            self.query_one("#chat", RichLog).write("[bold]Alphonse v2[/bold] native TUI")
            self.query_one("#events", RichLog).write("ready")
            self._refresh_status()

        def on_input_submitted(self, event: Input.Submitted) -> None:
            prompt = event.value
            event.input.value = ""
            result = submit_tui_input(self.runtime, prompt)
            if not result.prompt:
                return
            chat = self.query_one("#chat", RichLog)
            events = self.query_one("#events", RichLog)
            chat.write(f"[bold cyan]{self.runtime.user}[/bold cyan]: {result.prompt}")
            if result.response:
                chat.write(f"[bold green]Alphonse[/bold green]: {result.response}")
            events.write(result.event)
            if result.queued_message_id:
                self.last_message_id = result.queued_message_id
            self._refresh_status()
            if result.should_exit:
                self.exit()

        def _refresh_status(self) -> None:
            state = get_state()
            text = "\n".join(
                [
                    f"state: {state.key}",
                    f"user: {self.runtime.user}",
                    f"queue: {self.runtime.queue.size()}",
                    f"last message: {self.last_message_id or '-'}",
                ]
            )
            self.query_one("#status", Static).update(text)

    return AlphonseTuiApp
