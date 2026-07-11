"""Core loop contracts for Alphonse agent v2.

This module intentionally defines v2-native interfaces only. It does not
import or adapt v1 agent internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Protocol

from alphonse.agent_v2.core.state.ddfsm import AVAILABLE
from alphonse.agent_v2.core.state.ddfsm import ERROR
from alphonse.agent_v2.core.state.ddfsm import ERROR_CLEARED
from alphonse.agent_v2.core.state.ddfsm import MESSAGE_DEQUEUED
from alphonse.agent_v2.core.state.ddfsm import PROCESSOR_COMPLETED
from alphonse.agent_v2.core.state.ddfsm import PROCESSOR_FAILED
from alphonse.agent_v2.core.state.ddfsm import PROCESSOR_WAITING
from alphonse.agent_v2.core.state.ddfsm import STOP_REQUESTED
from alphonse.agent_v2.core.state.ddfsm import CoreSignal
from alphonse.agent_v2.core.state.ddfsm import CurrentState
from alphonse.agent_v2.core.state.ddfsm import DDFSM
from alphonse.agent_v2.core.state.ddfsm import build_default_ddfsm
from alphonse.agent_v2.core.state.runtime import State

if TYPE_CHECKING:
    from alphonse.agent_v2.core.inference import InferenceRouter
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.messages.queue import MessageSelector, QueuedMessage
    from alphonse.agent_v2.core.tools.registry import ToolDefinition


class ImprovementPhase(str, Enum):
    """PDCA-inspired phases used by the intelligence processor."""

    PLAN = "plan"
    DO = "do"
    CHECK = "check"
    ACT = "act"


class ToolKind(str, Enum):
    """Tool categories available to the registry."""

    NATIVE = "native"
    ARTIFACT = "artifact"


class ProcessingStatus(str, Enum):
    """Outcome states returned by the intelligence processor."""

    COMPLETED = "completed"
    PARKED = "parked"
    WAITING = "waiting"
    FAILED = "failed"


class LoopStepStatus(str, Enum):
    """Observable result of one core loop step."""

    STOPPED = "stopped"
    BUSY = "busy"
    EMPTY = "empty"
    PROCESSED = "processed"
    PARKED = "parked"
    WAITING = "waiting"
    FAILED = "failed"


@dataclass(frozen=True)
class CoreMessage:
    """Message envelope for all communication with the core loop."""

    timestamp: datetime
    prompt: str
    user: str
    project_id: str = ""
    tag: str = ""
    correlation_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolDescriptor:
    """Minimal registry descriptor for native tools and artifacts."""

    tool_id: str
    name: str
    kind: ToolKind
    description: str = ""
    argument_schema: dict[str, Any] = field(default_factory=dict)
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptFile:
    """Loaded agentic prompt file such as soul.md or constitution.md."""

    name: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class StateSnapshot:
    """Owner-visible snapshot of what Alphonse is currently working on."""

    phase: ImprovementPhase | None = None
    task_owner: str | None = None
    current_work: str | None = None
    thought_process: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MemoryRecord:
    """Markdown-backed memory record placeholder."""

    path: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProcessingResult:
    """Result returned by the intelligence processor."""

    snapshot: StateSnapshot
    status: ProcessingStatus = ProcessingStatus.COMPLETED
    error: str | None = None


@dataclass(frozen=True)
class LoopStepResult:
    """Result returned by one core loop step."""

    status: LoopStepStatus
    state_before: CurrentState
    state_after: CurrentState
    queued_message_id: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class CoreActivityEvent:
    """Owner-visible activity event emitted while CAPD is running."""

    phase: ImprovementPhase
    label: str
    message: str
    speaker: str = "Alphonse"
    task_id: str = ""
    message_id: str = ""
    user: str = ""
    integration_id: str = ""
    channel_target: str = ""


@dataclass(frozen=True)
class CoreUiEvent:
    """Protocol-neutral UI event emitted by the core."""

    event_type: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolExecutionContext:
    """Context supplied to native tools that need task/runtime boundaries."""

    task: TaskState
    messages: MessageQueue
    ui_event_sink: Callable[[CoreUiEvent], None] | None = None
    question_store: Any | None = None
    project_store: Any | None = None
    schedule_store: Any | None = None
    delivery_sink: Callable[[dict[str, Any]], Any] | None = None


@dataclass
class CoreLoopContext:
    """Processor-controlled access to selected queued messages."""

    messages: MessageQueue
    tools: ToolRegistry | None = None
    inference: InferenceRouter | None = None
    activity_sink: Callable[[CoreActivityEvent], None] | None = None
    ui_event_sink: Callable[[CoreUiEvent], None] | None = None
    question_store: Any | None = None
    project_store: Any | None = None
    schedule_store: Any | None = None
    delivery_sink: Callable[[dict[str, Any]], Any] | None = None
    consumed_message_ids: list[str] = field(default_factory=list)

    def consume_message(self, selector: MessageSelector | None = None) -> QueuedMessage | None:
        queued = self.messages.dequeue(selector)
        if queued is not None:
            self.consumed_message_ids.append(queued.message_id)
        return queued

    def acknowledge_consumed_messages(self) -> None:
        acknowledge = getattr(self.messages, "ack", None)
        if not callable(acknowledge):
            return
        for message_id in self.consumed_message_ids:
            acknowledge(message_id)

    def emit_activity(self, *, phase: ImprovementPhase, label: str, message: str) -> None:
        if self.activity_sink is None:
            return
        self.activity_sink(CoreActivityEvent(phase=phase, label=label, message=message))

    def emit_ui_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        if self.ui_event_sink is None:
            return
        self.ui_event_sink(CoreUiEvent(event_type=event_type, payload=dict(payload or {})))

    def tool_execution_context(self, task: TaskState) -> ToolExecutionContext:
        return ToolExecutionContext(
            task=task,
            messages=self.messages,
            ui_event_sink=self.ui_event_sink,
            question_store=self.question_store,
            project_store=self.project_store,
            schedule_store=self.schedule_store,
            delivery_sink=self.delivery_sink,
        )


class IntelligenceProcessor(Protocol):
    """PDCA-inspired processor boundary for the core loop."""

    def process(self, task: TaskState, context: CoreLoopContext) -> ProcessingResult:
        """Process one task state and return the resulting processing status."""


class MessageQueue(Protocol):
    """Required ingress path for all communication and work requests."""

    def enqueue(self, message: CoreMessage) -> QueuedMessage:
        """Add a message to the queue."""

    def peek(self, selector: MessageSelector | None = None) -> QueuedMessage | None:
        """Return the next matching queued message without removing it."""

    def dequeue(self, selector: MessageSelector | None = None) -> QueuedMessage | None:
        """Remove and return the next matching message."""

    def size(self, selector: MessageSelector | None = None) -> int:
        """Return the number of queued messages matching a selector."""


class ToolRegistry(Protocol):
    """Lookup and registration boundary for native tools and artifacts."""

    def register(self, tool: ToolDefinition) -> None:
        """Register an executable tool definition."""

    def get(self, name: str) -> ToolDescriptor | None:
        """Return a registered tool descriptor by name."""

    def list(self) -> tuple[ToolDescriptor, ...]:
        """Return all registered tool descriptors."""

    def execute(
        self,
        tool_id: str,
        arguments: dict[str, Any],
        execution_context: ToolExecutionContext | None = None,
    ) -> Any:
        """Execute a registered tool by id."""


class SystemPromptLoader(Protocol):
    """Loader boundary for special agentic prompt files."""

    def load(self, name: str) -> PromptFile:
        """Load a named prompt file."""


class InternalState(Protocol):
    """Owner-visible state projection for the current core activity."""

    def update(self, snapshot: StateSnapshot) -> None:
        """Replace the visible state snapshot."""

    def snapshot(self) -> StateSnapshot:
        """Return the current visible state snapshot."""


class Memory(Protocol):
    """Markdown-backed session and project memory boundary."""

    def write(self, record: MemoryRecord) -> None:
        """Persist a memory record."""

    def read(self, path: str) -> MemoryRecord | None:
        """Read a memory record by path."""


@dataclass
class AlphonseCore:
    """Small wiring object for the future v2 core loop."""

    intelligence: IntelligenceProcessor
    messages: MessageQueue
    tools: ToolRegistry
    prompts: SystemPromptLoader
    state: InternalState
    memory: Memory
    inference: InferenceRouter | None = None
    activity_sink: Callable[[CoreActivityEvent], None] | None = None
    ui_event_sink: Callable[[CoreUiEvent], None] | None = None
    question_store: Any | None = None
    project_store: Any | None = None
    schedule_store: Any | None = None
    delivery_sink: Callable[[dict[str, Any]], Any] | None = None
    fsm: DDFSM = field(default_factory=build_default_ddfsm)
    _stop_requested: bool = field(default=False, init=False, repr=False)

    def run_once(self, selector: MessageSelector | None = None) -> StateSnapshot | None:
        """Process one queued message and return its visible snapshot."""
        result = self.step(selector)
        if result.status in {
            LoopStepStatus.PROCESSED,
            LoopStepStatus.PARKED,
            LoopStepStatus.WAITING,
            LoopStepStatus.FAILED,
        }:
            return self.state.snapshot()
        return None

    def step(self, selector: MessageSelector | None = None) -> LoopStepResult:
        """Run one agentic loop step."""
        state_before = State.snapshot()
        if self._stop_requested:
            return LoopStepResult(
                status=LoopStepStatus.STOPPED,
                state_before=state_before,
                state_after=state_before,
            )

        if state_before.key != AVAILABLE:
            return LoopStepResult(
                status=LoopStepStatus.BUSY,
                state_before=state_before,
                state_after=state_before,
            )

        queued = self.messages.dequeue(selector)
        if queued is None:
            return LoopStepResult(
                status=LoopStepStatus.EMPTY,
                state_before=state_before,
                state_after=state_before,
            )

        working = self._transition(MESSAGE_DEQUEUED)
        from alphonse.agent_v2.core.intelligence.task_state import TaskState

        task = TaskState.from_queued_message(queued)

        def _task_activity_sink(event: CoreActivityEvent) -> None:
            if self.activity_sink is None:
                return
            channel = task.metadata.get("channel") if isinstance(task.metadata, dict) else {}
            channel = channel if isinstance(channel, dict) else {}
            self.activity_sink(
                replace(
                    event,
                    task_id=str(task.task_id or ""),
                    message_id=str(queued.message_id or ""),
                    user=str(task.user or ""),
                    integration_id=str(channel.get("integration_id") or ""),
                    channel_target=str(channel.get("channel_target") or ""),
                )
            )
        try:
            context = CoreLoopContext(
                messages=self.messages,
                tools=self.tools,
                inference=self.inference,
                activity_sink=_task_activity_sink,
                ui_event_sink=self.ui_event_sink,
                question_store=self.question_store,
                project_store=self.project_store,
                schedule_store=self.schedule_store,
                delivery_sink=self.delivery_sink,
            )
            result = self.intelligence.process(task, context)
            if result.status != ProcessingStatus.FAILED:
                context.acknowledge_consumed_messages()
        except Exception as exc:
            result = ProcessingResult(
                snapshot=StateSnapshot(
                    current_work=task.goal,
                    metadata={"exception_type": type(exc).__name__},
                ),
                status=ProcessingStatus.FAILED,
                error=str(exc),
            )

        self.state.update(result.snapshot)
        signal = _signal_for_processing_status(result.status)
        state_after = self._transition(signal)
        return LoopStepResult(
            status=_loop_status_for_processing_status(result.status),
            state_before=state_before,
            state_after=state_after,
            queued_message_id=queued.message_id,
            error=result.error,
        )

    def run_until_stopped(
        self,
        *,
        max_steps: int | None = None,
        idle_sleep_seconds: float = 0.05,
    ) -> None:
        """Run the agentic loop until a stop is requested."""
        steps = 0
        while not self._stop_requested:
            if max_steps is not None and steps >= max_steps:
                return
            result = self.step()
            steps += 1
            if result.status in {LoopStepStatus.EMPTY, LoopStepStatus.BUSY}:
                sleep(idle_sleep_seconds)

    def request_stop(self) -> None:
        """Request loop shutdown without consuming another queue message."""
        self._stop_requested = True
        self.fsm.handle(State.snapshot(), CoreSignal(STOP_REQUESTED))

    def clear_failure(self) -> CurrentState:
        """Release the core after its host has durably recorded a failed attempt."""
        current = State.snapshot()
        outcome = self.fsm.handle(current, CoreSignal(ERROR_CLEARED))
        return State.apply(outcome)

    def _transition(self, signal_key: str) -> CurrentState:
        current = State.snapshot()
        outcome = self.fsm.handle(current, CoreSignal(signal_key))
        return State.apply(outcome)


def _signal_for_processing_status(status: ProcessingStatus) -> str:
    if status == ProcessingStatus.COMPLETED:
        return PROCESSOR_COMPLETED
    if status == ProcessingStatus.PARKED:
        return PROCESSOR_COMPLETED
    if status == ProcessingStatus.WAITING:
        return PROCESSOR_WAITING
    return PROCESSOR_FAILED


def _loop_status_for_processing_status(status: ProcessingStatus) -> LoopStepStatus:
    if status == ProcessingStatus.COMPLETED:
        return LoopStepStatus.PROCESSED
    if status == ProcessingStatus.PARKED:
        return LoopStepStatus.PARKED
    if status == ProcessingStatus.WAITING:
        return LoopStepStatus.WAITING
    return LoopStepStatus.FAILED
