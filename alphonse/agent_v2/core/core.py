"""Core loop contracts for Alphonse agent v2.

This module intentionally defines v2-native interfaces only. It does not
import or adapt v1 agent internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from time import sleep
from typing import TYPE_CHECKING, Any, Protocol

from alphonse.agent_v2.core.state.ddfsm import AVAILABLE
from alphonse.agent_v2.core.state.ddfsm import ERROR
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
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.messages.queue import MessageSelector, QueuedMessage


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
    WAITING = "waiting"
    FAILED = "failed"


class LoopStepStatus(str, Enum):
    """Observable result of one core loop step."""

    STOPPED = "stopped"
    BUSY = "busy"
    EMPTY = "empty"
    PROCESSED = "processed"
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


@dataclass
class CoreLoopContext:
    """Processor-controlled access to selected queued messages."""

    messages: MessageQueue
    tools: ToolRegistry | None = None

    def consume_message(self, selector: MessageSelector | None = None) -> QueuedMessage | None:
        return self.messages.dequeue(selector)


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

    def register(self, tool: ToolDescriptor) -> None:
        """Register a tool descriptor."""

    def get(self, name: str) -> ToolDescriptor | None:
        """Return a registered tool descriptor by name."""

    def list(self) -> tuple[ToolDescriptor, ...]:
        """Return all registered tool descriptors."""


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
    fsm: DDFSM = field(default_factory=build_default_ddfsm)
    _stop_requested: bool = field(default=False, init=False, repr=False)

    def run_once(self, selector: MessageSelector | None = None) -> StateSnapshot | None:
        """Process one queued message and return its visible snapshot."""
        result = self.step(selector)
        if result.status in {LoopStepStatus.PROCESSED, LoopStepStatus.WAITING, LoopStepStatus.FAILED}:
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
        try:
            result = self.intelligence.process(
                task,
                CoreLoopContext(messages=self.messages, tools=self.tools),
            )
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

    def _transition(self, signal_key: str) -> CurrentState:
        current = State.snapshot()
        outcome = self.fsm.handle(current, CoreSignal(signal_key))
        return State.apply(outcome)


def _signal_for_processing_status(status: ProcessingStatus) -> str:
    if status == ProcessingStatus.COMPLETED:
        return PROCESSOR_COMPLETED
    if status == ProcessingStatus.WAITING:
        return PROCESSOR_WAITING
    return PROCESSOR_FAILED


def _loop_status_for_processing_status(status: ProcessingStatus) -> LoopStepStatus:
    if status == ProcessingStatus.COMPLETED:
        return LoopStepStatus.PROCESSED
    if status == ProcessingStatus.WAITING:
        return LoopStepStatus.WAITING
    return LoopStepStatus.FAILED
