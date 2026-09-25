"""Core loop contracts for Alphonse agent v2.

This module intentionally defines v2-native interfaces only. It does not
import or adapt v1 agent internals.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from enum import Enum
from time import sleep
from typing import TYPE_CHECKING, Any, Callable, Protocol
from uuid import uuid4

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
    CANCELLED = "cancelled"


class LoopStepStatus(str, Enum):
    """Observable result of one core loop step."""

    STOPPED = "stopped"
    BUSY = "busy"
    EMPTY = "empty"
    PROCESSED = "processed"
    PARKED = "parked"
    WAITING = "waiting"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CoreMessage:
    """Message envelope for all communication with the core loop."""

    timestamp: datetime
    prompt: str
    user: str
    project_id: str = ""
    memory_session_id: str = ""
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
    # Program mode uses these explicit semantics instead of inferring safety
    # from a tool name or description.
    program_behavior: str = "normal"
    read_only: bool = False


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
    progress: dict[str, Any] = field(default_factory=dict)
    occurred_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())


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
    user_context_provider: Callable[[str], str] | None = None
    user_timezone_provider: Callable[[str], str] | None = None
    memory: Any | None = None


@dataclass
class CoreLoopContext:
    """Processor-controlled access to selected queued messages."""

    messages: MessageQueue
    tools: ToolRegistry | None = None
    inference: InferenceRouter | None = None
    prompts: SystemPromptLoader | None = None
    activity_sink: Callable[[CoreActivityEvent], None] | None = None
    ui_event_sink: Callable[[CoreUiEvent], None] | None = None
    question_store: Any | None = None
    project_store: Any | None = None
    schedule_store: Any | None = None
    delivery_sink: Callable[[dict[str, Any]], Any] | None = None
    user_context_provider: Callable[[str], str] | None = None
    user_timezone_provider: Callable[[str], str] | None = None
    memory: Any | None = None
    program_runner: Any | None = None
    cancellation_checker: Callable[[], bool] | None = None
    telemetry_sink: Callable[[dict[str, Any]], None] | None = None
    system_one: Any | None = None
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

    def emit_activity(self, *, phase: ImprovementPhase, label: str, message: str, progress: dict[str, Any] | None = None) -> None:
        if self.activity_sink is None:
            return
        self.activity_sink(CoreActivityEvent(phase=phase, label=label, message=message, progress=dict(progress or {})))

    def emit_ui_event(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        if self.ui_event_sink is None:
            return
        self.ui_event_sink(CoreUiEvent(event_type=event_type, payload=dict(payload or {})))

    def emit_telemetry(self, event: dict[str, Any]) -> None:
        if self.telemetry_sink is not None:
            self.telemetry_sink(dict(event))

    def tool_execution_context(self, task: TaskState) -> ToolExecutionContext:
        return ToolExecutionContext(
            task=task,
            messages=self.messages,
            ui_event_sink=self.ui_event_sink,
            question_store=self.question_store,
            project_store=self.project_store,
            schedule_store=self.schedule_store,
            delivery_sink=self.delivery_sink,
            memory=self.memory,
            user_timezone_provider=self.user_timezone_provider,
        )

    def is_cancelled(self) -> bool:
        return bool(self.cancellation_checker and self.cancellation_checker())

    def record_memory_event(self, task: TaskState, heading: str, content: Any) -> None:
        if self.memory is None:
            return
        record = getattr(self.memory, "event", None)
        if callable(record):
            record(task, heading, content)


_SENSITIVE_PROGRESS_KEYS = ("secret", "token", "password", "authorization", "cookie", "api_key", "apikey")


def _task_progress_snapshot(task: Any) -> dict[str, Any]:
    """Build the Desktop progress view from the current TaskState only."""
    metadata = getattr(task, "metadata", {}) if isinstance(getattr(task, "metadata", {}), dict) else {}
    planned = metadata.get("planned_tool_call") if isinstance(metadata.get("planned_tool_call"), dict) else None
    latest_call = getattr(task, "get_latest_executed_plan_call", None)
    latest = latest_call() if callable(latest_call) else None
    latest = latest if isinstance(latest, dict) else None
    selected = planned or latest or {}
    execution = latest.get("execution") if isinstance(latest, dict) and isinstance(latest.get("execution"), dict) else {}
    try:
        plan_calls = json.loads(str(getattr(task, "plan_json", "") or ""))
    except (TypeError, ValueError):
        plan_calls = []
    steps = []
    task_calls = plan_calls if isinstance(plan_calls, list) else []
    recent_calls = task_calls[-10:]
    for attempt, call in enumerate(recent_calls, start=max(1, len(task_calls) - len(recent_calls) + 1)):
        if not isinstance(call, dict):
            continue
        call_execution = call.get("execution") if isinstance(call.get("execution"), dict) else {}
        tool_id = str(call.get("tool_id") or "").strip()
        steps.append({
            "attempt": attempt,
            "intention": _truncate_progress(str(call.get("internal_state") or ""), 500),
            "tool_name": str(call.get("tool_name") or _progress_tool_label(tool_id) or ("Program (Python)" if call.get("execution_mode") == "program" else "Tool")).strip(),
            "arguments": _safe_progress_value(call.get("arguments"), limit=6000),
            "action": _safe_progress_value(call.get("program"), limit=10000) if call.get("execution_mode") == "program" else None,
            "status": str(call_execution.get("status") or "planned").strip(),
            "result": _safe_progress_value(call_execution.get("result"), limit=4000),
        })
    hierarchical_state = getattr(task, "hierarchical_state", {})
    hierarchical_state = hierarchical_state if isinstance(hierarchical_state, dict) else {}
    phase = hierarchical_state.get("phase") if isinstance(hierarchical_state.get("phase"), dict) else {}
    subgoal_intentions = {
        str(item.get("subgoal_id") or "").strip(): str(item.get("objective") or "").strip()
        for item in phase.get("subgoals") or []
        if isinstance(item, dict) and str(item.get("subgoal_id") or "").strip()
    }
    v3_actions = [item for item in hierarchical_state.get("actions") or [] if isinstance(item, dict)]
    if v3_actions:
        steps = []
        recent_actions = v3_actions[-10:]
        for attempt, action in enumerate(recent_actions, start=max(1, len(v3_actions) - len(recent_actions) + 1)):
            subgoal_id = str(action.get("subgoal_id") or "").strip()
            result = action.get("result")
            error = str(action.get("error") or "").strip()
            if result in (None, "", {}, []) and error:
                result = {"error": error}
            steps.append({
                "attempt": attempt,
                "intention": _truncate_progress(subgoal_intentions.get(subgoal_id, ""), 500),
                "tool_name": _progress_tool_label(str(action.get("tool_id") or "").strip()),
                "arguments": _safe_progress_value(action.get("arguments"), limit=6000),
                "action": _safe_progress_value(action.get("command") or action.get("action"), limit=10000),
                "status": str(action.get("status") or "planned").strip(),
                "result": _safe_progress_value(result, limit=4000),
            })
        current = v3_actions[-1]
        current_subgoal_id = str(current.get("subgoal_id") or "").strip()
        current_result = current.get("result")
        current_error = str(current.get("error") or "").strip()
        if current_result in (None, "", {}, []) and current_error:
            current_result = {"error": current_error}
        selected = {
            "tool_id": str(current.get("tool_id") or "").strip(),
            "arguments": current.get("arguments"),
            "internal_state": subgoal_intentions.get(current_subgoal_id, ""),
        }
        execution = {
            "status": str(current.get("status") or "planned").strip(),
            "result": current_result,
        }
    acceptance_criteria = str(getattr(task, "acceptance_criteria_md", "") or "").strip()
    if acceptance_criteria == "- (none)":
        acceptance_criteria = ""
    return {
        "project_id": str(getattr(task, "project_id", "") or "").strip(),
        "acceptance_criteria": _truncate_progress(acceptance_criteria, 1200),
        "tool_name": str(selected.get("tool_name") or _progress_tool_label(str(selected.get("tool_id") or "").strip())).strip(),
        "tool_arguments": _safe_progress_value(selected.get("arguments") if isinstance(selected, dict) else {}),
        "tool_result": _safe_progress_value(execution.get("result") if isinstance(execution, dict) else None, limit=4000),
        "tool_status": str(execution.get("status") or "").strip() if isinstance(execution, dict) else "",
        "intention": _truncate_progress(str(selected.get("internal_state") or ""), 500),
        "steps": steps,
        "goal": _truncate_progress(str(getattr(task, "goal", "") or ""), 1200),
        "facts": _truncate_progress(str(getattr(task, "facts_md", "") or ""), 3000),
        "memory_facts": _truncate_progress(str(getattr(task, "memory_facts_md", "") or ""), 3000),
        "recent_conversation": _truncate_progress(str(getattr(task, "recent_conversation_md", "") or ""), 3000),
        "conversation_history": _truncate_progress(str(getattr(task, "conversation_history_md", "") or ""), 5000),
        "updates": _truncate_progress(str(getattr(task, "updates_md", "") or ""), 3000),
        "evidence": _safe_progress_value(getattr(task, "evidence_journal", [])[-10:], limit=3000),
        "question": _safe_progress_value(metadata.get("question_interrupt")),
        "outcome": _safe_progress_value(getattr(task, "outcome", None)),
        "status": str(getattr(task, "status", "") or "").strip(),
    }


def _safe_progress_value(value: Any, *, key: str = "", depth: int = 0, limit: int = 500) -> Any:
    if any(marker in key.lower() for marker in _SENSITIVE_PROGRESS_KEYS):
        return "[redacted]"
    if depth >= 3:
        return "[truncated]"
    if isinstance(value, str):
        return _truncate_progress(value, limit)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    if isinstance(value, list):
        return [_safe_progress_value(item, depth=depth + 1, limit=limit) for item in value[:12]]
    if isinstance(value, dict):
        return {str(item_key): _safe_progress_value(item_value, key=str(item_key), depth=depth + 1, limit=limit) for item_key, item_value in list(value.items())[:20]}
    return _truncate_progress(str(value), limit)


def _truncate_progress(value: str, limit: int) -> str:
    text = str(value or "").strip()
    return text if len(text) <= limit else f"{text[:limit - 1]}…"


def _progress_tool_label(tool_id: str) -> str:
    normalized = str(tool_id or "").strip()
    if normalized.startswith("native."):
        return normalized.removeprefix("native.").replace("_", " ").title()
    if normalized.startswith("artifact."):
        return f"Artifact · {normalized.removeprefix('artifact.')}"
    return normalized


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

    def list_pending(self, selector: MessageSelector | None = None, *, limit: int = 1000) -> list[QueuedMessage]:
        """List pending messages in arrival order without claiming them."""

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
    telemetry_sink: Callable[[dict[str, Any]], None] | None = None
    system_one: Any | None = None
    question_store: Any | None = None
    project_store: Any | None = None
    schedule_store: Any | None = None
    delivery_sink: Callable[[dict[str, Any]], Any] | None = None
    user_context_provider: Callable[[str], str] | None = None
    user_timezone_provider: Callable[[str], str] | None = None
    program_runner: Any | None = None
    cancellation_checker: Callable[[str], bool] | None = None
    active_task_callback: Callable[[QueuedMessage, TaskState], None] | None = None
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

        queued_engine = str(queued.message.metadata.get("intelligence_engine") or "").strip()
        task = None
        if queued_engine == "hierarchical_v3" and self.question_store is not None:
            task = self.question_store.load_task_checkpoint(queued.message_id)
        if task is None:
            task = TaskState.from_queued_message(queued)
        routing_disposition = str(queued.message.metadata.get("routing_disposition") or "")
        raw_task_state = queued.message.metadata.get("task_state")
        is_correlated_resume = (
            routing_disposition == "correlated_response"
            and isinstance(raw_task_state, dict)
            and bool(str(task.task_id or "").strip())
        )
        if queued_engine == "hierarchical_v3" and not is_correlated_resume:
            # A queue delivery may be retried, but it is still the same V3 task.
            # Using the durable message id prevents a retry from silently creating
            # a new acceptance contract and a different execution history.
            task.task_id = queued.message_id
        elif not task.task_id:
            task.task_id = str(uuid4())
        if self.active_task_callback is not None:
            self.active_task_callback(queued, task)

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
                    progress=_task_progress_snapshot(task),
                )
            )
        try:
            context = CoreLoopContext(
                messages=self.messages,
                tools=self.tools,
                inference=self.inference,
                prompts=self.prompts,
                activity_sink=_task_activity_sink,
                ui_event_sink=self.ui_event_sink,
                telemetry_sink=self.telemetry_sink,
                system_one=self.system_one,
                question_store=self.question_store,
                project_store=self.project_store,
                schedule_store=self.schedule_store,
                delivery_sink=self.delivery_sink,
                user_context_provider=self.user_context_provider,
                user_timezone_provider=self.user_timezone_provider,
                memory=self.memory,
                program_runner=self.program_runner,
                cancellation_checker=(lambda: bool(self.cancellation_checker and self.cancellation_checker(queued.message_id))),
            )
            start_memory = getattr(self.memory, "start_task", None)
            if callable(start_memory):
                task.conversation_history_md = str(start_memory(task) or "")
            result = self.intelligence.process(task, context)
            finish_memory = getattr(self.memory, "finish_task", None)
            if callable(finish_memory):
                serialized_task = result.snapshot.metadata.get("task_state") if isinstance(result.snapshot.metadata, dict) else None
                finished_task = TaskState.from_dict(serialized_task) if isinstance(serialized_task, dict) else task
                finish_memory(finished_task)
            if result.status not in {ProcessingStatus.FAILED, ProcessingStatus.CANCELLED}:
                context.acknowledge_consumed_messages()
        except Exception as exc:
            cancelled = bool(self.cancellation_checker and self.cancellation_checker(queued.message_id))
            if cancelled:
                task.status = "cancelled"
                task.outcome = {"status": "cancelled", "reason": "Execution was cancelled by the kill switch."}
            result = ProcessingResult(
                snapshot=StateSnapshot(
                    current_work=task.goal,
                    metadata={"exception_type": type(exc).__name__, "task_state": task.to_dict()},
                ),
                status=ProcessingStatus.CANCELLED if cancelled else ProcessingStatus.FAILED,
                error="inference_cancelled" if cancelled else str(exc),
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
    if status == ProcessingStatus.CANCELLED:
        return PROCESSOR_COMPLETED
    return PROCESSOR_FAILED


def _loop_status_for_processing_status(status: ProcessingStatus) -> LoopStepStatus:
    if status == ProcessingStatus.COMPLETED:
        return LoopStepStatus.PROCESSED
    if status == ProcessingStatus.PARKED:
        return LoopStepStatus.PARKED
    if status == ProcessingStatus.WAITING:
        return LoopStepStatus.WAITING
    if status == ProcessingStatus.CANCELLED:
        return LoopStepStatus.CANCELLED
    return LoopStepStatus.FAILED
