from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from alphonse.agent_v2.core.core import AlphonseCore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.core import LoopStepStatus
from alphonse.agent_v2.core.core import ProcessingResult
from alphonse.agent_v2.core.core import ProcessingStatus
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages import MessageSelector
from alphonse.agent_v2.core.state import AVAILABLE
from alphonse.agent_v2.core.state import ERROR
from alphonse.agent_v2.core.state import WAITING
from alphonse.agent_v2.core.state import WORKING
from alphonse.agent_v2.core.state import CoreSignal
from alphonse.agent_v2.core.state import DDFSM
from alphonse.agent_v2.core.state import State
from alphonse.agent_v2.core.state import get_state
from alphonse.agent_v2.core.state import reset_state
from alphonse.agent_v2.core.state.ddfsm import ERROR_CLEARED
from alphonse.agent_v2.core.state.ddfsm import MESSAGE_DEQUEUED
from alphonse.agent_v2.core.state.ddfsm import PROCESSOR_COMPLETED
from alphonse.agent_v2.core.state.ddfsm import PROCESSOR_FAILED
from alphonse.agent_v2.core.state.ddfsm import PROCESSOR_RESUMED
from alphonse.agent_v2.core.state.ddfsm import PROCESSOR_WAITING


def test_initial_global_state_is_available() -> None:
    reset_state()

    assert get_state().key == AVAILABLE


def test_available_state_consumes_message_and_returns_available_on_completion() -> None:
    core, queue, processor = _core(_Processor(ProcessingStatus.COMPLETED))
    queue.enqueue(_message("first"))

    result = core.step()

    assert result.status == LoopStepStatus.PROCESSED
    assert result.state_before.key == AVAILABLE
    assert result.state_after.key == AVAILABLE
    assert processor.processed == ["first"]
    assert queue.size() == 0


def test_working_state_does_not_let_outer_loop_consume_message() -> None:
    core, queue, processor = _core(_Processor(ProcessingStatus.COMPLETED))
    State.set(core.fsm.current_state_for_key(WORKING))
    queue.enqueue(_message("blocked"))

    result = core.step()

    assert result.status == LoopStepStatus.BUSY
    assert processor.processed == []
    assert queue.size() == 1


def test_processor_can_consume_steering_message_through_context() -> None:
    processor = _SteeringProcessor()
    core, queue, _ = _core(processor)
    queue.enqueue(_message("main", user="alex"))
    queue.enqueue(_message("steering", user="gaby"))

    result = core.step(MessageSelector(user="alex"))

    assert result.status == LoopStepStatus.PROCESSED
    assert processor.steering == "steering"
    assert queue.size() == 0


def test_processor_waiting_outcome_transitions_to_waiting() -> None:
    core, queue, _ = _core(_Processor(ProcessingStatus.WAITING))
    queue.enqueue(_message("needs answer"))

    result = core.step()

    assert result.status == LoopStepStatus.WAITING
    assert result.state_after.key == WAITING
    assert get_state().key == WAITING


def test_processor_failed_outcome_transitions_to_error() -> None:
    core, queue, _ = _core(_Processor(ProcessingStatus.FAILED, error="broken"))
    queue.enqueue(_message("fails"))

    result = core.step()

    assert result.status == LoopStepStatus.FAILED
    assert result.state_after.key == ERROR
    assert result.error == "broken"


def test_request_stop_prevents_further_queue_consumption() -> None:
    core, queue, processor = _core(_Processor(ProcessingStatus.COMPLETED))
    queue.enqueue(_message("do not process"))

    core.request_stop()
    result = core.step()

    assert result.status == LoopStepStatus.STOPPED
    assert processor.processed == []
    assert queue.size() == 1


def test_ddfsm_returns_expected_seeded_transitions() -> None:
    fsm = DDFSM()

    cases = (
        (AVAILABLE, MESSAGE_DEQUEUED, WORKING),
        (WORKING, PROCESSOR_COMPLETED, AVAILABLE),
        (WORKING, PROCESSOR_WAITING, WAITING),
        (WAITING, PROCESSOR_RESUMED, WORKING),
        (AVAILABLE, PROCESSOR_FAILED, ERROR),
        (WORKING, PROCESSOR_FAILED, ERROR),
        (WAITING, PROCESSOR_FAILED, ERROR),
        (ERROR, ERROR_CLEARED, AVAILABLE),
    )
    for from_state, signal, to_state in cases:
        outcome = fsm.handle(fsm.current_state_for_key(from_state), CoreSignal(signal))
        assert outcome.matched
        assert outcome.next_state_key == to_state


def _message(content: str, *, user: str = "alex") -> CoreMessage:
    return CoreMessage(timestamp=datetime.now().astimezone(), prompt=content, user=user)


def _core(processor: object) -> tuple[AlphonseCore, InMemoryMessageQueue, object]:
    reset_state()
    queue = InMemoryMessageQueue()
    return (
        AlphonseCore(
            intelligence=processor,
            messages=queue,
            tools=_NullTools(),
            prompts=_NullPrompts(),
            state=_RecordingState(),
            memory=_NullMemory(),
        ),
        queue,
        processor,
    )


@dataclass
class _Processor:
    status: ProcessingStatus
    error: str | None = None
    processed: list[str] | None = None

    def __post_init__(self) -> None:
        if self.processed is None:
            self.processed = []

    def process(self, message: CoreMessage, context: CoreLoopContext) -> ProcessingResult:
        _ = context
        assert self.processed is not None
        self.processed.append(message.prompt)
        return ProcessingResult(
            snapshot=StateSnapshot(current_work=message.prompt),
            status=self.status,
            error=self.error,
        )


@dataclass
class _SteeringProcessor:
    steering: str | None = None

    def process(self, message: CoreMessage, context: CoreLoopContext) -> ProcessingResult:
        _ = message
        steering = context.consume_message(MessageSelector(user="gaby"))
        self.steering = steering.message.prompt if steering else None
        return ProcessingResult(snapshot=StateSnapshot(current_work="main"))


@dataclass
class _RecordingState:
    value: StateSnapshot | None = None

    def update(self, snapshot: StateSnapshot) -> None:
        self.value = snapshot

    def snapshot(self) -> StateSnapshot:
        return self.value or StateSnapshot()


class _NullTools:
    pass


class _NullPrompts:
    pass


class _NullMemory:
    pass
