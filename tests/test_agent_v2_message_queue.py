from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from alphonse.agent_v2.core.core import AlphonseCore
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.core import ProcessingResult
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages import MessageSelector
from alphonse.agent_v2.core.state import reset_state


def test_dequeue_uses_fifo_order() -> None:
    queue = InMemoryMessageQueue()
    queue.enqueue(_message("first"))
    queue.enqueue(_message("second"))

    assert queue.dequeue().message.prompt == "first"
    assert queue.dequeue().message.prompt == "second"


def test_dequeue_can_select_next_message_from_user() -> None:
    queue = InMemoryMessageQueue()
    queue.enqueue(_message("alex request", user="alex"))
    queue.enqueue(_message("gaby request", user="gaby"))

    queued = queue.dequeue(MessageSelector(user="gaby"))

    assert queued is not None
    assert queued.message.prompt == "gaby request"
    assert queue.dequeue().message.prompt == "alex request"


def test_selector_filters_by_project_id_and_tag() -> None:
    queue = InMemoryMessageQueue()
    queue.enqueue(_message("file task", project_id="files", tag="household"))
    queue.enqueue(_message("scheduled job", project_id="jobs", tag="maintenance"))

    by_project = queue.peek(MessageSelector(project_id="jobs"))
    by_tag = queue.peek(MessageSelector(tag="maintenance"))

    assert by_project is not None
    assert by_project.message.prompt == "scheduled job"
    assert by_tag is not None
    assert by_tag.message.prompt == "scheduled job"


def test_peek_does_not_remove_messages() -> None:
    queue = InMemoryMessageQueue()
    queue.enqueue(_message("gaby request", user="gaby"))

    assert queue.peek(MessageSelector(user="gaby")) is not None
    assert queue.size() == 1
    assert queue.dequeue(MessageSelector(user="gaby")) is not None
    assert queue.size() == 0


def test_core_run_once_processes_only_matching_message() -> None:
    reset_state()
    queue = InMemoryMessageQueue()
    queue.enqueue(_message("alex request", user="alex"))
    queue.enqueue(_message("gaby request", user="gaby"))
    intelligence = _RecordingIntelligence()
    state = _RecordingState()
    core = AlphonseCore(
        intelligence=intelligence,
        messages=queue,
        tools=_NullTools(),
        prompts=_NullPrompts(),
        state=state,
        memory=_NullMemory(),
    )

    snapshot = core.run_once(MessageSelector(user="gaby"))

    assert snapshot == StateSnapshot(
        phase=ImprovementPhase.PLAN,
        task_owner="gaby",
        current_work="gaby request",
    )
    assert intelligence.processed == ["gaby request"]
    assert queue.dequeue().message.prompt == "alex request"


def _message(content: str, *, user: str = "alex", project_id: str = "", tag: str = "") -> CoreMessage:
    return CoreMessage(
        timestamp=datetime.now().astimezone(),
        prompt=content,
        user=user,
        project_id=project_id,
        tag=tag,
    )


@dataclass
class _RecordingIntelligence:
    processed: list[str] | None = None

    def process(self, message: CoreMessage, context: object) -> ProcessingResult:
        _ = context
        if self.processed is None:
            self.processed = []
        self.processed.append(message.prompt)
        return ProcessingResult(
            snapshot=StateSnapshot(
                phase=ImprovementPhase.PLAN,
                task_owner=message.user,
                current_work=message.prompt,
            )
        )


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
