from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent_v2.core.core import AlphonseCore
from alphonse.agent_v2.core.core import CoreMessage
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.core import MessagePriority
from alphonse.agent_v2.core.core import ProcessingResult
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages import MessageSelector
from alphonse.agent_v2.core.state import reset_state


def test_dequeue_uses_fifo_when_priorities_are_equal() -> None:
    queue = InMemoryMessageQueue()
    queue.enqueue(_message("first"))
    queue.enqueue(_message("second"))

    assert queue.dequeue().message.content == "first"
    assert queue.dequeue().message.content == "second"


def test_dequeue_prefers_higher_priority_before_older_lower_priority() -> None:
    queue = InMemoryMessageQueue()
    queue.enqueue(_message("low", priority=MessagePriority.LOW))
    queue.enqueue(_message("urgent", priority=MessagePriority.URGENT))
    queue.enqueue(_message("high", priority=MessagePriority.HIGH))

    assert queue.dequeue().message.content == "urgent"
    assert queue.dequeue().message.content == "high"
    assert queue.dequeue().message.content == "low"


def test_dequeue_can_select_next_message_from_sender() -> None:
    queue = InMemoryMessageQueue()
    queue.enqueue(_message("alex request", sender_id="alex"))
    queue.enqueue(_message("gaby request", sender_id="gaby"))

    queued = queue.dequeue(MessageSelector(sender_id="gaby"))

    assert queued is not None
    assert queued.message.content == "gaby request"
    assert queue.dequeue().message.content == "alex request"


def test_selector_filters_by_topic_and_tags() -> None:
    queue = InMemoryMessageQueue()
    queue.enqueue(_message("file task", topic="files", tags=("household", "work")))
    queue.enqueue(_message("scheduled job", topic="jobs", tags=("scheduled", "maintenance")))

    by_topic = queue.peek(MessageSelector(topic="jobs"))
    by_any_tag = queue.peek(MessageSelector(tags_any=("maintenance", "urgent")))
    by_all_tags = queue.peek(MessageSelector(tags_all=("household", "work")))

    assert by_topic is not None
    assert by_topic.message.content == "scheduled job"
    assert by_any_tag is not None
    assert by_any_tag.message.content == "scheduled job"
    assert by_all_tags is not None
    assert by_all_tags.message.content == "file task"


def test_peek_does_not_remove_messages() -> None:
    queue = InMemoryMessageQueue()
    queue.enqueue(_message("gaby request", sender_id="gaby"))

    assert queue.peek(MessageSelector(sender_id="gaby")) is not None
    assert queue.size() == 1
    assert queue.dequeue(MessageSelector(sender_id="gaby")) is not None
    assert queue.size() == 0


def test_core_run_once_processes_only_matching_message() -> None:
    reset_state()
    queue = InMemoryMessageQueue()
    queue.enqueue(_message("alex request", sender_id="alex"))
    queue.enqueue(_message("gaby request", sender_id="gaby"))
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

    snapshot = core.run_once(MessageSelector(sender_id="gaby"))

    assert snapshot == StateSnapshot(
        phase=ImprovementPhase.PLAN,
        task_owner="gaby",
        current_work="gaby request",
    )
    assert intelligence.processed == ["gaby request"]
    assert queue.dequeue().message.content == "alex request"


def _message(
    content: str,
    *,
    sender_id: str | None = None,
    topic: str | None = None,
    tags: tuple[str, ...] = (),
    priority: MessagePriority = MessagePriority.NORMAL,
) -> CoreMessage:
    return CoreMessage(
        content=content,
        source="test",
        sender_id=sender_id,
        topic=topic,
        tags=tags,
        priority=priority,
    )


@dataclass
class _RecordingIntelligence:
    processed: list[str] | None = None

    def process(self, message: CoreMessage, context: object) -> ProcessingResult:
        _ = context
        if self.processed is None:
            self.processed = []
        self.processed.append(message.content)
        return ProcessingResult(
            snapshot=StateSnapshot(
                phase=ImprovementPhase.PLAN,
                task_owner=message.sender_id,
                current_work=message.content,
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
