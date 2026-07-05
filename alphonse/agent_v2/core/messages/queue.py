"""In-memory message queue for Alphonse agent v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count
from threading import RLock
from uuid import uuid4

from alphonse.agent_v2.core.core import CoreMessage


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass(frozen=True)
class QueuedMessage:
    """A message plus queue-owned metadata."""

    message: CoreMessage
    message_id: str = field(default_factory=lambda: str(uuid4()))
    queued_at: datetime = field(default_factory=_now_utc)
    sequence: int = 0


@dataclass(frozen=True)
class MessageSelector:
    """Query criteria for retrieving messages from the queue."""

    user: str | None = None
    project_id: str | None = None
    tag: str | None = None


class InMemoryMessageQueue:
    """Selector-aware in-memory queue for the initial v2 core."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._messages: list[QueuedMessage] = []
        self._sequence = count(1)

    def enqueue(self, message: CoreMessage) -> QueuedMessage:
        queued = QueuedMessage(message=message, sequence=next(self._sequence))
        with self._lock:
            self._messages.append(queued)
        return queued

    def peek(self, selector: MessageSelector | None = None) -> QueuedMessage | None:
        with self._lock:
            return self._next_matching(selector)

    def dequeue(self, selector: MessageSelector | None = None) -> QueuedMessage | None:
        with self._lock:
            next_message = self._next_matching(selector)
            if next_message is None:
                return None
            self._messages.remove(next_message)
            return next_message

    def size(self, selector: MessageSelector | None = None) -> int:
        with self._lock:
            return sum(1 for message in self._messages if _matches(message, selector))

    def _next_matching(self, selector: MessageSelector | None) -> QueuedMessage | None:
        matches = (message for message in self._messages if _matches(message, selector))
        return min(matches, key=_selection_key, default=None)


def _selection_key(message: QueuedMessage) -> tuple[int, int]:
    return (message.sequence, 0)


def _matches(queued: QueuedMessage, selector: MessageSelector | None) -> bool:
    if selector is None:
        return True

    message = queued.message
    if selector.user is not None and message.user != selector.user:
        return False
    if selector.project_id is not None and message.project_id != selector.project_id:
        return False
    if selector.tag is not None and message.tag != selector.tag:
        return False
    return True

