"""In-memory message queue for Alphonse agent v2."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import count
from threading import RLock
from typing import Iterable
from uuid import uuid4

from alphonse.agent_v2.core.core import CoreMessage, MessagePriority


_PRIORITY_RANK = {
    MessagePriority.URGENT: 0,
    MessagePriority.HIGH: 1,
    MessagePriority.NORMAL: 2,
    MessagePriority.LOW: 3,
}


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

    sender_id: str | None = None
    owner_id: str | None = None
    source: str | None = None
    topic: str | None = None
    tags_any: tuple[str, ...] = ()
    tags_all: tuple[str, ...] = ()
    priorities: tuple[MessagePriority, ...] = ()


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
    return (_priority_rank(message.message.priority), message.sequence)


def _priority_rank(priority: MessagePriority | str) -> int:
    try:
        normalized = priority if isinstance(priority, MessagePriority) else MessagePriority(str(priority))
    except ValueError:
        normalized = MessagePriority.NORMAL
    return _PRIORITY_RANK[normalized]


def _matches(queued: QueuedMessage, selector: MessageSelector | None) -> bool:
    if selector is None:
        return True

    message = queued.message
    if selector.sender_id is not None and message.sender_id != selector.sender_id:
        return False
    if selector.owner_id is not None and message.owner_id != selector.owner_id:
        return False
    if selector.source is not None and message.source != selector.source:
        return False
    if selector.topic is not None and message.topic != selector.topic:
        return False
    if selector.priorities and message.priority not in _normalize_priorities(selector.priorities):
        return False

    message_tags = set(message.tags)
    if selector.tags_any and not message_tags.intersection(selector.tags_any):
        return False
    if selector.tags_all and not set(selector.tags_all).issubset(message_tags):
        return False
    return True


def _normalize_priorities(values: Iterable[MessagePriority | str]) -> set[MessagePriority]:
    priorities: set[MessagePriority] = set()
    for value in values:
        try:
            priorities.add(value if isinstance(value, MessagePriority) else MessagePriority(str(value)))
        except ValueError:
            continue
    return priorities

