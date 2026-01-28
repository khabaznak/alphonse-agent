"""In-memory signal bus.

The bus is transport-only: it enqueues signals and allows a consumer (Heart)

to block waiting for the next signal. No persistence, no handlers, no ack.
"""

import uuid
from queue import Queue, Empty
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Signal:
    """A unit of input for Rex's heart loop."""

    type: str
    payload: dict[str, object] = field(default_factory=dict)
    source: str | None = None
    created_at: str | None = None
    durable: bool = False
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


class Bus:
    """Transport-only signal bus.

    Producers call `emit()` to enqueue signals. The consumer calls `get()` to
    block waiting for the next signal.
    """

    def __init__(self) -> None:
        self._q: Queue[Signal] = Queue()

    def emit(self, signal: Signal) -> None:
        """Enqueue a signal."""
        self._q.put(signal)

    def get(self, timeout: Optional[float] = None) -> Optional[Signal]:
        """Return the next signal, blocking up to `timeout` seconds.

        If `timeout` is None, blocks indefinitely. If the timeout expires,
        returns None.
        """
        try:
            return self._q.get(timeout=timeout)
        except Empty:
            return None
