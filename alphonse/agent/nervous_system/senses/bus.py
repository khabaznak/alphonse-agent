"""In-memory signal bus.

The bus is transport-only: it enqueues signals and allows a consumer (Heart)
to block waiting for the next signal. No persistence, no handlers, no ack.
"""

import uuid
from functools import lru_cache
from queue import Queue, Empty
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Signal:
    """A unit of input for Alphonse's heart loop."""

    type: str
    payload: dict[str, object] = field(default_factory=dict)
    source: str | None = None
    created_at: str | None = None
    durable: bool = False
    correlation_id: str | None = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        if not self.correlation_id:
            self.correlation_id = self.id


class Bus:
    """Transport-only signal bus.

    Producers call `emit()` to enqueue signals. The consumer calls `get()` to
    block waiting for the next signal.
    """

    def __init__(self) -> None:
        self._q: Queue[Signal] = Queue()

    def emit(self, signal: Signal) -> None:
        """Enqueue a signal."""
        _validate_signal_contract(signal)
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

@lru_cache(maxsize=1)
def _allowed_signal_keys_by_source() -> dict[str, set[str]]:
    from alphonse.agent.nervous_system.senses.registry import all_signal_specs
    from alphonse.agent.nervous_system.senses.registry import all_senses

    sense_keys = {cls.key for cls in all_senses()}
    allowed: dict[str, set[str]] = {key: set() for key in sense_keys}
    for spec in all_signal_specs():
        src = str(spec.source or "").strip()
        key = str(spec.key or "").strip()
        if src in allowed and key:
            allowed[src].add(key)
    return allowed


def _validate_signal_contract(signal: Signal) -> None:
    source = str(signal.source or "").strip()
    if not source:
        return
    allowed_map = _allowed_signal_keys_by_source()
    allowed = allowed_map.get(source)
    if not allowed:
        return
    if str(signal.type or "").strip() in allowed:
        return
    raise ValueError(
        f"invalid_signal_type_for_source: source={source} type={signal.type}"
    )
