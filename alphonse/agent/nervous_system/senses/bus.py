"""In-memory signal bus.

The bus is transport-only: it enqueues signals and allows a consumer (Heart)

to block waiting for the next signal. No persistence, no handlers, no ack.
"""

import json
import uuid
import sqlite3
from pathlib import Path
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
        self._q.put(signal)
        _persist_signal(signal)

    def get(self, timeout: Optional[float] = None) -> Optional[Signal]:
        """Return the next signal, blocking up to `timeout` seconds.

        If `timeout` is None, blocks indefinitely. If the timeout expires,
        returns None.
        """
        try:
            return self._q.get(timeout=timeout)
        except Empty:
            return None


def _persist_signal(signal: Signal) -> None:
    try:
        from alphonse.agent.nervous_system.paths import resolve_nervous_system_db_path

        db_path = resolve_nervous_system_db_path()
        db_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **(signal.payload or {}),
            "correlation_id": signal.correlation_id,
        }
        with sqlite3.connect(db_path) as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO signal_queue
                  (signal_id, signal_type, payload, source, durable)
                VALUES
                  (?, ?, ?, ?, ?)
                """,
                (
                    signal.id,
                    signal.type,
                    json.dumps(payload),
                    signal.source,
                    1 if signal.durable else 0,
                ),
            )
            conn.commit()
    except Exception:
        return
