"""Thread-safe runtime state for the Alphonse agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock


def _now() -> datetime:
    return datetime.now(tz=timezone.utc)


@dataclass
class AgentRuntimeState:
    started_at: datetime = field(default_factory=_now)
    last_tick_at: datetime | None = None
    tick_count: int = 0
    current_state: dict[str, object | None] = field(
        default_factory=lambda: {"id": None, "key": None, "name": None}
    )
    last_signal: dict[str, object | None] = field(
        default_factory=lambda: {"type": None, "ts": None, "source": None}
    )
    _lock: Lock = field(default_factory=Lock, repr=False)

    def update(self, **changes: object) -> None:
        """Update fields atomically via keyword args."""
        with self._lock:
            for key, value in changes.items():
                setattr(self, key, value)

    def update_tick(self) -> None:
        with self._lock:
            self.last_tick_at = _now()
            self.tick_count += 1

    def update_state(self, state_id: int | None, key: str | None, name: str | None) -> None:
        with self._lock:
            self.current_state = {"id": state_id, "key": key, "name": name}

    def update_signal(self, signal_type: str | None, source: str | None) -> None:
        with self._lock:
            self.last_signal = {
                "type": signal_type,
                "ts": _now().isoformat(),
                "source": source,
            }

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            now = _now()
            uptime_seconds = (now - self.started_at).total_seconds()
            return {
                "started_at": self.started_at.isoformat(),
                "last_tick_at": self.last_tick_at.isoformat() if self.last_tick_at else None,
                "tick_count": self.tick_count,
                "current_state": dict(self.current_state),
                "last_signal": dict(self.last_signal),
                "uptime_seconds": uptime_seconds,
            }


_RUNTIME = AgentRuntimeState()


def get_runtime() -> AgentRuntimeState:
    return _RUNTIME
