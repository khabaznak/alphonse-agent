"""Provider-neutral presence projection for optional v2 integrations."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from threading import RLock
from time import monotonic
from typing import Callable, Iterator, Protocol

from alphonse.agent_v2.core.core import CoreActivityEvent
from alphonse.agent_v2.core.io import ChannelAddress
from alphonse.agent_v2.core.io import channel_address_from_metadata
from alphonse.agent_v2.core.messages.queue import QueuedMessage


class PresencePhase(StrEnum):
    ACKNOWLEDGED = "acknowledged"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING_USER = "waiting_user"
    DONE = "done"
    FAILED = "failed"


@dataclass(frozen=True)
class PresenceCapabilities:
    transient_activity: bool = False
    reactions: bool = False
    editable_progress: bool = False


@dataclass(frozen=True)
class PresenceState:
    phase: PresencePhase
    address: ChannelAddress
    task_id: str = ""
    correlation_id: str = ""
    provider_message_id: str = ""
    label: str = ""
    message: str = ""
    started_at: float = field(default_factory=monotonic)

    @property
    def key(self) -> str:
        return self.correlation_id or self.task_id or self.provider_message_id or self.address.channel_target


class PresenceAdapter(Protocol):
    capabilities: PresenceCapabilities

    def start(self, presence: PresenceState) -> None: ...

    def update(self, presence: PresenceState) -> None: ...

    def heartbeat(self, presence: PresenceState) -> None: ...

    def stop(self, presence: PresenceState) -> None: ...


class PresenceProjector:
    """Projects core activity into provider-specific presence adapters."""

    def __init__(self, *, heartbeat_interval_sec: float = 3.0) -> None:
        self.heartbeat_interval_sec = max(0.5, float(heartbeat_interval_sec))
        self._adapters: dict[str, PresenceAdapter] = {}
        self._active: dict[str, PresenceState] = {}
        self._current: ContextVar[PresenceState | None] = ContextVar("v2_presence_current", default=None)
        self._last_heartbeat: dict[str, float] = {}
        self._lock = RLock()

    def register(self, integration_id: str, adapter: PresenceAdapter) -> None:
        key = str(integration_id or "").strip()
        if not key:
            raise ValueError("presence_integration_id_required")
        with self._lock:
            self._adapters[key] = adapter

    def unregister(self, integration_id: str) -> None:
        with self._lock:
            self._adapters.pop(str(integration_id or "").strip(), None)

    @contextmanager
    def processing(self, queued: QueuedMessage | None) -> Iterator[None]:
        address = _address_from_queued(queued)
        state: PresenceState | None = None
        if address is not None:
            state = PresenceState(
                phase=PresencePhase.ACKNOWLEDGED,
                address=address,
                correlation_id=str(queued.message.correlation_id or "").strip() if queued else "",
                provider_message_id=address.provider_message_id,
            )
            self._current.set(state)
            self._active[state.key] = state
            self._dispatch("start", state)
        try:
            yield
        finally:
            if state is not None and self._current.get() is state:
                self.finish(failed=True)
            self._current.set(None)

    def on_activity(self, event: CoreActivityEvent) -> None:
        with self._lock:
            current = self._current.get()
            if current is None:
                return
            phase = _phase_from_activity(event)
            if phase is None or phase == current.phase:
                return
            updated = PresenceState(
                phase=phase,
                address=current.address,
                task_id=current.task_id,
                correlation_id=current.correlation_id,
                provider_message_id=current.provider_message_id,
                label=event.label,
                message="",
                started_at=current.started_at,
            )
            self._current.set(updated)
            self._active[updated.key] = updated
            self._dispatch("update", updated)

    def finish(self, *, failed: bool = False, waiting: bool = False) -> None:
        with self._lock:
            current = self._current.get()
            if current is None:
                return
            phase = PresencePhase.FAILED if failed else PresencePhase.WAITING_USER if waiting else PresencePhase.DONE
            terminal = PresenceState(
                phase=phase,
                address=current.address,
                task_id=current.task_id,
                correlation_id=current.correlation_id,
                provider_message_id=current.provider_message_id,
                label=phase.value,
                started_at=current.started_at,
            )
            self._dispatch("update", terminal)
            self._dispatch("stop", terminal)
            self._active.pop(current.key, None)
            self._last_heartbeat.pop(current.key, None)
            self._current.set(None)

    def heartbeat(self, *, now: float | None = None) -> None:
        current_time = monotonic() if now is None else float(now)
        with self._lock:
            for state in tuple(self._active.values()):
                last = self._last_heartbeat.get(state.key, state.started_at)
                if current_time - last < self.heartbeat_interval_sec:
                    continue
                self._last_heartbeat[state.key] = current_time
                self._dispatch("heartbeat", state)

    def _dispatch(self, method: str, state: PresenceState) -> None:
        adapter = self._adapters.get(state.address.integration_id)
        if adapter is None:
            return
        try:
            getattr(adapter, method)(state)
        except Exception:
            # Presence is advisory and must never affect CAPD or message delivery.
            return


class TuiPresenceAdapter:
    capabilities = PresenceCapabilities(transient_activity=True)

    def __init__(self, sink: Callable[[str], None] | None = None) -> None:
        self.sink = sink

    def start(self, presence: PresenceState) -> None:
        self._emit(presence)

    def update(self, presence: PresenceState) -> None:
        self._emit(presence)

    def heartbeat(self, presence: PresenceState) -> None:
        return

    def stop(self, presence: PresenceState) -> None:
        return

    def _emit(self, presence: PresenceState) -> None:
        if self.sink is not None:
            self.sink(presence.phase.value.capitalize())


def _address_from_queued(queued: QueuedMessage | None) -> ChannelAddress | None:
    if queued is None:
        return None
    address = channel_address_from_metadata(queued.message.metadata)
    if address is not None:
        return address
    return ChannelAddress(
        integration_id="tui",
        provider_key="tui",
        channel_target=queued.message.user,
        alphonse_user_id=queued.message.user,
    )


def _phase_from_activity(event: CoreActivityEvent) -> PresencePhase | None:
    label = str(event.label or "").strip().lower()
    if label in {"deliberating", "thinking", "deciding"}:
        return PresencePhase.THINKING
    if label in {"working", "executing"}:
        return PresencePhase.EXECUTING
    return None
