"""Task-scoped browser session fencing.

This module does not launch Chrome or grant browser access. A caller must resolve
current actor-scoped authorization before each operation and use a transport
which targets the recorded page ID. Session identities are deliberately
process-local: a restart invalidates them instead of guessing which tab to use.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from time import monotonic
from typing import Callable
from uuid import uuid4


class BrowserSessionError(RuntimeError):
    """A sanitized browser session state error."""


class SessionState(str, Enum):
    ACTIVE = "active"
    WAITING_FOR_HUMAN = "waiting_for_human"
    LOST = "lost"
    CLOSED = "closed"


@dataclass(frozen=True)
class BrowserTarget:
    session_id: str
    page_id: int
    generation: int


class BrowserSession:
    """Fence a single task's page and human handoff with current authorization.

    The authorization callback must consult live grants, not a captured decision.
    Callers must serialize transport calls and check the target again after an
    asynchronous call before accepting its result. A timeout or disconnect
    makes the target permanently stale.
    """

    def __init__(self, *, actor_id: str, task_id: str, page_id: int,
                 authorized: Callable[[str, str], bool], deadline: float,
                 clock: Callable[[], float] = monotonic) -> None:
        if not actor_id or not task_id or isinstance(page_id, bool) or not isinstance(page_id, int) or page_id < 0:
            raise ValueError("A task, actor and valid page ID are required")
        self._actor_id = actor_id
        self._task_id = task_id
        self._page_id = page_id
        self._authorized = authorized
        self._deadline = deadline
        self._clock = clock
        self._id = uuid4().hex
        self._state = SessionState.ACTIVE
        self._generation = 0
        self._lock = RLock()

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    def _check(self, actor_id: str, task_id: str) -> None:
        if actor_id != self._actor_id or task_id != self._task_id:
            raise BrowserSessionError("browser_session_denied")
        if self._state in (SessionState.LOST, SessionState.CLOSED):
            raise BrowserSessionError("browser_session_unavailable")
        if self._clock() >= self._deadline:
            self._invalidate(SessionState.LOST)
            raise BrowserSessionError("browser_session_expired")
        if not self._authorized(actor_id, task_id):
            self._invalidate(SessionState.LOST)
            raise BrowserSessionError("browser_session_denied")

    def _invalidate(self, state: SessionState) -> None:
        self._state = state
        self._generation += 1

    def target(self, actor_id: str, task_id: str) -> BrowserTarget:
        with self._lock:
            self._check(actor_id, task_id)
            if self._state != SessionState.ACTIVE:
                raise BrowserSessionError("browser_session_waiting_for_human")
            return BrowserTarget(self._id, self._page_id, self._generation)

    def accept_result(self, actor_id: str, task_id: str, target: BrowserTarget) -> None:
        with self._lock:
            current = self.target(actor_id, task_id)
            if target != current:
                raise BrowserSessionError("browser_session_stale_result")

    def pause_for_human(self, actor_id: str, task_id: str) -> None:
        with self._lock:
            self.target(actor_id, task_id)
            self._invalidate(SessionState.WAITING_FOR_HUMAN)

    def page_id_for_handoff(self, actor_id: str, task_id: str) -> int:
        with self._lock:
            self._check(actor_id, task_id)
            if self._state != SessionState.WAITING_FOR_HUMAN:
                raise BrowserSessionError("browser_session_not_waiting")
            return self._page_id

    def resume(self, actor_id: str, task_id: str, *, page_still_available: bool) -> None:
        with self._lock:
            self._check(actor_id, task_id)
            if self._state != SessionState.WAITING_FOR_HUMAN:
                raise BrowserSessionError("browser_session_not_waiting")
            if not page_still_available:
                self._invalidate(SessionState.LOST)
                raise BrowserSessionError("browser_session_unavailable")
            self._invalidate(SessionState.ACTIVE)

    def disconnect(self) -> None:
        with self._lock:
            self._invalidate(SessionState.LOST)

    def close(self) -> None:
        with self._lock:
            self._invalidate(SessionState.CLOSED)
