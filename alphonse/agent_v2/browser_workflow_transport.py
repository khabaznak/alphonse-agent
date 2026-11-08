"""Fenced adapter between a task's browser page and workflow state.

The backend is supplied by an explicitly enabled browser host. It must serialize
operations on the fenced page. Guarded mutation and verification are unavailable
unless the backend implements their contracts; ordinary click/fill return values
are never promoted to verified evidence. Backends must not log arguments, page
payloads, or transport exceptions containing them; no backend is enabled here.

``fill_if_unchanged`` and ``submit_if_unchanged`` must compare the authorized
observation and dispatch the requested effect as one fenced operation; a false
or ambiguous result is never evidence of success. ``verify_fill`` must inspect
the current page and match the intended field/value to the fresh observation.
``verify_submission`` must independently establish a new result after the
submission and bind it to the intended subject; a click receipt or an old result
is insufficient. ``verify_owned_session`` must establish task ownership without
creating a tab or guessing an existing one. A backend that cannot establish these
properties must omit the method or return false.
"""
from __future__ import annotations

from typing import Any

from alphonse.agent_v2.browser_session import BrowserSession, BrowserSessionError


class FencedBrowserWorkflowTransport:
    def __init__(self, *, session: BrowserSession, backend: Any, actor_id: str, task_id: str) -> None:
        self.session, self.backend = session, backend
        self.actor_id, self.task_id = actor_id, task_id

    def _target(self, actor: str, task: str):
        if actor != self.actor_id or task != self.task_id:
            raise BrowserSessionError("browser_session_denied")
        return self.session.target(actor, task)

    def session_identity(self, actor: str, task: str) -> str:
        target = self._target(actor, task)
        return f"{target.session_id}:{target.page_id}:{target.generation}"

    def attach_owned_session(self, actor: str, task: str) -> bool:
        self._target(actor, task)
        # The host must prove ownership of a session it has already established;
        # a missing capability cannot fall back to adopting a guessed tab.
        return self._call("verify_owned_session", actor, task) is True

    def _call(self, name: str, *args: Any) -> Any:
        target = self._target(self.actor_id, self.task_id)
        method = getattr(self.backend, name, None)
        if not callable(method):
            raise BrowserSessionError("browser_operation_unavailable")
        try:
            result = method(target.page_id, *args)
            self.session.accept_result(self.actor_id, self.task_id, target)
            return result
        except BaseException:
            self.session.disconnect()
            raise

    def observe(self) -> dict[str, Any]:
        # The backend must return a bounded JSON snapshot with target IDs.
        snapshot = self._call("observe")
        if not isinstance(snapshot, dict) or not isinstance(snapshot.get("targets"), list):
            raise BrowserSessionError("browser_observation_invalid")
        return snapshot

    def fill_if_unchanged(self, target: str, value: str, observed: Any) -> bool:
        return self._call("fill_if_unchanged", target, value, observed) is True

    def submit_if_unchanged(self, target: str, observed: Any) -> bool:
        return self._call("submit_if_unchanged", target, observed) is True

    def verify_fill(self, target: str, value: str, observed: Any) -> bool:
        return self._call("verify_fill", target, value, observed) is True

    def verify_submission(self, target: str, before: Any, after: Any) -> bool:
        return self._call("verify_submission", target, before, after) is True
