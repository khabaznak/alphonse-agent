"""Volatile, actor/task-scoped browser data handles.

Handles are process-local and expire; they are not authorization grants and may
be persisted only as opaque references. The caller must recheck live access on
both insertion and retrieval. No payload is written to disk or logs here.
"""
from __future__ import annotations

import copy
import json
import secrets
import threading
from dataclasses import dataclass
from time import monotonic
from typing import Any, Callable


class BrowserPrivateDataError(RuntimeError):
    pass


@dataclass
class _Record:
    actor: str
    task: str
    grant_id: str
    payload: Any
    size: int
    expires: float
    category: str
    parent: str | None = None


class BrowserPrivateData:
    def __init__(self, *, max_bytes: int = 1_000_000, max_entries: int = 128,
                 clock: Callable[[], float] = monotonic) -> None:
        if max_bytes <= 0 or max_entries <= 0:
            raise ValueError("browser_private_limit_invalid")
        self._max_bytes, self._max_entries, self._clock = max_bytes, max_entries, clock
        self._entries: dict[str, _Record] = {}
        self._timers: dict[str, threading.Timer] = {}
        self._lock = threading.RLock()

    def put(self, *, actor: str, task: str, grant_id: str, payload: Any, authorized: Callable[[], bool], ttl: float = 300, category: str = "private", parent: str | None = None) -> str:
        if category not in {"private", "identifier", "observation", "binding"} or not all(isinstance(value, str) and value for value in (actor, task, grant_id)) or not 0 < ttl <= 3600 or not authorized():
            raise BrowserPrivateDataError("browser_private_access_denied")
        try:
            serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            value = json.loads(serialized)
        except (TypeError, ValueError, UnicodeError):
            raise BrowserPrivateDataError("browser_private_payload_invalid") from None
        with self._lock:
            self._prune()
            if not authorized():
                raise BrowserPrivateDataError("browser_private_access_denied")
            expires = self._clock() + ttl
            if parent is not None:
                source = self._entries.get(parent)
                if source is None or (source.actor, source.task, source.grant_id, source.category) != (actor, task, grant_id, "observation"):
                    raise BrowserPrivateDataError("browser_private_access_denied")
                expires = min(expires, source.expires)
            if len(self._entries) >= self._max_entries or sum(item.size for item in self._entries.values()) + len(serialized) > self._max_bytes:
                raise BrowserPrivateDataError("browser_private_capacity_exceeded")
            handle = secrets.token_urlsafe(24)
            self._entries[handle] = _Record(actor, task, grant_id, value, len(serialized), expires, category, parent)
            self._schedule_expiry(handle, max(0, expires - self._clock()))
            return handle

    def get(self, *, actor: str, task: str, grant_id: str, handle: str, authorized: Callable[[], bool], expected_parent: str | None = None) -> Any:
        if not isinstance(handle, str):
            raise BrowserPrivateDataError("browser_private_access_denied")
        with self._lock:
            self._prune()
            record = self._entries.get(handle)
            if not authorized() or record is None or record.actor != actor or record.task != task or record.grant_id != grant_id:
                raise BrowserPrivateDataError("browser_private_access_denied")
            if record.parent is not None and record.parent not in self._entries:
                self._remove(handle)
                raise BrowserPrivateDataError("browser_private_access_denied")
            if expected_parent is not None and record.parent != expected_parent:
                raise BrowserPrivateDataError("browser_private_access_denied")
            return copy.deepcopy(record.payload)

    def category(self, *, actor: str, task: str, grant_id: str, handle: str, authorized: Callable[[], bool]) -> str:
        with self._lock:
            self.get(actor=actor, task=task, grant_id=grant_id, handle=handle, authorized=authorized)
            return self._entries[handle].category

    def clear_task(self, *, actor: str, task: str, category: str | None = None) -> None:
        with self._lock:
            for key, record in list(self._entries.items()):
                if (record.actor, record.task) == (actor, task) and (category is None or record.category == category):
                    self._remove(key)

    def clear(self) -> None:
        with self._lock:
            for key in list(self._entries):
                self._remove(key)

    def _remove(self, handle: str) -> None:
        self._entries.pop(handle, None)
        timer = self._timers.pop(handle, None)
        if timer is not None:
            timer.cancel()

    def _schedule_expiry(self, handle: str, delay: float) -> None:
        timer = threading.Timer(delay, self._expire, args=(handle,))
        timer.daemon = True
        self._timers[handle] = timer
        timer.start()

    def _expire(self, handle: str) -> None:
        with self._lock:
            record = self._entries.get(handle)
            if record is None:
                return
            remaining = record.expires - self._clock()
            if remaining > 0:
                self._schedule_expiry(handle, remaining)
            else:
                self._remove(handle)

    def _prune(self) -> None:
        now = self._clock()
        for key, record in list(self._entries.items()):
            if record.expires <= now:
                self._remove(key)
