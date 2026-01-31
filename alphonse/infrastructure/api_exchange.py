from __future__ import annotations

import threading
from typing import Any


class ApiExchange:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._responses: dict[str, dict[str, Any]] = {}

    def publish(self, correlation_id: str, payload: dict[str, Any]) -> None:
        with self._condition:
            self._responses[correlation_id] = payload
            self._condition.notify_all()

    def wait(self, correlation_id: str, timeout: float) -> dict[str, Any] | None:
        with self._condition:
            if correlation_id in self._responses:
                return self._responses.pop(correlation_id)
            self._condition.wait(timeout=timeout)
            return self._responses.pop(correlation_id, None)
