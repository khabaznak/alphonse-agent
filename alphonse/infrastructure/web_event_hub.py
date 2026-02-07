from __future__ import annotations

import queue
import threading
import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class _Subscription:
    channel_target: str
    events: queue.Queue[dict[str, Any]]


class WebEventHub:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: dict[str, _Subscription] = {}

    def subscribe(self, channel_target: str) -> str:
        sub_id = str(uuid.uuid4())
        sub = _Subscription(channel_target=channel_target, events=queue.Queue())
        with self._lock:
            self._subs[sub_id] = sub
        return sub_id

    def unsubscribe(self, subscriber_id: str) -> None:
        with self._lock:
            self._subs.pop(subscriber_id, None)

    def publish(self, channel_target: str, payload: dict[str, Any]) -> None:
        with self._lock:
            subscribers = list(self._subs.values())
        for sub in subscribers:
            if sub.channel_target != channel_target:
                continue
            sub.events.put(payload)

    def next_event(self, subscriber_id: str, timeout: float = 15.0) -> dict[str, Any] | None:
        with self._lock:
            sub = self._subs.get(subscriber_id)
        if not sub:
            return None
        try:
            return sub.events.get(timeout=timeout)
        except queue.Empty:
            return None


web_event_hub = WebEventHub()
