"""Thread-safe, age-bounded storage for transient runtime events."""
from datetime import datetime, timedelta, timezone
from dataclasses import replace
from threading import Lock


class TransientEvents:
    def __init__(self):
        self._items = []
        self._lock = Lock()

    def append(self, event):
        with self._lock:
            now = datetime.now(timezone.utc)
            self._items.append((self._clamp_future(event, now), now))

    def extend(self, events):
        with self._lock:
            now = datetime.now(timezone.utc)
            self._items.extend((self._clamp_future(event, now), now) for event in events)

    @staticmethod
    def _clamp_future(event, now):
        try:
            stamp = datetime.fromisoformat(event.occurred_at.replace('Z', '+00:00'))
            if stamp.tzinfo is not None and stamp.utcoffset() is not None and stamp > now:
                return replace(event, occurred_at=now.isoformat())
        except (AttributeError, TypeError, ValueError, OverflowError):
            pass
        return event

    def drain(self):
        with self._lock:
            items, self._items = self._items, []
        return [event for event, _ in items]

    def prune(self, *, now):
        cutoff = now - timedelta(days=30)
        def keep(entry):
            event, received_at = entry
            try:
                occurred = datetime.fromisoformat(event.occurred_at.replace('Z', '+00:00'))
                if occurred.tzinfo is None or occurred.utcoffset() is None:
                    return False
                return received_at >= cutoff and occurred.astimezone(timezone.utc) >= cutoff
            except (AttributeError, TypeError, ValueError, OverflowError):
                return False
        with self._lock:
            self._items[:] = [entry for entry in self._items if keep(entry)]

    def __len__(self):
        with self._lock:
            return len(self._items)

    def __iter__(self):
        with self._lock:
            return iter([event for event, _ in self._items])
