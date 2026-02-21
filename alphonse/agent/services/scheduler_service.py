from __future__ import annotations

from datetime import datetime, timezone as dt_timezone
from typing import Any

from alphonse.agent.nervous_system.timed_store import insert_timed_signal


class SchedulerService:
    """Pure timed-signal scheduler: store payload + due time."""

    def schedule_event(
        self,
        *,
        trigger_time: str,
        timezone_name: str,
        payload: dict[str, Any] | None = None,
        target: str | None = None,
        origin: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        event_payload = dict(payload or {})
        event_payload.setdefault("created_at", datetime.now(dt_timezone.utc).isoformat())
        event_payload.setdefault("trigger_at", trigger_time)
        event_payload.setdefault("fire_at", trigger_time)
        event_payload.setdefault("delivery_target", str(target) if target is not None else None)
        return insert_timed_signal(
            trigger_at=trigger_time,
            timezone=timezone_name,
            payload=event_payload,
            target=str(target) if target is not None else None,
            origin=origin,
            correlation_id=correlation_id,
        )
