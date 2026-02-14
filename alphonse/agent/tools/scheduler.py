from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from typing import Any

from alphonse.agent.nervous_system.timed_store import insert_timed_signal


@dataclass(frozen=True)
class SchedulerTool:
    def create_time_event_trigger(
        self,
        *,
        time: str,
        timezone_name: str | None = None,
    ) -> dict[str, str]:
        _ = timezone_name
        value = str(time or "").strip()
        if not value:
            raise ValueError("time is required")
        return {"type": "time", "time": value}

    def schedule_reminder_event(
        self,
        *,
        message: str,
        to: str,
        from_: str,
        event_trigger: dict[str, Any],
        timezone_name: str,
        correlation_id: str | None = None,
    ) -> str:
        trigger_type = str(event_trigger.get("type") or "").strip().lower()
        if trigger_type != "time":
            raise ValueError("only time event triggers are supported")
        trigger_time = str(event_trigger.get("time") or "").strip()
        if not trigger_time:
            raise ValueError("event trigger time is required")
        payload = {
            "message": message,
            "reminder_text_raw": message,
            "to": to,
            "from": from_,
            "created_at": datetime.now(dt_timezone.utc).isoformat(),
            "trigger_at": trigger_time,
            "event_trigger": event_trigger,
        }
        return self.schedule_event(
            trigger_time=trigger_time,
            timezone_name=timezone_name,
            signal_type="reminder",
            payload=payload,
            target=to,
            origin=from_,
            correlation_id=correlation_id,
        )

    def schedule_event(
        self,
        *,
        trigger_time: str,
        timezone_name: str,
        signal_type: str,
        payload: dict[str, Any] | None = None,
        target: str | None = None,
        origin: str | None = None,
        correlation_id: str | None = None,
    ) -> str:
        event_payload = dict(payload or {})
        event_payload.setdefault("created_at", datetime.now(dt_timezone.utc).isoformat())
        event_payload.setdefault("trigger_at", trigger_time)
        return insert_timed_signal(
            trigger_at=trigger_time,
            timezone=timezone_name,
            signal_type=signal_type,
            payload=event_payload,
            target=str(target) if target is not None else None,
            origin=origin,
            correlation_id=correlation_id,
        )

    # Compatibility helper while reminder payloads are still being migrated.
    def schedule_reminder(
        self,
        *,
        reminder_text: str,
        trigger_time: str,
        chat_id: str,
        channel_type: str,
        actor_person_id: str | None,
        intent_evidence: dict[str, Any],
        correlation_id: str,
        timezone_name: str,
        locale_hint: str | None,
    ) -> str:
        payload = {
            "message": reminder_text,
            "reminder_text_raw": reminder_text,
            "person_id": actor_person_id or chat_id,
            "chat_id": chat_id,
            "origin_channel": channel_type,
            "locale_hint": locale_hint,
            "intent_evidence": intent_evidence,
        }
        return self.schedule_event(
            trigger_time=trigger_time,
            timezone_name=timezone_name,
            signal_type="reminder",
            payload=payload,
            target=str(actor_person_id or chat_id),
            origin=channel_type,
            correlation_id=correlation_id,
        )
