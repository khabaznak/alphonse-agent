from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone as dt_timezone
import re
from typing import Any
from zoneinfo import ZoneInfo

from alphonse.agent.nervous_system.timed_store import insert_timed_signal


@dataclass(frozen=True)
class SchedulerTool:
    def create_reminder(
        self,
        *,
        for_whom: str,
        time: str,
        message: str,
        timezone_name: str | None,
        correlation_id: str | None = None,
        from_: str = "assistant",
        channel_target: str | None = None,
    ) -> dict[str, str]:
        whom_raw = str(for_whom or "").strip()
        trigger_expr = str(time or "").strip()
        reminder_message = str(message or "").strip()
        if not whom_raw:
            raise ValueError("for_whom is required")
        if not trigger_expr:
            raise ValueError("time is required")
        if not reminder_message:
            raise ValueError("message is required")
        resolved_timezone = _resolve_timezone_name(timezone_name)
        fire_at = _normalize_time_expression_to_iso(
            expression=trigger_expr,
            timezone_name=resolved_timezone,
        )
        delivery_target = _normalize_delivery_target(
            for_whom=whom_raw,
            channel_target=channel_target,
        )
        schedule_id = self.schedule_reminder_event(
            message=reminder_message,
            to=delivery_target,
            from_=str(from_ or "assistant"),
            event_trigger={
                "type": "time",
                "time": fire_at,
                "original_time_expression": trigger_expr,
            },
            timezone_name=resolved_timezone,
            correlation_id=correlation_id,
        )
        return {
            "reminder_id": schedule_id,
            "fire_at": fire_at,
            "delivery_target": delivery_target,
            "message": reminder_message,
            "original_time_expression": trigger_expr,
        }

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
            "fire_at": trigger_time,
            "delivery_target": to,
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
        event_payload.setdefault("fire_at", trigger_time)
        event_payload.setdefault("delivery_target", str(target) if target is not None else None)
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


def _resolve_timezone_name(timezone_name: str | None) -> str:
    candidate = str(timezone_name or "").strip()
    if not candidate:
        return "America/Mexico_City"
    try:
        ZoneInfo(candidate)
        return candidate
    except Exception:
        return "America/Mexico_City"


def _normalize_delivery_target(*, for_whom: str, channel_target: str | None) -> str:
    value = str(for_whom or "").strip()
    norm = value.lower()
    if norm in {"me", "yo", "current_conversation"}:
        channel = str(channel_target or "").strip()
        if channel:
            return channel
    return value


def _normalize_time_expression_to_iso(*, expression: str, timezone_name: str) -> str:
    raw = str(expression or "").strip()
    if not raw:
        raise ValueError("time is required")
    iso_direct = _try_parse_iso(raw, timezone_name=timezone_name)
    if iso_direct is not None:
        return iso_direct
    rel = _try_parse_relative(raw, timezone_name=timezone_name)
    if rel is not None:
        return rel
    tomorrow = _try_parse_tomorrow_clock(raw, timezone_name=timezone_name)
    if tomorrow is not None:
        return tomorrow
    raise ValueError("time expression could not be normalized")


def _try_parse_iso(raw: str, *, timezone_name: str) -> str | None:
    candidate = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(candidate)
    except Exception:
        return None
    tz = ZoneInfo(timezone_name)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(dt_timezone.utc).isoformat()


def _try_parse_relative(raw: str, *, timezone_name: str) -> str | None:
    lower = raw.lower().strip()
    patterns = [
        r"^in\s+(\d+)\s*(minute|minutes|min|mins)$",
        r"^en\s+(\d+)\s*(minuto|minutos|min)$",
    ]
    minutes: int | None = None
    for pattern in patterns:
        match = re.match(pattern, lower)
        if match:
            minutes = int(match.group(1))
            break
    if minutes is None:
        match = re.match(r"^(\d+)\s*(minute|minutes|min|mins|minuto|minutos)$", lower)
        if match:
            minutes = int(match.group(1))
    if minutes is None:
        return None
    now_local = datetime.now(ZoneInfo(timezone_name))
    fire_local = now_local + timedelta(minutes=minutes)
    return fire_local.astimezone(dt_timezone.utc).isoformat()


def _try_parse_tomorrow_clock(raw: str, *, timezone_name: str) -> str | None:
    lower = raw.lower().strip()
    match = re.match(r"^(manana|mañana|tomorrow)\s*(a las|at)?\s*(\d{1,2})(?::(\d{2}))?$", lower)
    if not match:
        return None
    hour = int(match.group(3))
    minute = int(match.group(4) or "0")
    if hour > 23 or minute > 59:
        return None
    tz = ZoneInfo(timezone_name)
    now = datetime.now(tz)
    tomorrow_date = (now + timedelta(days=1)).date()
    fire_local = datetime(
        year=tomorrow_date.year,
        month=tomorrow_date.month,
        day=tomorrow_date.day,
        hour=hour,
        minute=minute,
        tzinfo=tz,
    )
    return fire_local.astimezone(dt_timezone.utc).isoformat()
