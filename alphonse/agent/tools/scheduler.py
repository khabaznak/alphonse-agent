from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone as dt_timezone
from typing import Any
from zoneinfo import ZoneInfo

from alphonse.agent.nervous_system.timed_store import insert_timed_signal


class SchedulerToolError(Exception):
    def __init__(
        self,
        *,
        code: str,
        message: str,
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code or "scheduler_error")
        self.message = str(message or "Scheduler operation failed")
        self.retryable = bool(retryable)
        self.details = dict(details or {})

    def as_payload(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "details": self.details,
        }


@dataclass(frozen=True)
class SchedulerTool:
    llm_client: Any | None = None

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
            raise SchedulerToolError(
                code="missing_for_whom",
                message="for_whom is required",
                retryable=False,
            )
        if not trigger_expr:
            raise SchedulerToolError(
                code="missing_time",
                message="time is required",
                retryable=False,
            )
        if not reminder_message:
            raise SchedulerToolError(
                code="missing_message",
                message="message is required",
                retryable=False,
            )
        resolved_timezone = _resolve_timezone_name(timezone_name)
        fire_at = _normalize_time_expression_to_iso(
            expression=trigger_expr,
            timezone_name=resolved_timezone,
            llm_client=self.llm_client,
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
            raise SchedulerToolError(
                code="missing_time",
                message="time is required",
                retryable=False,
            )
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
            raise SchedulerToolError(
                code="unsupported_event_trigger",
                message="only time event triggers are supported",
                retryable=False,
                details={"event_trigger_type": trigger_type},
            )
        trigger_time = str(event_trigger.get("time") or "").strip()
        if not trigger_time:
            raise SchedulerToolError(
                code="missing_event_trigger_time",
                message="event trigger time is required",
                retryable=False,
            )
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


def _normalize_time_expression_to_iso(*, expression: str, timezone_name: str, llm_client: Any | None = None) -> str:
    raw = str(expression or "").strip()
    if not raw:
        raise SchedulerToolError(
            code="missing_time_expression",
            message="time is required",
            retryable=False,
        )
    iso_direct = _try_parse_iso(raw, timezone_name=timezone_name)
    if iso_direct is not None:
        return iso_direct
    llm_iso = _normalize_with_llm(
        expression=raw,
        timezone_name=timezone_name,
        llm_client=llm_client,
    )
    if llm_iso is not None:
        return llm_iso
    raise SchedulerToolError(
        code="time_expression_unresolvable",
        message="time expression could not be normalized",
        retryable=True,
        details={"expression": raw, "timezone": timezone_name},
    )


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


def _normalize_with_llm(*, expression: str, timezone_name: str, llm_client: Any | None) -> str | None:
    client = llm_client or _build_scheduler_llm_client()
    if client is None:
        return None
    now_local = datetime.now(ZoneInfo(timezone_name))
    system_prompt = (
        "# Role\n"
        "You convert natural language time expressions into a single ISO timestamp.\n"
        "# Output Contract\n"
        "- Return ONLY one line.\n"
        "- Either a valid ISO-8601 datetime in UTC (offset +00:00) or UNRESOLVABLE.\n"
        "- No markdown, no extra words."
    )
    user_prompt = (
        "## Input\n"
        f"- timezone: {timezone_name}\n"
        f"- now_local: {now_local.isoformat()}\n"
        f"- expression: {expression}\n\n"
        "## Task\n"
        "Convert expression to exact datetime and output UTC ISO only."
    )
    raw = _llm_complete_text(client=client, system_prompt=system_prompt, user_prompt=user_prompt)
    candidate = str(raw or "").strip()
    if not candidate:
        return None
    if candidate.upper() == "UNRESOLVABLE":
        return None
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate.lower().startswith("text"):
            candidate = candidate[4:].strip()
    parsed = _try_parse_iso(candidate, timezone_name=timezone_name)
    return parsed


def _llm_complete_text(*, client: Any, system_prompt: str, user_prompt: str) -> str:
    complete = getattr(client, "complete", None)
    if not callable(complete):
        return ""
    try:
        return str(complete(system_prompt=system_prompt, user_prompt=user_prompt))
    except TypeError:
        try:
            return str(complete(system_prompt, user_prompt))
        except Exception:
            return ""
    except Exception:
        return ""


def _build_scheduler_llm_client() -> Any | None:
    try:
        from alphonse.agent.cognition.providers.factory import build_llm_client

        return build_llm_client()
    except Exception:
        return None
