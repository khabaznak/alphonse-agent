from __future__ import annotations

import re
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from dateutil.parser import parse as parse_datetime

from alphonse.agent.cognition.skills.command_plans import (
    CreateReminderPlan,
    ReminderDelivery,
    ReminderMessage,
    ReminderSchedule,
    SendMessagePlan,
)


def interpret_plan(
    *,
    text: str,
    created_by: str | None,
    source: str,
    correlation_id: str,
    timezone: str,
    language: str = "en",
) -> CreateReminderPlan | SendMessagePlan | None:
    lowered = text.lower()
    if "remind" in lowered or "reminder" in lowered:
        trigger_at = infer_trigger_at(text, timezone)
        message = _extract_message(text) or text
        return CreateReminderPlan(
            plan_id=str(uuid.uuid4()),
            kind="create_reminder",
            created_at=datetime.utcnow().isoformat(),
            created_by=created_by,
            source=source,
            correlation_id=correlation_id,
            original_text=text,
            target_person_id=created_by,
            schedule=ReminderSchedule(timezone=timezone, trigger_at=trigger_at, rrule=None),
            message=ReminderMessage(language=language, text=message),
            delivery=ReminderDelivery(preferred_channel_type=source, priority="normal"),
        )
    if lowered.startswith("send "):
        message = _extract_message(text) or text
        return SendMessagePlan(
            plan_id=str(uuid.uuid4()),
            kind="send_message",
            created_at=datetime.utcnow().isoformat(),
            created_by=created_by,
            source=source,
            correlation_id=correlation_id,
            original_text=text,
            target_person_id=created_by,
            message=ReminderMessage(language=language, text=message),
            channel_type=source,
        )
    return None


def infer_trigger_at(text: str, timezone: str) -> str | None:
    if not _has_time_hint(text):
        return None
    return _parse_trigger_at(text, timezone)


def _parse_trigger_at(text: str, timezone: str) -> str:
    tzinfo = ZoneInfo(timezone)
    now = datetime.now(tz=tzinfo)
    parsed = parse_datetime(text, default=now, fuzzy=True)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=tzinfo)
    return parsed.astimezone(tzinfo).isoformat()


def _extract_message(text: str) -> str | None:
    lowered = text.lower()
    if "to" in lowered:
        parts = text.split("to", 1)
        if len(parts) == 2:
            candidate = parts[1].strip()
            candidate = candidate.split(" at ")[0].strip()
            return candidate or None
    return None


def _has_time_hint(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ("today", "tomorrow", "tonight", "am", "pm")):
        return True
    return bool(re.search(r"\b\d{1,2}(:\d{2})?\b", lowered))
