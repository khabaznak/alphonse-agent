from __future__ import annotations

import re
from datetime import datetime
from zoneinfo import ZoneInfo

from dateutil.parser import parse as parse_datetime


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


def _has_time_hint(text: str) -> bool:
    lowered = text.lower()
    if any(token in lowered for token in ("today", "tomorrow", "tonight", "am", "pm")):
        return True
    return bool(re.search(r"\b\d{1,2}(:\d{2})?\b", lowered))
