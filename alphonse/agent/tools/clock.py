from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class ClockTool:
    def current_time(self, timezone_name: str) -> datetime:
        return datetime.now(tz=ZoneInfo(timezone_name))
