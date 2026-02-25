from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from alphonse.agent.nervous_system.senses.clock import get_time_now
from alphonse.agent.tools.base import ToolResult


@dataclass(frozen=True)
class ClockTool:
    def execute(self, *, timezone_name: str | None = None, **_: object) -> ToolResult:
        tz = str(timezone_name or "").strip() or "UTC"
        now = get_time_now(tz)
        return {
            "status": "ok",
            "result": {
                "time": now.isoformat(),
                "timezone": tz,
            },
            "error": None,
            "metadata": {"tool": "get_time"},
        }

    def get_time(self) -> datetime:
        return get_time_now()

    # Backward compatibility
    def current_time(self, timezone_name: str) -> datetime:
        return get_time_now(timezone_name)
