from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar

from alphonse.agent.nervous_system.senses.clock import get_time_now
from alphonse.agent.tools.base import ToolResult


@dataclass(frozen=True)
class ClockTool:
    canonical_name: ClassVar[str] = "get_time"
    capability: ClassVar[str] = "context"

    def execute(self, *, timezone_name: str | None = None, **_: object) -> ToolResult:
        tz = str(timezone_name or "").strip() or "UTC"
        now = get_time_now(tz)
        return {
            "output": {
                "time": now.isoformat(),
                "timezone": tz,
            },
            "exception": None,
            "metadata": {"tool": "get_time"},
        }

    def get_time(self) -> datetime:
        return get_time_now()

    # Backward compatibility
    def current_time(self, timezone_name: str) -> datetime:
        return get_time_now(timezone_name)
