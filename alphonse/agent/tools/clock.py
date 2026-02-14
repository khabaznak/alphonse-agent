from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from alphonse.agent.nervous_system.senses.clock import get_time_now


@dataclass(frozen=True)
class ClockTool:
    def get_time(self) -> datetime:
        return get_time_now()

    # Backward compatibility
    def current_time(self, timezone_name: str) -> datetime:
        return get_time_now(timezone_name)
