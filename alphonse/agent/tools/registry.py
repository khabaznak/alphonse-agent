from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from alphonse.agent.tools.clock import ClockTool
from alphonse.agent.tools.scheduler import SchedulerTool


@dataclass
class ToolRegistry:
    _tools: dict[str, Any] = field(default_factory=dict)

    def register(self, key: str, tool: Any) -> None:
        self._tools[str(key)] = tool

    def get(self, key: str) -> Any | None:
        return self._tools.get(str(key))

    def keys(self) -> list[str]:
        return sorted(self._tools.keys())


def build_default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register("clock", ClockTool())
    registry.register("scheduler", SchedulerTool())
    return registry
