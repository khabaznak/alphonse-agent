from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ToolExecutionState(str, Enum):
    PLANNED = "planned"
    STARTED = "started"
    EXECUTED = "executed"
    WAITING_USER = "waiting_user"
    FAILED = "failed"


def tool_execution_mark(tool_key: str, state: ToolExecutionState | str) -> str:
    value = state.value if isinstance(state, ToolExecutionState) else str(state).strip().lower()
    key = str(tool_key or "").strip()
    return f"tool::{key}::{value}"


@dataclass(frozen=True)
class ToolExecutionEvent:
    tool: str
    state: ToolExecutionState
    payload: dict[str, Any] = field(default_factory=dict)

    @property
    def mark(self) -> str:
        return tool_execution_mark(self.tool, self.state)

