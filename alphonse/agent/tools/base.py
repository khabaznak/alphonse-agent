from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypedDict

from alphonse.agent.tools.spec import ToolSpec


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


class ToolResult(TypedDict):
    output: Any
    exception: Any | None
    metadata: dict[str, Any]


class ToolProtocol(Protocol):
    canonical_name: str
    capability: str

    def execute(self, **kwargs: Any) -> ToolResult: ...


class ToolContractError(Exception):
    pass


def ensure_tool_result(*, tool_key: str, value: Any) -> ToolResult:
    if not isinstance(value, dict):
        raise ToolContractError(f"{tool_key}: result must be an object")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise ToolContractError(f"{tool_key}: metadata must be an object")
    if "output" not in value or "exception" not in value:
        raise ToolContractError(f"{tool_key}: result keys must include output,exception,metadata")
    return {
        "output": value.get("output"),
        "exception": value.get("exception"),
        "metadata": dict(metadata),
    }


@dataclass(frozen=True)
class ToolDefinition:
    spec: ToolSpec
    executor: ToolProtocol

    def invoke(self, args: dict[str, Any]) -> ToolResult:
        raw = self.executor.execute(**dict(args or {}))
        return ensure_tool_result(tool_key=self.spec.canonical_name, value=raw)
