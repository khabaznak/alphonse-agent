from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, TypedDict


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
    status: str
    result: Any
    error: dict[str, Any] | None
    metadata: dict[str, Any]


class ToolProtocol(Protocol):
    def execute(self, **kwargs: Any) -> ToolResult: ...


class ToolContractError(Exception):
    pass


def ensure_tool_result(*, tool_key: str, value: Any) -> ToolResult:
    if not isinstance(value, dict):
        raise ToolContractError(f"{tool_key}: result must be an object")
    if set(value.keys()) != {"status", "result", "error", "metadata"}:
        raise ToolContractError(f"{tool_key}: result keys must be status,result,error,metadata")
    status = str(value.get("status") or "").strip().lower()
    if status not in {"ok", "failed"}:
        raise ToolContractError(f"{tool_key}: status must be ok|failed")
    error = value.get("error")
    if status == "ok":
        if error is not None:
            raise ToolContractError(f"{tool_key}: error must be null when status=ok")
    else:
        if not isinstance(error, dict):
            raise ToolContractError(f"{tool_key}: error must be an object when status=failed")
        code = str(error.get("code") or "").strip()
        message = str(error.get("message") or "").strip()
        if not code or not message:
            raise ToolContractError(f"{tool_key}: failed error requires code/message")
    metadata = value.get("metadata")
    if not isinstance(metadata, dict):
        raise ToolContractError(f"{tool_key}: metadata must be an object")
    out: ToolResult = {
        "status": status,
        "result": value.get("result"),
        "error": error if isinstance(error, dict) else None,
        "metadata": dict(metadata),
    }
    return out
