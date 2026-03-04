from __future__ import annotations

from typing import Any, Literal, TypedDict


class NextStepProposal(TypedDict, total=False):
    kind: Literal["call_tool"]
    tool_name: str
    args: dict[str, Any]
    acceptance_criteria: list[str]


class ValidationResult(TypedDict, total=False):
    ok: bool
    executable: bool
    reason: str | None


class TraceEvent(TypedDict, total=False):
    type: str
    summary: str
    correlation_id: str | None
