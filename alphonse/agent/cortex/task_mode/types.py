from __future__ import annotations

from typing import Any, Literal, TypedDict


class NextStepProposal(TypedDict, total=False):
    kind: Literal["call_tool"]
    tool_name: str
    args: dict[str, Any]
    acceptance_criteria: list[str]


class CurrentPlanStep(TypedDict, total=False):
    step_id: str
    tool_call: NextStepProposal
    planner_intent: str


class CheckDecision(TypedDict, total=False):
    route: Literal["direct_reply", "tool_plan", "clarify"]
    intent: str
    confidence: float
    reply_text: str
    clarify_question: str
    acceptance_criteria: list[str]
    parse_ok: bool
    retried: bool
    invalid_json_fallback: bool


class ValidationResult(TypedDict, total=False):
    ok: bool
    executable: bool
    reason: str | None


class TraceEvent(TypedDict, total=False):
    type: str
    summary: str
    correlation_id: str | None
