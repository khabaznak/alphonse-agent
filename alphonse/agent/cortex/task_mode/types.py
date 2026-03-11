from __future__ import annotations

from typing import Any, Literal, TypedDict


class AcceptanceCriterion(TypedDict, total=False):
    id: str
    text: str
    status: Literal["pending", "satisfied"]
    evidence_refs: list[str]
    created_by_case: Literal["new_request", "execution_review", "task_resumption"]


class CriteriaUpdate(TypedDict, total=False):
    op: Literal["append", "mark_satisfied"]
    criterion_id: str
    text: str
    evidence_refs: list[str]


class JudgeVerdict(TypedDict, total=False):
    kind: Literal["conversation", "plan", "mission_success", "mission_failed"]
    case_type: Literal["new_request", "execution_review", "task_resumption"]
    reason: str
    confidence: float
    criteria_updates: list[CriteriaUpdate]
    evidence_refs: list[str]
    failure_class: str | None


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
