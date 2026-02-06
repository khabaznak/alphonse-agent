from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from alphonse.agent.cognition.planning import (
    AcceptanceCriterion,
    AgentPlan,
    PlanFeedback,
    PlanFeedbackKind,
    PlanStep,
    PlanStepOption,
    PlanStepOptionKind,
    PlanningContext,
    PlanningMode,
    resolve_planning_context,
)
from alphonse.agent.policy.planning_policy import PlanningPolicy
from alphonse.agent.cognition.intent_lifecycle import LifecycleHint


@dataclass(frozen=True)
class PlanningProposal:
    plan: AgentPlan
    context: PlanningContext


def propose_plan(
    *,
    intent: str,
    autonomy_level: float | None = None,
    requested_mode: PlanningMode | None = None,
    draft_steps: list[str] | None = None,
    acceptance_criteria: list[str] | None = None,
    policy: PlanningPolicy | None = None,
    lifecycle_hint: LifecycleHint | None = None,
) -> PlanningProposal:
    policy = policy or PlanningPolicy()
    context = resolve_planning_context(
        autonomy_level=autonomy_level,
        requested_mode=requested_mode,
    )
    if lifecycle_hint is not None and requested_mode is None:
        context = resolve_planning_context(
            autonomy_level=min(context.autonomy_level, lifecycle_hint.autonomy_cap),
            requested_mode=lifecycle_hint.preferred_mode,
        )
    adjusted_autonomy = policy.adjust_autonomy_level(
        context.autonomy_level,
        intent=intent,
    )
    adjusted_mode = policy.adjust_planning_mode(
        context.effective_mode,
        intent=intent,
        autonomy_level=adjusted_autonomy,
    )
    steps = _build_steps(draft_steps or [])
    criteria = _build_criteria(acceptance_criteria)
    if adjusted_mode == PlanningMode.CONTRATO_DE_RESULTADO and criteria is None:
        criteria = _build_criteria(["TODO: define acceptance criteria"])
    plan = AgentPlan(
        intent=intent,
        planning_mode=adjusted_mode,
        autonomy_level=adjusted_autonomy,
        steps=steps,
        acceptance_criteria=criteria,
    )
    if context.user_override:
        plan.feedback_log.append(
            PlanFeedback(
                kind=PlanFeedbackKind.MODE_OVERRIDDEN,
                payload={
                    "requested_mode": context.requested_mode.value,
                    "suggested_mode": context.suggested_mode.value,
                },
            )
        )
    if adjusted_mode != context.effective_mode or adjusted_autonomy != context.autonomy_level:
        plan.feedback_log.append(
            PlanFeedback(
                kind=PlanFeedbackKind.MODE_POLICY_ADJUSTED,
                payload={
                    "original_mode": context.effective_mode.value,
                    "original_autonomy": context.autonomy_level,
                    "adjusted_mode": adjusted_mode.value,
                    "adjusted_autonomy": adjusted_autonomy,
                },
            )
        )
    return PlanningProposal(plan=plan, context=context)


def record_step_feedback(
    plan: AgentPlan,
    *,
    step_id: str,
    option_id: str,
    kind: PlanFeedbackKind,
    payload: dict[str, Any] | None = None,
) -> AgentPlan:
    feedback = PlanFeedback(
        kind=kind,
        step_id=step_id,
        option_id=option_id,
        payload=payload or {},
    )
    plan.feedback_log.append(feedback)
    return plan


def default_step_options() -> list[PlanStepOption]:
    return [
        PlanStepOption(kind=PlanStepOptionKind.ACCEPT, order=0),
        PlanStepOption(kind=PlanStepOptionKind.REJECT, order=1),
        PlanStepOption(kind=PlanStepOptionKind.ALTERNATIVE, order=2),
        PlanStepOption(kind=PlanStepOptionKind.ASK_CLARIFY, order=3),
    ]


def _build_steps(draft_steps: list[str]) -> list[PlanStep]:
    steps: list[PlanStep] = []
    for summary in draft_steps:
        step = PlanStep(
            summary=summary,
            options=default_step_options(),
        )
        step.options = _reindex_options(step.options)
        steps.append(step)
    return steps


def _build_criteria(criteria: list[str] | None) -> list[AcceptanceCriterion] | None:
    if criteria is None:
        return None
    return [AcceptanceCriterion(description=item) for item in criteria]


def _reindex_options(options: list[PlanStepOption]) -> list[PlanStepOption]:
    for idx, option in enumerate(options):
        option.order = idx
    return options
