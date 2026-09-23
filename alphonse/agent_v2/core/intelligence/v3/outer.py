"""Outer hierarchical CAPD review, routing, and final response boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.intelligence.acceptance_contract import apply_status_patch
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseOutcome, PhaseStatus, SideEffectClass, TacticalState
from alphonse.agent_v2.system_one import SystemOneUnavailableError

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState


class PhaseReviewStatus(str, Enum):
    PHASE_VERIFIED_TASK_COMPLETE = "phase_verified_task_complete"
    PHASE_VERIFIED_TASK_INCOMPLETE = "phase_verified_task_incomplete"
    PHASE_BLOCKED = "phase_blocked"
    WAITING_USER = "waiting_user"
    VERIFICATION_FAILED = "verification_failed"
    CANCELLED = "cancelled"


class StrategicAction(str, Enum):
    COMPLETE = "complete"
    CONTINUE = "continue"
    REPLAN = "replan"
    ASK_USER = "ask_user"
    FAIL = "fail"


@dataclass(frozen=True)
class PhaseReview:
    phase_id: str
    status: PhaseReviewStatus
    reason: str
    evidence_refs: tuple[str, ...] = ()
    violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class StrategicDecision:
    action: StrategicAction
    reason: str


class V3OuterController:
    """Review a completed phase and choose the next outer transition."""

    def review_and_route(
        self,
        task: "TaskState",
        state: TacticalState,
        outcome: PhaseOutcome,
        context: "CoreLoopContext",
    ) -> tuple[PhaseReview, StrategicDecision]:
        acceptance_before = _acceptance_progress_signature(task)
        review = review_phase(task, state, outcome, context)
        context.emit_activity(
            phase=ImprovementPhase.CHECK,
            label="phase reviewed",
            message=review.reason,
            progress={"phase_id": review.phase_id, "phase_review_status": review.status.value},
        )
        act_directive = task.metadata.get("act_directive")
        act_directive = act_directive if isinstance(act_directive, dict) else {}
        if act_directive.get("closure_only") and task.has_prepared_user_response():
            desired = str(act_directive.get("terminal_outcome") or "failed")
            decision = StrategicDecision(
                StrategicAction.FAIL if desired == "failed" else StrategicAction.COMPLETE,
                "Act's bounded final explanation was delivered; closing with the previously selected outcome.",
            )
            task.status = "failed" if desired == "failed" else "completed"
            task.outcome = {"status": desired, "reason": str(act_directive.get("reason") or decision.reason)}
            task.metadata["v3_route"] = "end"
            task.metadata["v3_strategic_decision"] = {"action": decision.action.value, "reason": decision.reason}
            return review, decision
        decision, recommendation = _recommend_act(task, state, outcome, review, context)
        task.metadata["system_one_act_recommendation"] = recommendation.to_metadata()
        decision = _apply_no_progress_guard(
            task,
            review,
            decision,
            acceptance_changed=acceptance_before != _acceptance_progress_signature(task),
        )
        if (
            decision.action == StrategicAction.FAIL
            and review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE
            and recommendation.action != "fail_explain"
            and recommendation.answers.get("closure_explanation_is_warranted", 0.0) >= float(getattr(recommendation, "yes_threshold", 0.8))
        ):
            task.metadata["act_directive"] = {
                "action": "fail_explain", "closure_only": True, "response_required": True,
                "terminal_outcome": "failed", "reason": decision.reason,
            }
            task.metadata["v3_route"] = "strategic_replan"
            task.metadata["v3_phase_review"] = _review_dict(review)
            task.metadata["v3_strategic_decision"] = {"action": "fail", "reason": decision.reason}
            return review, decision
        context.emit_activity(
            phase=ImprovementPhase.ACT,
            label="strategy selected",
            message=decision.reason,
            progress={"phase_id": review.phase_id, "strategic_action": decision.action.value},
        )
        task.metadata["v3_phase_review"] = _review_dict(review)
        task.metadata["v3_strategic_decision"] = {"action": decision.action.value, "reason": decision.reason}
        if decision.action == StrategicAction.COMPLETE:
            if not task.has_prepared_user_response():
                task.metadata["act_directive"] = {"action": "complete", "response_required": True}
                task.metadata["v3_route"] = "plan_next_phase"
                return review, StrategicDecision(StrategicAction.CONTINUE, "Plan must prepare the final user-facing response before task completion.")
            task.status = "completed"
            task.outcome = {
                "status": "success",
                "reason": "Verified phase evidence satisfies all required acceptance criteria.",
                "phase_id": state.phase.phase_id,
            }
            task.metadata["v3_route"] = "respond_and_end"
        elif decision.action == StrategicAction.CONTINUE:
            if str(task.metadata.get("system_one_act_recommendation", {}).get("action") or "") == "fail_explain":
                task.metadata["act_directive"] = {
                    "action": "fail_explain", "closure_only": True, "response_required": True,
                    "terminal_outcome": "failed", "reason": recommendation.rationale,
                }
                task.metadata["v3_route"] = "strategic_replan"
            else:
                task.metadata["v3_route"] = "plan_next_phase"
        elif decision.action == StrategicAction.REPLAN:
            task.metadata["act_directive"] = {"action": "replan", "rationale": decision.reason}
            task.metadata["v3_route"] = "strategic_replan"
        elif decision.action == StrategicAction.ASK_USER:
            task.metadata["act_directive"] = {"action": "ask_user", "rationale": decision.reason}
            task.metadata["v3_route"] = "strategic_replan"
        elif recommendation.action == "fail_explain":
            task.metadata["act_directive"] = {
                "action": "fail_explain", "closure_only": True, "response_required": True,
                "terminal_outcome": "failed", "reason": recommendation.rationale,
            }
            task.metadata["v3_route"] = "strategic_replan"
        else:
            task.status = "failed"
            task.outcome = {"status": "failure", "reason": decision.reason, "phase_id": state.phase.phase_id}
            task.metadata["v3_route"] = "end"
        return review, decision


def review_phase(
    task: "TaskState",
    state: TacticalState,
    outcome: PhaseOutcome,
    context: "CoreLoopContext",
) -> PhaseReview:
    violations = _invariant_violations(state)
    evidence_refs = _successful_evidence_refs(state)
    if violations:
        return PhaseReview(
            state.phase.phase_id,
            PhaseReviewStatus.VERIFICATION_FAILED,
            "Phase effects exceeded authorization or contradicted verification invariants.",
            evidence_refs,
            tuple(violations),
        )
    if outcome.status == PhaseStatus.CANCELLED:
        return PhaseReview(state.phase.phase_id, PhaseReviewStatus.CANCELLED, outcome.reason, evidence_refs)
    if outcome.status == PhaseStatus.WAITING_USER:
        return PhaseReview(state.phase.phase_id, PhaseReviewStatus.WAITING_USER, outcome.reason, evidence_refs)
    if outcome.status in {PhaseStatus.BLOCKED, PhaseStatus.BUDGET_EXHAUSTED}:
        return PhaseReview(state.phase.phase_id, PhaseReviewStatus.PHASE_BLOCKED, outcome.reason, evidence_refs)
    if outcome.status != PhaseStatus.PHASE_COMPLETE:
        return PhaseReview(
            state.phase.phase_id,
            PhaseReviewStatus.VERIFICATION_FAILED,
            f"Unexpected nonterminal phase outcome: {outcome.status.value}.",
            evidence_refs,
        )
    _review_acceptance_statuses(task, state, evidence_refs, context)
    if task.acceptance_criteria_all_complete():
        return PhaseReview(
            state.phase.phase_id,
            PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE,
            "All required acceptance criteria are supported by cumulative phase evidence.",
            evidence_refs,
        )
    return PhaseReview(
        state.phase.phase_id,
        PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE,
        "The phase completed, but one or more required task outcomes remain unmet.",
        evidence_refs,
    )


def decide_next_action(
    review: PhaseReview,
    *,
    recommended_route: str = "",
    recommendation_confident: bool = False,
) -> StrategicDecision:
    """Apply an advisory System One route without relaxing deterministic gates."""
    if review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE:
        if recommendation_confident and recommended_route in {"continue", "replan", "ask_user", "fail"}:
            return StrategicDecision(
                StrategicAction.REPLAN,
                f"System One advised {recommended_route}; completion was conservatively withheld for strategic review.",
            )
        return StrategicDecision(StrategicAction.COMPLETE, review.reason)
    if review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE:
        if recommendation_confident and recommended_route == "replan":
            return StrategicDecision(StrategicAction.REPLAN, "System One recommended replanning the incomplete task.")
        return StrategicDecision(StrategicAction.CONTINUE, review.reason)
    if review.status == PhaseReviewStatus.WAITING_USER:
        return StrategicDecision(StrategicAction.ASK_USER, review.reason)
    if review.status in {PhaseReviewStatus.PHASE_BLOCKED, PhaseReviewStatus.VERIFICATION_FAILED}:
        return StrategicDecision(StrategicAction.REPLAN, review.reason)
    return StrategicDecision(StrategicAction.FAIL, review.reason)


def _recommend_act(task, state, outcome, review: PhaseReview, context):
    if context.system_one is None or not callable(getattr(context.system_one, "recommend_act", None)):
        raise SystemOneUnavailableError("act_recommendation_unavailable")
    # Cancellation and authorization violations are deterministic hard stops,
    # not candidates for probabilistic override.
    if review.status == PhaseReviewStatus.CANCELLED:
        from alphonse.agent_v2.system_one import SystemOneActRecommendation
        return StrategicDecision(StrategicAction.FAIL, review.reason), SystemOneActRecommendation(
            "fail", 1.0, True, review.reason,
        )
    check_verdict = (
        "success" if review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE else
        "wip" if review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE else "failure"
    )
    input_state = {
        "goal": task.goal,
        "task_id": task.task_id,
        "check_verdict": check_verdict,
        "check_reason": review.reason,
        "acceptance_contract": task.ensure_acceptance_contract(),
        "strategic_plan": state.phase.to_dict(),
        "phase_outcome": outcome.to_dict(),
        "complete_tool_execution_log": state.to_dict(),
        "task_context": {
            "recent_conversation": task.recent_conversation_md,
            "facts": task.facts_md,
            "user_constraints": task.metadata.get("user_constraints", {}),
            "failure_reason": task.metadata.get("failure_reason", ""),
            "plan_call_exception_count": task.count_plan_call_exceptions(),
            "consecutive_no_progress_phases": task.metadata.get("v3_consecutive_no_progress_phases", 0),
        },
    }
    try:
        recommendation = context.system_one.recommend_act(state=input_state)
    except Exception as exc:
        raise SystemOneUnavailableError(f"act_recommendation:{type(exc).__name__}") from exc
    action = str(recommendation.action)
    answers = recommendation.answers
    yes_threshold = float(getattr(recommendation, "yes_threshold", 0.8))
    if action == "complete" and review.status != PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE:
        action = "continue" if answers.get("continuation_is_worthwhile", 0.0) >= yes_threshold else "replan"
    elif action == "continue" and answers.get("continuation_is_worthwhile", 0.0) < yes_threshold:
        action = "fail_explain" if answers.get("closure_explanation_is_warranted", 0.0) >= yes_threshold else "fail"
    elif action == "ask_user" and answers.get("user_input_can_unblock", 0.0) < yes_threshold:
        action = "replan" if answers.get("continuation_is_worthwhile", 0.0) >= yes_threshold else "fail_explain"
    elif action == "fail_explain" and answers.get("closure_explanation_is_warranted", 0.0) < yes_threshold:
        action = "fail"
    elif not recommendation.confident and action not in {"fail", "fail_explain"}:
        action = "replan"
    rationale = recommendation.rationale
    mapping = {
        "complete": StrategicAction.COMPLETE, "continue": StrategicAction.CONTINUE,
        "replan": StrategicAction.REPLAN, "ask_user": StrategicAction.ASK_USER,
        "fail_explain": StrategicAction.FAIL, "fail": StrategicAction.FAIL,
    }
    if action not in mapping:
        raise SystemOneUnavailableError("act_recommendation_action_invalid")
    return StrategicDecision(mapping[action], rationale), recommendation


def _apply_no_progress_guard(
    task: "TaskState",
    review: PhaseReview,
    decision: StrategicDecision,
    *,
    acceptance_changed: bool,
) -> StrategicDecision:
    if review.status != PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE:
        task.metadata["v3_consecutive_no_progress_phases"] = 0
        return decision
    count = 0 if acceptance_changed else int(task.metadata.get("v3_consecutive_no_progress_phases") or 0) + 1
    task.metadata["v3_consecutive_no_progress_phases"] = count
    if count < 3:
        return decision
    return StrategicDecision(
        StrategicAction.FAIL,
        "Three consecutive completed phases made no acceptance-criteria progress; stopping to prevent repeated work.",
    )


def _acceptance_progress_signature(task: "TaskState") -> tuple[tuple[str, str, tuple[str, ...]], ...]:
    return tuple(
        (
            str(item.get("id") or ""),
            str(item.get("status") or "pending"),
            tuple(str(ref) for ref in item.get("evidence_refs") or []),
        )
        for item in task.ensure_acceptance_contract().get("criteria") or []
        if isinstance(item, dict) and item.get("superseded") is not True
    )


def _review_acceptance_statuses(
    task: "TaskState",
    state: TacticalState,
    evidence_refs: tuple[str, ...],
    context: "CoreLoopContext",
) -> None:
    if context.system_one is None or not hasattr(context.system_one, "evaluate"):
        raise SystemOneUnavailableError("acceptance_review_unavailable")
    try:
        system_one_result = context.system_one.evaluate(
            contract=task.ensure_acceptance_contract(),
            phase=state.phase.to_dict(),
            evidence=state.evidence.to_dict(),
        )
    except Exception as exc:
        raise SystemOneUnavailableError(f"acceptance_review:{type(exc).__name__}") from exc
    metadata = system_one_result.to_metadata()
    ambiguous_ids = set(system_one_result.ambiguous_criterion_ids)
    confident_updates = [
        dict(update) for update in system_one_result.updates
        if str(update.get("criterion_id") or "") not in ambiguous_ids
    ]
    rejected: list[str] = []
    if confident_updates:
        contract, rejected = apply_status_patch(
            task.ensure_acceptance_contract(),
            {"updates": confident_updates},
            valid_evidence_refs=set(evidence_refs),
        )
        task.acceptance_contract = contract
        task.sync_acceptance_criteria_view()
    task.metadata["v3_phase_review_rejections"] = rejected
    if not ambiguous_ids:
        task.metadata["system_one_review"] = {"status": "used", **metadata}
        context.emit_telemetry({
            "event": "system_one_review",
            "task_id": task.task_id,
            "phase_id": state.phase.phase_id,
            "status": "used",
            "duration_ms": system_one_result.duration_ms,
            "model": system_one_result.model,
            "usage": dict(system_one_result.usage),
        })
        return
    task.metadata["system_one_review"] = {
        "status": "partial_fallback" if confident_updates else "ambiguous_fallback",
        **metadata,
    }
    context.emit_telemetry({
        "event": "system_one_review",
        "task_id": task.task_id,
        "phase_id": state.phase.phase_id,
        "status": "partial_fallback" if confident_updates else "ambiguous_fallback",
        "ambiguous_criterion_count": len(system_one_result.ambiguous_criterion_ids),
        "duration_ms": system_one_result.duration_ms,
        "model": system_one_result.model,
        "usage": dict(system_one_result.usage),
    })
    # Ambiguous evidence remains unresolved. Check does not call the inference
    # model to break ties; Act receives that uncertainty in its Jev state.


def _successful_evidence_refs(state: TacticalState) -> tuple[str, ...]:
    return tuple(
        str(item.get("evidence_ref"))
        for item in state.evidence.entries
        if str(item.get("status") or "") == "success" and item.get("evidence_ref")
    )


def _invariant_violations(state: TacticalState) -> list[str]:
    allowed_paths = set(state.phase.mutation_scope.allowed_paths)
    subgoals = {item.subgoal_id: item for item in state.phase.subgoals}
    violations: list[str] = []
    for entry in state.evidence.entries:
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        affected = result.get("affected_paths")
        if isinstance(affected, list):
            subgoal = subgoals.get(str(entry.get("subgoal_id") or ""))
            effects = set(subgoal.allowed_side_effects) if subgoal is not None else set()
            if SideEffectClass.PROJECT_MUTATION in effects:
                unauthorized = [str(path) for path in affected if str(path) not in allowed_paths]
                if unauthorized:
                    violations.append(f"unauthorized_affected_paths:{','.join(unauthorized)}")
            elif effects & {SideEffectClass.EXTERNAL_REVERSIBLE, SideEffectClass.EXTERNAL_IRREVERSIBLE}:
                if not state.phase.mutation_scope.allow_external_effects:
                    violations.append(f"external_effect_not_authorized:{entry.get('evidence_ref') or '(unknown)'}")
            else:
                violations.append(f"affected_paths_without_side_effect_authorization:{entry.get('evidence_ref') or '(unknown)'}")
        verification = result.get("verification")
        if isinstance(verification, dict) and verification.get("status") not in {None, "verified"}:
            violations.append(f"verification_not_verified:{entry.get('evidence_ref') or '(unknown)'}")
    return violations


def _review_dict(review: PhaseReview) -> dict[str, Any]:
    return {
        "phase_id": review.phase_id,
        "status": review.status.value,
        "reason": review.reason,
        "evidence_refs": list(review.evidence_refs),
        "violations": list(review.violations),
    }
