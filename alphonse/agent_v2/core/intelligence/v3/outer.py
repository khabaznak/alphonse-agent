"""Outer hierarchical CAPD review, routing, and final response boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.intelligence.acceptance_contract import apply_status_patch
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseOutcome, PhaseStatus, SideEffectClass, TacticalState

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
        review = review_phase(task, state, outcome, context)
        context.emit_activity(
            phase=ImprovementPhase.CHECK,
            label="phase reviewed",
            message=review.reason,
            progress={"phase_id": review.phase_id, "phase_review_status": review.status.value},
        )
        system_one_review = task.metadata.get("system_one_review")
        system_one_review = system_one_review if isinstance(system_one_review, dict) else {}
        decision = decide_next_action(
            review,
            recommended_route=str(system_one_review.get("recommended_route") or ""),
            recommendation_confident=bool(system_one_review.get("route_confident")),
        )
        context.emit_activity(
            phase=ImprovementPhase.ACT,
            label="strategy selected",
            message=decision.reason,
            progress={"phase_id": review.phase_id, "strategic_action": decision.action.value},
        )
        task.metadata["v3_phase_review"] = _review_dict(review)
        task.metadata["v3_strategic_decision"] = {"action": decision.action.value, "reason": decision.reason}
        if decision.action == StrategicAction.COMPLETE:
            message = generate_verified_response(task, state, review, context)
            task.metadata["prepared_user_response"] = {
                "source": "v3_final_response",
                "phase_id": state.phase.phase_id,
                "message": message,
            }
            task.status = "completed"
            task.outcome = {
                "status": "success",
                "reason": "Verified phase evidence satisfies all required acceptance criteria.",
                "phase_id": state.phase.phase_id,
            }
            task.metadata["v3_route"] = "respond_and_end"
        elif decision.action == StrategicAction.CONTINUE:
            task.metadata["v3_route"] = "plan_next_phase"
        elif decision.action == StrategicAction.REPLAN:
            task.metadata["v3_route"] = "strategic_replan"
        elif decision.action == StrategicAction.ASK_USER:
            task.status = "waiting_user"
            task.metadata["v3_route"] = "ask_user"
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


def generate_verified_response(
    task: "TaskState",
    state: TacticalState,
    review: PhaseReview,
    context: "CoreLoopContext",
) -> str:
    fallback = f"Listo. {state.phase.objective.rstrip('.')} quedó verificado."
    if context.inference is None:
        return fallback
    evidence = [
        item for item in state.evidence.entries
        if str(item.get("evidence_ref") or "") in set(review.evidence_refs)
    ][-6:]
    prompt = (
        "Write one concise, warm user-facing completion response using only the verified facts below. "
        "Do not claim unobserved effects, tests, or unrelated changes. Do not call tools.\n\n"
        f"User goal: {task.goal}\n"
        f"Verified phase objective: {state.phase.objective}\n"
        f"Review reason: {review.reason}\n"
        f"Verified evidence: {json.dumps(evidence, ensure_ascii=False, default=str)}"
    )
    result = context.inference.generate_markdown(
        InferenceRequest(
            prompt=prompt,
            purpose=InferencePurpose.FINAL_RESPONSE,
            project_id=task.project_id,
            user=task.user,
            task_id=task.task_id,
            tools=(),
            metadata={"phase_id": state.phase.phase_id},
            cancel_checker=context.is_cancelled if context.cancellation_checker is not None else None,
        )
    )
    return str(result.content or "").strip() or fallback


def _review_acceptance_statuses(
    task: "TaskState",
    state: TacticalState,
    evidence_refs: tuple[str, ...],
    context: "CoreLoopContext",
) -> None:
    if context.system_one is not None:
        try:
            system_one_result = context.system_one.evaluate(
                contract=task.ensure_acceptance_contract(),
                phase=state.phase.to_dict(),
                evidence=state.evidence.to_dict(),
            )
        except Exception as exc:
            task.metadata["system_one_review"] = {
                "status": "fallback",
                "error": f"{type(exc).__name__}:{str(exc)[:300]}",
            }
            context.emit_telemetry({
                "event": "system_one_review",
                "task_id": task.task_id,
                "phase_id": state.phase.phase_id,
                "status": "fallback",
                "error_type": type(exc).__name__,
            })
        else:
            metadata = system_one_result.to_metadata()
            if not system_one_result.ambiguous_criterion_ids:
                contract, rejected = apply_status_patch(
                    task.ensure_acceptance_contract(),
                    {"updates": list(system_one_result.updates)},
                    valid_evidence_refs=set(evidence_refs),
                )
                task.acceptance_contract = contract
                task.sync_acceptance_criteria_view()
                task.metadata["v3_phase_review_rejections"] = rejected
                task.metadata["system_one_review"] = {"status": "used", **metadata}
                context.emit_telemetry({
                    "event": "system_one_review",
                    "task_id": task.task_id,
                    "phase_id": state.phase.phase_id,
                    "status": "used",
                    "recommended_route": system_one_result.recommended_route,
                    "route_confident": system_one_result.route_confident,
                    "duration_ms": system_one_result.duration_ms,
                    "model": system_one_result.model,
                    "usage": dict(system_one_result.usage),
                })
                return
            task.metadata["system_one_review"] = {"status": "ambiguous_fallback", **metadata}
            context.emit_telemetry({
                "event": "system_one_review",
                "task_id": task.task_id,
                "phase_id": state.phase.phase_id,
                "status": "ambiguous_fallback",
                "ambiguous_criterion_count": len(system_one_result.ambiguous_criterion_ids),
                "duration_ms": system_one_result.duration_ms,
                "model": system_one_result.model,
                "usage": dict(system_one_result.usage),
            })
    if context.inference is None:
        return
    prompt = (
        "Evaluate the immutable acceptance contract against the complete phase evidence. "
        "Return status updates only; do not redefine criteria. Satisfied criteria require one of the supplied evidence refs.\n\n"
        f"Contract: {json.dumps(task.ensure_acceptance_contract(), ensure_ascii=False)}\n"
        f"Phase: {json.dumps(state.phase.to_dict(), ensure_ascii=False)}\n"
        f"Evidence: {json.dumps(state.evidence.to_dict(), ensure_ascii=False)}\n"
        f"Valid evidence refs: {json.dumps(evidence_refs)}"
    )
    result = context.inference.generate_json(
        InferenceRequest(
            prompt=prompt,
            purpose=InferencePurpose.PHASE_REVIEW,
            project_id=task.project_id,
            user=task.user,
            task_id=task.task_id,
            tools=(),
            metadata={"phase_id": state.phase.phase_id},
            cancel_checker=context.is_cancelled if context.cancellation_checker is not None else None,
        )
    )
    if not isinstance(result.json_value, dict):
        return
    contract, rejected = apply_status_patch(
        task.ensure_acceptance_contract(), result.json_value, valid_evidence_refs=set(evidence_refs)
    )
    task.acceptance_contract = contract
    task.sync_acceptance_criteria_view()
    task.metadata["v3_phase_review_rejections"] = rejected


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
