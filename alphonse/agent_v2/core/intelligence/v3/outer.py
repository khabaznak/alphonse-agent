"""Outer hierarchical CAPD review, routing, and final response boundary."""

from __future__ import annotations

from alphonse.agent_v2.core.context_inventory import source, prompt_receipt

import json
import inspect
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.intelligence.v3.response_state import invalidate_outdated_response
from alphonse.agent_v2.core.tools.registry import ToolExposurePolicy
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.intelligence.acceptance_contract import apply_status_patch
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseOutcome, PhaseStatus, SideEffectClass, TacticalState
from alphonse.agent_v2.system_one import SystemOneUnavailableError
from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision
from alphonse.agent_v2.core.intelligence.v3.projections import enabled as projections_enabled

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
        invalidate_outdated_response(task, state)
        review = review_phase(task, state, outcome, context)
        context.emit_activity(
            phase=ImprovementPhase.CHECK,
            label="phase reviewed",
            message=review.reason,
            progress={"phase_id": review.phase_id, "phase_review_status": review.status.value},
        )
        act_directive = task.metadata.get("act_directive")
        act_directive = act_directive if isinstance(act_directive, dict) else {}
        desired = str(act_directive.get("terminal_outcome") or "failed")
        # Successful closure must still pass the mission-outcome review below.
        closure_allowed = desired == "failed"
        if review.status != PhaseReviewStatus.CANCELLED and closure_allowed and act_directive.get("closure_only") and task.has_prepared_user_response():
            decision = StrategicDecision(
                StrategicAction.FAIL if desired == "failed" else StrategicAction.COMPLETE,
                "Act's final explanation was delivered; closing with the previously selected outcome.",
            )
            task.status = "failed" if desired == "failed" else "completed"
            task.outcome = {"status": desired, "reason": str(act_directive.get("reason") or decision.reason)}
            task.metadata["v3_route"] = "end"
            task.metadata["v3_strategic_decision"] = {"action": decision.action.value, "reason": decision.reason}
            return review, decision
        recommendation = None
        if review.status in {
            PhaseReviewStatus.WAITING_USER,
            PhaseReviewStatus.CANCELLED,
            PhaseReviewStatus.PHASE_BLOCKED,
            PhaseReviewStatus.VERIFICATION_FAILED,
        }:
            decision = decide_next_action(review)
            if review.status == PhaseReviewStatus.WAITING_USER:
                task.status = "waiting_user"
                task.metadata["v3_route"] = "waiting_user"
                task.metadata["v3_phase_review"] = _review_dict(review)
                task.metadata["v3_strategic_decision"] = {"action": decision.action.value, "reason": decision.reason}
                return review, decision
            if review.status == PhaseReviewStatus.CANCELLED:
                task.status = "cancelled"
                task.metadata["v3_route"] = "end"
                task.metadata["v3_phase_review"] = _review_dict(review)
                task.metadata["v3_strategic_decision"] = {"action": decision.action.value, "reason": decision.reason}
                return review, decision
        elif context.system_one is not None and callable(getattr(context.system_one, "recommend_act", None)):
            try:
                decision, recommendation = _recommend_act(task, state, outcome, review, context)
            except SystemOneUnavailableError:
                if (getattr(context.system_one, "requires_mission_outcome_review", False)
                        and review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE):
                    raise
                decision = decide_next_action(review)
        else:
            decision = decide_next_action(review)
        if context.is_cancelled():
            review = PhaseReview(state.phase.phase_id, PhaseReviewStatus.CANCELLED, "Task cancelled.")
            decision = decide_next_action(review)
            task.status = "cancelled"
            task.metadata["v3_route"] = "end"
            task.metadata["v3_phase_review"] = _review_dict(review)
            task.metadata["v3_strategic_decision"] = {"action": decision.action.value, "reason": decision.reason}
            return review, decision
        if recommendation is not None:
            task.metadata["system_one_act_recommendation"] = recommendation.to_metadata()
        if (
            decision.action == StrategicAction.FAIL
            and review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE
            and recommendation is not None
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
            if (
                str(task.metadata.get("system_one_act_recommendation", {}).get("action") or "") == "fail_explain"
                and recommendation is not None
                and review.status != PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE
            ):
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
        elif (
            recommendation is not None
            and recommendation.action == "fail_explain"
            and review.status != PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE
        ):
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
    if context.is_cancelled():
        return PhaseReview(state.phase.phase_id, PhaseReviewStatus.CANCELLED, "Task cancelled.")
    violations = _invariant_violations(task, state)
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
    if outcome.status == PhaseStatus.BLOCKED:
        return PhaseReview(state.phase.phase_id, PhaseReviewStatus.PHASE_BLOCKED, outcome.reason, evidence_refs)
    if outcome.status != PhaseStatus.PHASE_COMPLETE:
        return PhaseReview(
            state.phase.phase_id,
            PhaseReviewStatus.VERIFICATION_FAILED,
            f"Unexpected nonterminal phase outcome: {outcome.status.value}.",
            evidence_refs,
        )
    _review_acceptance_statuses(task, state, evidence_refs, context)
    if context.is_cancelled():
        return PhaseReview(state.phase.phase_id, PhaseReviewStatus.CANCELLED, "Task cancelled.", evidence_refs)
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
        "goal": "" if projections_enabled() else task.goal,
        "task_id": task.task_id,
        "check_verdict": check_verdict,
        "check_reason": review.reason,
        "acceptance_contract": task.ensure_acceptance_contract(),
        "strategic_plan": state.phase.to_dict(),
        "phase_outcome": outcome.to_dict(),
        "complete_tool_execution_log": state.to_dict(),
        "task_context": _task_context(task, context),
        "user_constraints": task.metadata.get("user_constraints", {}),
        "failure_reason": task.metadata.get("failure_reason", ""),
        "plan_call_exception_count": task.count_plan_call_exceptions(),
        "authenticated_followups": list(task.metadata.get("v3_resumption_evidence") or []),
        "available_tool_ids": [tool.tool_id for tool in ToolExposurePolicy().select_tools(
            registry=context.tools, project_id=task.project_id, user=task.user, task=task)] if context.tools is not None else [],
    }
    try:
        recommendation = context.system_one.recommend_act(state=input_state)
    except Exception as exc:
        raise SystemOneUnavailableError(f"act_recommendation:{type(exc).__name__}") from exc
    action = str(recommendation.action)
    recommended_action = action
    answers = recommendation.answers
    yes_threshold = float(getattr(recommendation, "yes_threshold", 0.8))
    mission_fulfilled = answers.get("mission_outcome_is_fulfilled")
    valid_mission_judgment = (type(mission_fulfilled) in {int, float}
                             and math.isfinite(mission_fulfilled) and 0 <= mission_fulfilled <= 1)
    requires_effect = answers.get("mission_requires_persisted_effect", 0.0)
    # A semantic positive cannot manufacture execution evidence out of a reply.
    # Successful operational evidence is necessary, never sufficient: the
    # mission judgment and existing scope/verification checks still apply.
    missing_effect_evidence = (
        type(requires_effect) in {int, float} and math.isfinite(requires_effect)
        and requires_effect >= yes_threshold
        and not any(
            item.get("status") == "success" and item.get("tool_id")
            and item["tool_id"] not in {"native.respond", "native.ask_question"}
            for item in state.evidence.entries
        )
    )
    mission_unfulfilled = valid_mission_judgment and mission_fulfilled < yes_threshold
    required_judgment_missing = (
        getattr(context.system_one, "requires_mission_outcome_review", False) and not valid_mission_judgment
    )
    if action == "complete" and (mission_unfulfilled or missing_effect_evidence or required_judgment_missing):
        action = "replan"
    elif action == "fail_explain" and review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE:
        # A completed phase is not the same as a completed mission. If the
        # phase review confirms that required acceptance criteria remain unmet,
        # do not turn a low continuation score into a user-facing failure while
        # an authorized next phase may still satisfy them. Give Plan a chance
        # to continue the unfinished mission; actual blockers are represented
        # by BLOCKED/VERIFICATION_FAILED reviews and retain the failure route.
        action = "continue"
    elif action == "complete" and review.status != PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE:
        action = "continue" if answers.get("continuation_is_worthwhile", 0.0) >= yes_threshold else "replan"
    elif action == "continue" and answers.get("continuation_is_worthwhile", 0.0) < yes_threshold:
        # A disagreement between Jev's categorical choice and its continuation
        # fuse is uncertainty about the current strategy, not evidence that the
        # mission must fail. Return to Plan for a fresh strategy.
        action = "replan"
    elif action == "ask_user" and answers.get("user_input_can_unblock", 0.0) < yes_threshold:
        action = "replan" if answers.get("continuation_is_worthwhile", 0.0) >= yes_threshold else "fail_explain"
    elif action == "fail_explain" and answers.get("closure_explanation_is_warranted", 0.0) < yes_threshold:
        action = "fail"
    elif not recommendation.confident and action not in {"fail", "fail_explain"}:
        action = "replan"
    rationale = recommendation.rationale
    if action != recommended_action:
        rationale = {
            "continue": "The task is not yet verified complete; Plan must continue with any useful authorized work.",
            "replan": ("The current user request or authenticated follow-up remains unfulfilled despite the acceptance status; plan the remaining authorized work using the available tool registry."
                       if isinstance(mission_fulfilled, (int, float)) and mission_fulfilled < yes_threshold
                       else "Jev's recommendation and resilience checks disagreed; returning to Plan for a fresh strategy."),
            "ask_user": "A concrete user answer is needed before useful work can continue.",
            "fail_explain": "Further work is not worthwhile, and Jev recommends a final evidence-based explanation.",
            "fail": "Jev determined that the mission must stop without another execution phase.",
        }[action]
    mapping = {
        "complete": StrategicAction.COMPLETE, "continue": StrategicAction.CONTINUE,
        "replan": StrategicAction.REPLAN, "ask_user": StrategicAction.ASK_USER,
        "fail_explain": StrategicAction.FAIL, "fail": StrategicAction.FAIL,
    }
    if action not in mapping:
        raise SystemOneUnavailableError("act_recommendation_action_invalid")
    return StrategicDecision(mapping[action], rationale), recommendation


def _review_acceptance_statuses(
    task: "TaskState",
    state: TacticalState,
    evidence_refs: tuple[str, ...],
    context: "CoreLoopContext",
) -> None:
    system_one_result = None
    system_one = context.system_one
    # Validate required context outside provider recovery: budget/cancellation
    # failures must not be mistaken for an unavailable judgment service.
    task_context = _task_context(task, context)
    if system_one is not None and callable(getattr(system_one, "evaluate", None)):
        try:
            system_one_result = _evaluate_system_one_with_context(
                system_one,
                contract=task.ensure_acceptance_contract(),
                phase=state.phase.to_dict(),
                evidence=state.evidence.to_dict(),
                task_context=task_context,
            )
        except Exception:
            system_one_result = None
    if context.is_cancelled():
        return
    if system_one_result is None:
        payload = _system_two_acceptance_review(task, state, context)
        updates = payload.get("updates")
        updates = [dict(item) for item in updates or () if isinstance(item, dict)]
        contract, rejected = apply_status_patch(
            task.ensure_acceptance_contract(), {"updates": updates}, valid_evidence_refs=set(evidence_refs),
        )
        task.acceptance_contract = contract
        task.sync_acceptance_criteria_view()
        task.metadata["v3_phase_review_rejections"] = rejected
        task.metadata["criteria_review_updated"] = bool(updates)
        task.metadata["system_one_review"] = {"status": "fallback", "provider": "system_two"}
        task.append_update("Check used the System Two phase review because System One was unavailable.")
        return
    metadata = system_one_result.to_metadata()
    ambiguous_ids = set(system_one_result.ambiguous_criterion_ids)
    confident_updates = [
        dict(update) for update in system_one_result.updates
        if str(update.get("criterion_id") or "") not in ambiguous_ids
    ]
    focused_rereview = None
    if ambiguous_ids:
        # Re-review just the unresolved criteria against the same cumulative,
        # ordered phase evidence before Check hands uncertainty to Act/Plan.
        # This is bounded to one additional Jev evaluation per phase review.
        current_contract = task.ensure_acceptance_contract()
        focused_contract = {
            **current_contract,
            "criteria": [
                dict(item) for item in current_contract.get("criteria") or []
                if isinstance(item, dict) and str(item.get("id") or "") in ambiguous_ids
            ],
        }
        focused_phase = {
            **state.phase.to_dict(),
            "review_focus": {
                "criterion_ids": sorted(ambiguous_ids),
                "instructions": (
                    "Re-evaluate only these unresolved criteria against all successful evidence in its recorded order. "
                    "Use each evidence item's phase and subgoal identity and the ordered subgoal objectives; "
                    "for criteria requiring a prerequisite before a mutation, verify that the successful prerequisite "
                    "evidence occurs before the mutation evidence. Do not require new tool work if existing evidence suffices."
                ),
            },
        }
        try:
            focused_rereview = _evaluate_system_one_with_context(
                context.system_one,
                contract=focused_contract,
                phase=focused_phase,
                evidence=state.evidence.to_dict(),
                task_context=_task_context(task, context),
            )
        except Exception as exc:
            # A failed focused pass must not erase the original result or
            # prevent the task from following its existing unresolved-evidence
            # route.
            task.metadata["v3_focused_rereview_error"] = type(exc).__name__
        else:
            remaining_ambiguous = set(focused_rereview.ambiguous_criterion_ids)
            confident_updates.extend(
                dict(update) for update in focused_rereview.updates
                if str(update.get("criterion_id") or "") not in remaining_ambiguous
            )
            ambiguous_ids = remaining_ambiguous
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
        review_metadata = {"status": "used", **metadata}
        if focused_rereview is not None:
            review_metadata["focused_rereview"] = focused_rereview.to_metadata()
            review_metadata["status"] = "focused_rereview_resolved"
        task.metadata["system_one_review"] = review_metadata
        context.emit_telemetry({
            "event": "system_one_review",
            "task_id": task.task_id,
            "phase_id": state.phase.phase_id,
            "status": "focused_rereview_resolved" if focused_rereview is not None else "used",
            "duration_ms": system_one_result.duration_ms + (focused_rereview.duration_ms if focused_rereview else 0),
            "model": focused_rereview.model if focused_rereview else system_one_result.model,
            "usage": dict(focused_rereview.usage if focused_rereview else system_one_result.usage),
        })
        return
    if focused_rereview is not None:
        metadata = {
            **metadata,
            "updates": [*metadata.get("updates", []), *focused_rereview.to_metadata().get("updates", [])],
            "ambiguous_criterion_ids": sorted(ambiguous_ids),
            "focused_rereview": focused_rereview.to_metadata(),
        }
    focused_contract = {
        **task.ensure_acceptance_contract(),
        "criteria": [
            dict(item) for item in task.ensure_acceptance_contract().get("criteria") or []
            if isinstance(item, dict) and str(item.get("id") or "") in ambiguous_ids
        ],
    }
    try:
        fallback_payload = _system_two_acceptance_review(task, state, context, contract=focused_contract)
    except SystemOneUnavailableError:
        fallback_payload = {}
    fallback_updates = [
        dict(item) for item in fallback_payload.get("updates") or ()
        if isinstance(item, dict) and str(item.get("criterion_id") or "") in ambiguous_ids
    ]
    if fallback_updates:
        contract, fallback_rejected = apply_status_patch(
            task.ensure_acceptance_contract(), {"updates": fallback_updates},
            valid_evidence_refs=set(evidence_refs),
        )
        task.acceptance_contract = contract
        task.sync_acceptance_criteria_view()
        task.metadata["v3_phase_review_rejections"] = [*rejected, *fallback_rejected]
        task.metadata["system_one_review"] = {"status": "partial_fallback", **metadata}
        task.append_update("Check used a focused System Two review for ambiguous evidence.")
        return
    task.metadata["system_one_review"] = {
        "status": (
            "focused_rereview_unresolved" if focused_rereview is not None
            else "partial_fallback" if confident_updates else "ambiguous_fallback"
        ),
        **metadata,
    }
    context.emit_telemetry({
        "event": "system_one_review",
        "task_id": task.task_id,
        "phase_id": state.phase.phase_id,
        "status": (
            "focused_rereview_unresolved" if focused_rereview is not None
            else "partial_fallback" if confident_updates else "ambiguous_fallback"
        ),
        "ambiguous_criterion_count": len(system_one_result.ambiguous_criterion_ids),
        "duration_ms": system_one_result.duration_ms + (focused_rereview.duration_ms if focused_rereview else 0),
        "model": focused_rereview.model if focused_rereview else system_one_result.model,
        "usage": dict(focused_rereview.usage if focused_rereview else system_one_result.usage),
    })


def _system_two_acceptance_review(
    task: "TaskState",
    state: TacticalState,
    context: "CoreLoopContext",
    *,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if context.inference is None:
        raise SystemOneUnavailableError("acceptance_review_unavailable")
    shared_context = _task_context(task, context) if projections_enabled() else ""
    review_contract = contract or task.ensure_acceptance_contract()
    valid_refs = _successful_evidence_refs(state)
    schema = {
        "type": "object", "additionalProperties": False, "required": ["updates"],
        "properties": {"updates": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["criterion_id", "status", "evidence_refs", "reason"],
            "properties": {
                "criterion_id": {"type": "string", "enum": [str(item["id"]) for item in review_contract.get("criteria", [])]},
                "status": {"type": "string", "enum": ["pending", "satisfied", "blocked"]},
                "evidence_refs": {"type": "array", "items": {"type": "string", **({"enum": list(valid_refs)} if valid_refs else {})}},
                "reason": {"type": "string"},
            },
        }}},
    }
    prompt = (
            "Review the verified evidence against the immutable acceptance criteria. Return JSON with an updates array. "
            "Each update must include criterion_id, status, evidence_refs (an array of valid reference strings), and reason. "
            "Return exactly the supplied JSON Schema; singular evidence_ref is not a supported field. "
            "Do not invent evidence or rewrite criteria. "
            "The only valid evidence_refs are the successful action references listed below. "
            "Image-input receipts (inference-images:...) record delivery only and are not valid outcome evidence references. "
            "When images are attached to this review, inspect them directly to verify image-derived details in a successful "
            "action's output, then cite that action's reference. A receipt alone cannot verify extraction or external effects. "
            "Use the actual image-to-asset mapping supplied with this inference request. If the image or successful action "
            "does not support a criterion, leave it pending. Never require an OCR or Bash call to read attached images.\n\n"
            + (f"Required decision context:\n{shared_context}\n" if shared_context else "") +
            f"Valid evidence references: {json.dumps(valid_refs, ensure_ascii=False)}\n"
            f"Response JSON Schema: {json.dumps(schema, ensure_ascii=False)}\n"
            f"Acceptance contract: {json.dumps(review_contract, ensure_ascii=False)}\n"
            f"Phase: {json.dumps(state.phase.to_dict(), ensure_ascii=False)}\n"
            f"Verified evidence: {json.dumps(state.evidence.to_dict(), ensure_ascii=False)}"
    )
    review_parts = [
        shared_context,
        source('Acceptance contract: ' + json.dumps(review_contract, ensure_ascii=False), 'task:' + str(task.task_id) + ':contract', 'acceptance_contract'),
        source('Phase: ' + json.dumps(state.phase.to_dict(), ensure_ascii=False), 'phase:' + state.phase.phase_id, 'strategic_plan'),
        source('Verified evidence: ' + json.dumps(state.evidence.to_dict(), ensure_ascii=False), 'phase:' + state.phase.phase_id + ':evidence', 'execution_evidence'),
        source('Valid evidence references: ' + json.dumps(valid_refs, ensure_ascii=False), 'phase:' + state.phase.phase_id + ':valid_refs', 'verification_filter'),
    ]
    request = InferenceRequest(
        prompt=prompt,
        context_manifests=prompt_receipt(prompt, review_parts, purpose='review'),
        purpose=InferencePurpose.PHASE_REVIEW,
        input_sources=tuple([
            {"source_id": "criterion:" + str(item["id"]), "source_type": "acceptance_criterion", "reason": "completion_review"}
            for item in review_contract.get("criteria", [])
        ] + [
            {"source_id": str(entry.get('evidence_ref')), "source_type": "execution_evidence", "reason": "completion_review", "evidence_status": str(entry.get('status') or '')}
            for entry in state.evidence.entries if entry.get('evidence_ref')
        ]),
        project_id=task.project_id,
        user=task.user,
        task_id=task.task_id,
        images=context.image_inputs(task),
        output_schema=schema,
        cancel_checker=context.is_cancelled if context.cancellation_checker is not None else None,
    )
    try:
        result = context.inference.generate_json(request)
    except Exception as exc:
        raise SystemOneUnavailableError(f"acceptance_review_fallback:{type(exc).__name__}") from exc
    if not isinstance(result.json_value, dict):
        raise SystemOneUnavailableError("acceptance_review_fallback_invalid")
    return result.json_value
    # After one focused re-review, any remaining ambiguity is passed to Act.


def _task_context(task, context) -> str:
    selection = task.metadata.get("v3_context_selection")
    selected_ids = selection.get("selected_context_ids", []) if isinstance(selection, dict) else []
    return task_context_for_decision(task, context, selected_context_ids=selected_ids, purpose="review")


def _evaluate_system_one_with_context(system_one, *, contract, phase, evidence, task_context):
    evaluate = system_one.evaluate
    parameters = inspect.signature(evaluate).parameters.values()
    accepts_context = any(
        item.name == "task_context" or item.kind == inspect.Parameter.VAR_KEYWORD
        for item in parameters
    )
    kwargs = {"contract": contract, "phase": phase, "evidence": evidence}
    if accepts_context:
        kwargs["task_context"] = task_context
    return evaluate(**kwargs)


def _successful_evidence_refs(state: TacticalState) -> tuple[str, ...]:
    return tuple(
        str(item.get("evidence_ref"))
        for item in state.evidence.entries
        if str(item.get("status") or "") == "success" and item.get("evidence_ref")
    )


def _invariant_violations(task: "TaskState", state: TacticalState) -> list[str]:
    phase_by_evidence_ref: dict[str, dict[str, Any]] = {}
    history = task.metadata.get("v3_phase_history") if isinstance(task.metadata, dict) else None
    for item in history if isinstance(history, list) else []:
        if not isinstance(item, dict):
            continue
        phase = item.get("phase")
        evidence = item.get("evidence")
        entries = evidence.get("entries") if isinstance(evidence, dict) else None
        if not isinstance(phase, dict) or not isinstance(entries, list):
            continue
        for entry in entries:
            if (
                isinstance(entry, dict)
                and str(entry.get("phase_id") or "") == str(phase.get("phase_id") or "")
                and str(entry.get("evidence_ref") or "").strip()
            ):
                phase_by_evidence_ref[str(entry["evidence_ref"])] = phase

    # Current phase policy is authoritative for its evidence even if an older
    # checkpoint happens to reuse an evidence reference.
    current_phase = state.phase.to_dict()
    current_refs = {
        str(item.get("evidence_ref") or "").strip()
        for item in state.evidence.entries
        if str(item.get("phase_id") or "") in {"", state.phase.phase_id}
    }
    phase_by_evidence_ref.update({ref: current_phase for ref in current_refs if ref})
    violations: list[str] = []
    for entry in state.evidence.entries:
        result = entry.get("result")
        if not isinstance(result, dict):
            continue
        evidence_ref = str(entry.get("evidence_ref") or "").strip()
        phase = phase_by_evidence_ref.get(evidence_ref)
        if phase is None:
            violations.append(f"phase_scope_missing_for_evidence:{evidence_ref or '(unknown)'}")
            continue
        raw_scope = phase.get("mutation_scope") if isinstance(phase.get("mutation_scope"), dict) else {}
        allowed_paths = set(raw_scope.get("allowed_paths") or ())
        raw_subgoals = phase.get("subgoals") if isinstance(phase.get("subgoals"), list) else []
        subgoals = {
            str(item.get("subgoal_id") or ""): item
            for item in raw_subgoals if isinstance(item, dict)
        }
        affected = result.get("affected_paths")
        if isinstance(affected, list):
            subgoal = subgoals.get(str(entry.get("subgoal_id") or ""))
            effects = set(subgoal.get("allowed_side_effects") or ()) if subgoal is not None else set()
            if SideEffectClass.PROJECT_MUTATION in effects:
                unauthorized = [str(path) for path in affected if str(path) not in allowed_paths]
                if unauthorized:
                    violations.append(f"unauthorized_affected_paths:{','.join(unauthorized)}")
            elif effects & {SideEffectClass.EXTERNAL_REVERSIBLE, SideEffectClass.EXTERNAL_IRREVERSIBLE}:
                if not bool(raw_scope.get("allow_external_effects", False)):
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
