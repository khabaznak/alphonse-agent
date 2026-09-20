from __future__ import annotations

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRouter, ModelProfile, StubInferenceProvider
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import CompletionCondition
from alphonse.agent_v2.core.intelligence.v3 import MutationScope
from alphonse.agent_v2.core.intelligence.v3 import PhaseLimits
from alphonse.agent_v2.core.intelligence.v3 import PhaseOutcome
from alphonse.agent_v2.core.intelligence.v3 import PhasePlan
from alphonse.agent_v2.core.intelligence.v3 import PhaseReviewStatus
from alphonse.agent_v2.core.intelligence.v3 import PhaseStatus
from alphonse.agent_v2.core.intelligence.v3 import PhaseSubgoal
from alphonse.agent_v2.core.intelligence.v3 import StrategicAction
from alphonse.agent_v2.core.intelligence.v3 import SideEffectClass
from alphonse.agent_v2.core.intelligence.v3 import V3OuterController
from alphonse.agent_v2.core.intelligence.v3 import new_tactical_state
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.system_one import SystemOneReviewResult


def _phase():
    return PhasePlan(
        "solar", "Mark the solar project complete",
        (PhaseSubgoal(
            "update", "Update", "verified_mutation",
            allowed_capabilities=("exact_text_mutation",),
            allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,),
            completion=CompletionCondition("field_equals", field="verification.status", expected="verified"),
        ),),
        criterion_ids=("ac-1",),
        authorized_capabilities=("exact_text_mutation",),
        mutation_scope=MutationScope(("backlog.md",)),
        limits=PhaseLimits(2, 20),
    )


def _completed_state(*, affected_path="backlog.md"):
    state = new_tactical_state(_phase())
    state.status = PhaseStatus.PHASE_COMPLETE
    state.completed_subgoal_ids = ["update"]
    state.evidence.append({
        "evidence_ref": "tactical-action:edit",
        "subgoal_id": "update",
        "status": "success",
        "result": {
            "affected_paths": [affected_path],
            "verification": {"status": "verified"},
        },
    })
    return state


def _task():
    task = TaskState(goal="Marca completo el proyecto solar", user="alex", project_id="home")
    task.set_acceptance_contract_from_markdown("1.- [ ] Solar project is complete")
    return task


def _context(*, satisfy=True, response="Listo, Alex. El proyecto solar quedó completo."):
    updates = []
    if satisfy:
        updates.append({
            "criterion_id": "ac-1", "status": "satisfied",
            "evidence_refs": ["tactical-action:edit"], "reason": "Verified edit",
        })
    provider = StubInferenceProvider(
        json_by_purpose={InferencePurpose.PHASE_REVIEW: {"updates": updates}},
        markdown_by_purpose={InferencePurpose.FINAL_RESPONSE: response},
    )
    router = InferenceRouter(provider=provider, default_profile=ModelProfile("test", "test", "default"))
    return CoreLoopContext(messages=InMemoryMessageQueue(), inference=router), provider


def test_verified_complete_phase_routes_directly_to_response_and_end() -> None:
    task = _task()
    state = _completed_state()
    context, provider = _context()

    review, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context
    )

    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE
    assert decision.action == StrategicAction.COMPLETE
    assert task.status == "completed"
    assert task.metadata["v3_route"] == "respond_and_end"
    assert task.metadata["prepared_user_response"]["message"].startswith("Listo")
    assert [item.purpose for item in provider.requests] == [
        InferencePurpose.PHASE_REVIEW,
        InferencePurpose.FINAL_RESPONSE,
    ]
    assert all(item.tools == () for item in provider.requests)


def test_completed_phase_with_unmet_criteria_routes_to_next_phase() -> None:
    task = _task()
    context, provider = _context(satisfy=False)

    review, decision = V3OuterController().review_and_route(
        task, _completed_state(), PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context
    )

    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE
    assert decision.action == StrategicAction.CONTINUE
    assert task.metadata["v3_route"] == "plan_next_phase"
    assert [item.purpose for item in provider.requests] == [InferencePurpose.PHASE_REVIEW]
    assert "prepared_user_response" not in task.metadata


def test_scope_violation_prevents_completion_without_model_review() -> None:
    task = _task()
    context, provider = _context()

    review, decision = V3OuterController().review_and_route(
        task, _completed_state(affected_path="other-project.md"),
        PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context,
    )

    assert review.status == PhaseReviewStatus.VERIFICATION_FAILED
    assert "unauthorized_affected_paths" in review.violations[0]
    assert decision.action == StrategicAction.REPLAN
    assert task.metadata["v3_route"] == "strategic_replan"
    assert provider.requests == []


def test_authorized_external_effect_is_not_treated_as_project_path_mutation() -> None:
    phase = PhasePlan(
        "medical", "Record medical event",
        (PhaseSubgoal(
            "record", "Record", "artifact_result",
            allowed_capabilities=("project_artifact_query",),
            allowed_side_effects=(SideEffectClass.EXTERNAL_REVERSIBLE,),
        ),),
        criterion_ids=("ac-1",),
        authorized_capabilities=("project_artifact_query",),
        mutation_scope=MutationScope((), allow_external_effects=True),
    )
    state = new_tactical_state(phase)
    state.status = PhaseStatus.PHASE_COMPLETE
    state.completed_subgoal_ids = ["record"]
    state.evidence.append({
        "evidence_ref": "tactical-action:edit", "subgoal_id": "record", "status": "success",
        "result": {"affected_paths": ["resolved medical artifact"], "recorded": True},
    })
    task = _task()
    context, _ = _context()

    review, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome("medical", PhaseStatus.PHASE_COMPLETE), context,
    )

    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE
    assert decision.action == StrategicAction.COMPLETE


def test_blocked_phase_routes_to_strategic_replan_and_keeps_failure_visible() -> None:
    task = _task()
    state = new_tactical_state(_phase())
    state.status = PhaseStatus.BLOCKED
    state.evidence.append({"evidence_ref": "tactical-action:failed", "status": "failed", "error": "not found"})
    context, _ = _context()

    review, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome("solar", PhaseStatus.BLOCKED, reason="not found"), context
    )

    assert review.status == PhaseReviewStatus.PHASE_BLOCKED
    assert decision.action == StrategicAction.REPLAN
    assert task.metadata["v3_phase_review"]["reason"] == "not found"
    assert "prepared_user_response" not in task.metadata


def test_waiting_phase_parks_task() -> None:
    task = _task()
    state = new_tactical_state(_phase())
    state.status = PhaseStatus.WAITING_USER
    context, _ = _context()

    review, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome("solar", PhaseStatus.WAITING_USER, reason="Choose record"), context
    )

    assert review.status == PhaseReviewStatus.WAITING_USER
    assert decision.action == StrategicAction.ASK_USER
    assert task.status == "waiting_user"


def test_final_response_fallback_is_generated_without_inference() -> None:
    task = _task()
    task.acceptance_contract["criteria"][0]["status"] = "satisfied"
    task.sync_acceptance_criteria_view()
    state = _completed_state()

    review, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE),
        CoreLoopContext(messages=InMemoryMessageQueue()),
    )

    assert decision.action == StrategicAction.COMPLETE
    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE
    assert "quedó verificado" in task.metadata["prepared_user_response"]["message"]


class _SystemOne:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def evaluate(self, **_values):
        if self.error:
            raise self.error
        return self.result


def test_system_one_check_and_act_can_conservatively_withhold_completion() -> None:
    task = _task()
    state = _completed_state()
    context, provider = _context()
    context.system_one = _SystemOne(SystemOneReviewResult(
        updates=({"criterion_id": "ac-1", "status": "satisfied", "evidence_refs": ["tactical-action:edit"]},),
        ambiguous_criterion_ids=(), recommended_route="replan", route_confidence=0.92, route_confident=True,
    ))

    review, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context,
    )

    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE
    assert decision.action == StrategicAction.REPLAN
    assert task.metadata["v3_route"] == "strategic_replan"
    assert task.metadata["system_one_review"]["status"] == "used"
    assert provider.requests == []


def test_system_one_failure_falls_back_to_existing_phase_review() -> None:
    task = _task()
    context, provider = _context()
    context.system_one = _SystemOne(error=RuntimeError("unavailable"))

    review, decision = V3OuterController().review_and_route(
        task, _completed_state(), PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context,
    )

    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE
    assert decision.action == StrategicAction.COMPLETE
    assert task.metadata["system_one_review"]["status"] == "fallback"
    assert [item.purpose for item in provider.requests] == [InferencePurpose.PHASE_REVIEW, InferencePurpose.FINAL_RESPONSE]
