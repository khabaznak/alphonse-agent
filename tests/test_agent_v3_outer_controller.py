from __future__ import annotations

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRouter, ModelProfile, StubInferenceProvider
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import CompletionCondition
from alphonse.agent_v2.core.intelligence.v3 import MutationScope
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
from alphonse.agent_v2.system_one import SystemOneActRecommendation, SystemOneReviewResult


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


def test_verified_complete_phase_routes_to_final_response_planning() -> None:
    task = _task()
    state = _completed_state()
    context, provider = _context()

    review, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context
    )

    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE
    assert decision.action == StrategicAction.CONTINUE
    assert task.status == "running"
    assert task.metadata["v3_route"] == "plan_next_phase"
    assert task.metadata["act_directive"]["response_required"] is True
    assert [item.purpose for item in provider.requests] == [InferencePurpose.PHASE_REVIEW]
    assert all(item.tools == () for item in provider.requests)


def test_later_work_invalidates_an_earlier_response_even_after_checkpoint_restore():
    task = _task()
    task.metadata["prepared_user_response"] = {
        "source": "native.respond", "tool_call_id": "old-response",
        "message": "I could not update the record.", "work_evidence_refs": [],
    }
    task = TaskState.from_dict(task.to_checkpoint_dict())
    state = _completed_state()
    context, _ = _context()
    review, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context)
    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE
    assert decision.action == StrategicAction.CONTINUE
    assert not task.has_prepared_user_response()
    assert task.metadata["act_directive"]["response_required"]
    assert task.metadata["v3_response_refresh_required"]


def test_same_evidence_does_not_discard_a_current_response():
    task = _task()
    task.metadata["prepared_user_response"] = {
        "source": "native.respond", "message": "The record is updated.",
        "work_evidence_refs": ["tactical-action:edit"],
    }
    state = _completed_state()
    context, _ = _context()
    _, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.COMPLETE
    assert task.metadata["prepared_user_response"]["message"] == "The record is updated."


@pytest.mark.parametrize("authenticated_followup", [False, True])
def test_mission_guard_catches_unfulfilled_effects_with_or_without_followup(authenticated_followup):
    task = _task()
    task.metadata["prepared_user_response"] = {"message": "I acknowledged your request."}
    if authenticated_followup:
        task.metadata["v3_resumption_evidence"] = [{"kind": "authenticated_question_answer", "result": {"answer": True}}]
    state = _completed_state()
    context, _ = _context()

    class MissionUnmet(_SystemOne):
        def recommend_act(self, *, state):
            assert "available_tool_ids" in state
            return SystemOneActRecommendation("complete", 0.99, True, "Contract satisfied", answers={
                "mission_outcome_is_fulfilled": 0.05, "continuation_is_worthwhile": 0.99,
                "user_input_can_unblock": 0.05, "closure_explanation_is_warranted": 0.9,
            })

    context.system_one = MissionUnmet(SystemOneReviewResult(({
        "criterion_id": "ac-1", "status": "satisfied", "evidence_refs": ["tactical-action:edit"],
    },), ()))
    review, decision = V3OuterController().review_and_route(task, state,
        PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context)
    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE
    assert decision.action == StrategicAction.REPLAN
    assert task.status == "running"
    assert task.metadata["v3_route"] == "strategic_replan"


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


def test_repeated_completed_phases_without_acceptance_progress_keep_routing() -> None:
    task = _task()
    context, _ = _context(satisfy=False)
    controller = V3OuterController()

    decisions = [
        controller.review_and_route(
            task,
            _completed_state(),
            PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE),
            context,
        )[1]
        for _ in range(5)
    ]

    assert [decision.action for decision in decisions] == [StrategicAction.CONTINUE] * 5
    assert task.status == "running"
    assert "v3_consecutive_no_progress_phases" not in task.metadata


def test_continue_choice_with_uncertain_continuation_replans_instead_of_failing() -> None:
    class Jev:
        def evaluate(self, **_values):
            return SystemOneReviewResult(updates=(), ambiguous_criterion_ids=("ac-1",))

        def recommend_act(self, *, state):
            assert state["check_verdict"] == "wip"
            return SystemOneActRecommendation(
                "continue",
                0.91,
                True,
                "More authorized work under the current strategy is likely to make useful progress.",
                answers={
                    "continuation_is_worthwhile": 0.72,
                    "user_input_can_unblock": 0.1,
                    "closure_explanation_is_warranted": 0.75,
                },
            )

    task = _task()
    context = CoreLoopContext(messages=InMemoryMessageQueue(), system_one=Jev())

    _review, decision = V3OuterController().review_and_route(
        task,
        _completed_state(),
        PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE),
        context,
    )

    assert decision.action == StrategicAction.REPLAN
    assert task.status != "failed"
    assert task.metadata["v3_route"] == "strategic_replan"
    assert "returning to Plan" in decision.reason
    assert "More authorized work" not in decision.reason


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
    assert decision.action == StrategicAction.CONTINUE
    assert task.metadata["v3_route"] == "plan_next_phase"


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


def test_check_requires_available_reviewer_when_inference_and_jev_are_unavailable() -> None:
    task = _task()
    task.acceptance_contract["criteria"][0]["status"] = "satisfied"
    task.sync_acceptance_criteria_view()
    state = _completed_state()

    with pytest.raises(RuntimeError, match="acceptance_review_unavailable"):
        V3OuterController().review_and_route(
            task, state, PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE),
            CoreLoopContext(messages=InMemoryMessageQueue()),
        )


class _SystemOne:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error

    def evaluate(self, **_values):
        if self.error:
            raise self.error
        return self.result

    def recommend_act(self, *, state):
        if self.error or self.result is None:
            raise RuntimeError("unavailable")
        action = self.result.recommended_route or (
            "complete" if state.get("check_verdict") == "success" else "continue"
        )
        return SystemOneActRecommendation(
            action=action,
            confidence=self.result.route_confidence,
            confident=self.result.route_confident,
            rationale=f"Test recommendation: {action}.",
            answers={"continuation_is_worthwhile": 0.99 if action == "continue" else 0.1,
                     "user_input_can_unblock": 0.99 if action == "ask_user" else 0.1,
                     "closure_explanation_is_warranted": 0.1},
        )


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


def test_ambiguous_acceptance_criterion_is_focused_rereviewed_before_act() -> None:
    task = _task()
    context, _ = _context(satisfy=False)

    class _AmbiguousThenResolvedSystemOne(_SystemOne):
        def __init__(self):
            super().__init__()
            self.reviews = []

        def evaluate(self, **values):
            self.reviews.append(values)
            if len(self.reviews) == 1:
                return SystemOneReviewResult(
                    updates=(), ambiguous_criterion_ids=("ac-1",),
                )
            return SystemOneReviewResult(
                updates=({
                    "criterion_id": "ac-1",
                    "status": "satisfied",
                    "evidence_refs": ["tactical-action:edit"],
                    "reason": "The ordered inspection evidence precedes the verified update.",
                },),
                ambiguous_criterion_ids=(),
            )

        def recommend_act(self, **_values):
            return SystemOneActRecommendation(
                action="complete", confidence=0.99, confident=True,
                rationale="All criteria have supporting evidence.",
                answers={
                    "continuation_is_worthwhile": 0.1,
                    "user_input_can_unblock": 0.1,
                    "closure_explanation_is_warranted": 0.9,
                },
            )

    system_one = _AmbiguousThenResolvedSystemOne()
    context.system_one = system_one

    review, _decision = V3OuterController().review_and_route(
        task, _completed_state(), PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context,
    )

    assert len(system_one.reviews) == 2
    assert [item["id"] for item in system_one.reviews[1]["contract"]["criteria"]] == ["ac-1"]
    assert system_one.reviews[1]["phase"]["review_focus"]["criterion_ids"] == ["ac-1"]
    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE
    assert task.acceptance_criteria_all_complete()
    assert task.metadata["system_one_review"]["status"] == "focused_rereview_resolved"


def test_fail_explain_recommendation_does_not_close_a_completed_incomplete_phase() -> None:
    task = _task()
    context, _ = _context(satisfy=False)

    class _FailExplainSystemOne(_SystemOne):
        def evaluate(self, **_values):
            return SystemOneReviewResult(
                updates=(), ambiguous_criterion_ids=(),
                recommended_route="continue", route_confidence=0.5, route_confident=False,
            )

        def recommend_act(self, **_values):
            return SystemOneActRecommendation(
                action="fail_explain", confidence=0.94, confident=True,
                rationale="Continue fuse was low after a prerequisite phase.",
                answers={
                    "continuation_is_worthwhile": 0.2,
                    "user_input_can_unblock": 0.1,
                    "closure_explanation_is_warranted": 0.95,
                },
            )

    context.system_one = _FailExplainSystemOne()

    review, decision = V3OuterController().review_and_route(
        task, _completed_state(), PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context,
    )

    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE
    assert decision.action == StrategicAction.CONTINUE
    assert task.metadata["v3_route"] == "plan_next_phase"
    assert "act_directive" not in task.metadata


def test_system_one_failure_falls_back_to_existing_phase_review() -> None:
    task = _task()
    context, provider = _context()
    context.system_one = _SystemOne(error=RuntimeError("unavailable"))

    review, decision = V3OuterController().review_and_route(
        task, _completed_state(), PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context,
    )

    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE
    assert decision.action == StrategicAction.CONTINUE
    assert task.metadata["v3_route"] == "plan_next_phase"
    assert task.metadata["system_one_review"]["status"] == "fallback"
    assert [item.purpose for item in provider.requests] == [InferencePurpose.PHASE_REVIEW]


def test_system_one_confident_updates_survive_partial_ambiguity_and_response_is_reused() -> None:
    task = TaskState(goal="Provide a safe routine", user="alex", project_id="home")
    task.set_acceptance_contract_from_markdown(
        "1.- [ ] The response contains a routine\n"
        "2.- [ ] The response contains safety guidance"
    )
    task.metadata["prepared_user_response"] = {
        "source": "native.respond",
        "tool_call_id": "respond-once",
        "message": "Routine with safety guidance.",
    }
    state = _completed_state()
    state.evidence.entries[0].update({
        "evidence_ref": "tactical-action:respond-once",
        "tool_id": "native.respond",
        "result": {"message": "Routine with safety guidance."},
    })
    provider = StubInferenceProvider(json_by_purpose={
        InferencePurpose.PHASE_REVIEW: {
            "updates": [{
                "criterion_id": "ac-2",
                "status": "satisfied",
                "evidence_refs": ["tactical-action:respond-once"],
                "reason": "Safety guidance is present.",
            }],
        },
    })
    context = CoreLoopContext(
        messages=InMemoryMessageQueue(),
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile("test", "test", "default")),
        system_one=_SystemOne(SystemOneReviewResult(
            updates=({
                "criterion_id": "ac-1",
                "status": "satisfied",
                "evidence_refs": ["tactical-action:respond-once"],
                "reason": "Routine is present.",
            },),
            ambiguous_criterion_ids=("ac-2",),
            recommended_route="continue",
            route_confidence=0.55,
            route_confident=False,
        )),
    )

    review, decision = V3OuterController().review_and_route(
        task, state, PhaseOutcome("solar", PhaseStatus.PHASE_COMPLETE), context,
    )

    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_COMPLETE
    assert decision.action == StrategicAction.REPLAN
    assert task.metadata["v3_route"] == "strategic_replan"
    assert task.metadata["system_one_review"]["status"] == "partial_fallback"
    assert task.metadata["prepared_user_response"]["tool_call_id"] == "respond-once"
    assert [item.purpose for item in provider.requests] == [InferencePurpose.PHASE_REVIEW]
    assert "The response contains a routine" not in provider.requests[0].prompt
    assert "The response contains safety guidance" in provider.requests[0].prompt


@pytest.mark.parametrize('during_review', [False, True])
def test_dc1_cancellation_prevents_terminal_success(during_review):
    from alphonse.agent_v2.core.intelligence.v3.outer import review_phase
    task = _task()
    state = _completed_state()
    cancelled = {'value': not during_review}
    context, _ = _context()
    context.cancellation_checker = lambda: cancelled['value']
    class Judge:
        def evaluate(self, **kwargs):
            cancelled['value'] = True
            return SystemOneReviewResult(updates=({'criterion_id': 'ac-1', 'status': 'satisfied', 'evidence_refs': ['tactical-action:edit']},), ambiguous_criterion_ids=())
    context.system_one = Judge()
    review = review_phase(task, state, PhaseOutcome(state.phase.phase_id, PhaseStatus.PHASE_COMPLETE), context)
    assert review.status == PhaseReviewStatus.CANCELLED
    assert task.status != 'completed'


def test_dc1_review_budget_failure_never_falls_back_to_inference(monkeypatch):
    from alphonse.agent_v2.core.intelligence.v3.outer import review_phase
    from alphonse.agent_v2.core.intelligence.v3.projections import ContextBudgetExceeded
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '1')
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_TOKEN_BUDGET', '1')
    task, state = _task(), _completed_state()
    context, provider = _context()
    class Judge:
        def evaluate(self, **kwargs):
            pytest.fail('Oversized required context reached Jev')
    context.system_one = Judge()
    with pytest.raises(ContextBudgetExceeded):
        review_phase(task, state, PhaseOutcome(state.phase.phase_id, PhaseStatus.PHASE_COMPLETE), context)
    assert provider.requests == []


@pytest.mark.parametrize('provider', ['jev', 'system_two'])
def test_dc1_failed_or_historical_evidence_cannot_satisfy_contract(monkeypatch, provider):
    import copy
    from types import SimpleNamespace
    from alphonse.agent_v2.core.intelligence.v3.outer import review_phase
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '1')
    task, state = _task(), _completed_state()
    original = copy.deepcopy(task.ensure_acceptance_contract())
    task.memory_context_md = 'HISTORICAL_SUCCESS_SENTINEL'
    state.evidence.entries[0]['status'] = 'failed'
    updates = [{'criterion_id': 'ac-1', 'status': 'satisfied', 'evidence_refs': ['tactical-action:edit', 'historical:write'], 'reason': 'Untrusted success judgment'}]
    captured = []
    context = CoreLoopContext(messages=InMemoryMessageQueue())
    if provider == 'jev':
        def evaluate(**kwargs):
            captured.append(kwargs)
            assert kwargs['contract'] == original
            assert kwargs['evidence'] == state.evidence.to_dict()
            return SystemOneReviewResult(updates=tuple(updates), ambiguous_criterion_ids=())
        context.system_one = SimpleNamespace(evaluate=evaluate)
    else:
        def generate(request):
            captured.append(request.prompt)
            assert 'failed' in request.prompt
            assert 'Solar project is complete' in request.prompt
            return SimpleNamespace(json_value={'updates': updates})
        context.inference = SimpleNamespace(generate_json=generate)
    review = review_phase(task, state, PhaseOutcome(state.phase.phase_id, PhaseStatus.PHASE_COMPLETE), context)
    assert 'HISTORICAL_SUCCESS_SENTINEL' not in str(captured)
    assert review.status == PhaseReviewStatus.PHASE_VERIFIED_TASK_INCOMPLETE
    assert task.acceptance_contract == original


@pytest.mark.parametrize('has_jev', [False, True])
def test_dc1_fallback_review_receives_required_context_before_inference(monkeypatch, has_jev):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.intelligence.v3.outer import review_phase
    from alphonse.agent_v2.core.intelligence.v3.projections import ContextBudgetExceeded
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '1')
    task, state = _task(), _completed_state()
    task.append_conversation_message('alex', 'STEERING_TAIL: Do not contact anyone.')
    calls = []
    def generate(request):
        calls.append(request)
        assert request.prompt.count(task.goal) == 1
        assert 'STEERING_TAIL' in request.prompt
        return SimpleNamespace(json_value={'updates': []})
    def fail(**kwargs):
        raise OSError('synthetic provider outage')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), inference=SimpleNamespace(generate_json=generate),
                              system_one=SimpleNamespace(evaluate=fail) if has_jev else None)
    review_phase(task, state, PhaseOutcome(state.phase.phase_id, PhaseStatus.PHASE_COMPLETE), context)
    assert len(calls) == 1
    calls.clear()
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_TOKEN_BUDGET', '1')
    with pytest.raises(ContextBudgetExceeded):
        review_phase(task, state, PhaseOutcome(state.phase.phase_id, PhaseStatus.PHASE_COMPLETE), context)
    assert calls == []


@pytest.mark.parametrize('closure_only', [False, True])
def test_dc1_cancellation_cannot_be_overridden_by_closure_or_act(closure_only):
    from types import SimpleNamespace
    task, state = _task(), _completed_state()
    context, _ = _context()
    cancelled = {'value': closure_only}
    context.cancellation_checker = lambda: cancelled['value']
    if closure_only:
        task.metadata['act_directive'] = {'closure_only': True, 'terminal_outcome': 'success'}
        task.metadata['prepared_user_response'] = {'message': 'Synthetic reply'}
    else:
        def recommend_act(**kwargs):
            cancelled['value'] = True
            return SystemOneActRecommendation('complete', 1.0, True, 'Synthetic late cancellation')
        context.system_one = SimpleNamespace(recommend_act=recommend_act)
    review, decision = V3OuterController().review_and_route(task, state, PhaseOutcome(state.phase.phase_id, PhaseStatus.PHASE_COMPLETE), context)
    assert review.status == PhaseReviewStatus.CANCELLED
    assert task.status == 'cancelled'
    assert decision.action == StrategicAction.FAIL


def test_dc1_closure_directive_cannot_turn_failed_verification_into_success():
    task, state = _task(), _completed_state(affected_path='outside-authorized-scope.md')
    task.metadata['act_directive'] = {'closure_only': True, 'terminal_outcome': 'success'}
    # The response matches the current evidence so freshness cannot mask the guard.
    task.metadata['prepared_user_response'] = {'message': 'Synthetic reply', 'work_evidence_refs': ['tactical-action:edit']}
    context, provider = _context()
    review, decision = V3OuterController().review_and_route(task, state, PhaseOutcome(state.phase.phase_id, PhaseStatus.PHASE_COMPLETE), context)
    assert review.status == PhaseReviewStatus.VERIFICATION_FAILED
    assert task.status != 'completed'
    assert decision.action == StrategicAction.REPLAN
    assert provider.requests == []


@pytest.mark.parametrize('requires_effect', [False, True])
def test_reply_only_evidence_cannot_complete_persisted_effect_but_can_answer_fact(requires_effect):
    task = TaskState(user='alex', goal='Save the reviewed record' if requires_effect else 'Explain the uncertain result')
    task.set_acceptance_contract_from_markdown('1.- [ ] Request is acknowledged')
    task.metadata['prepared_user_response'] = {'message': 'Here is the qualified answer.'}
    phase = PhasePlan('reply', 'Reply', (PhaseSubgoal(
        'reply', 'Reply', 'user_response', allowed_capabilities=('user_response',),
        allowed_side_effects=(SideEffectClass.USER_RESPONSE,),
    ),), criterion_ids=('ac-1',), authorized_capabilities=('user_response',))
    state = new_tactical_state(phase)
    state.status = PhaseStatus.PHASE_COMPLETE
    state.completed_subgoal_ids = ['reply']
    state.evidence.append({'evidence_ref': 'response:1', 'subgoal_id': 'reply', 'tool_id': 'native.respond',
                           'status': 'success', 'result': {'message': 'Here is the qualified answer.'}})

    class PositiveJev(_SystemOne):
        def recommend_act(self, **kwargs):
            return SystemOneActRecommendation('complete', .99, True, 'All criteria satisfied', answers={
                'mission_outcome_is_fulfilled': .99, 'mission_requires_persisted_effect': .99 if requires_effect else .01,
            })

    context, _ = _context()
    context.system_one = PositiveJev(SystemOneReviewResult(({
        'criterion_id': 'ac-1', 'status': 'satisfied', 'evidence_refs': ['response:1'],
    },), ()))
    _, decision = V3OuterController().review_and_route(task, state, PhaseOutcome('reply', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == (StrategicAction.REPLAN if requires_effect else StrategicAction.COMPLETE)


@pytest.mark.parametrize('probability', [None, float('nan'), float('inf'), -1, 2])
def test_invalid_required_mission_judgment_withholds_completion(probability):
    task = _task()
    task.metadata['prepared_user_response'] = {'message': 'Done'}
    context, _ = _context()

    class InvalidJev(_SystemOne):
        requires_mission_outcome_review = True

        def recommend_act(self, **kwargs):
            return SystemOneActRecommendation('complete', .99, True, 'Complete', answers={
                'mission_outcome_is_fulfilled': probability,
            })

    context.system_one = InvalidJev(SystemOneReviewResult(({
        'criterion_id': 'ac-1', 'status': 'satisfied', 'evidence_refs': ['tactical-action:edit'],
    },), ()))
    _, decision = V3OuterController().review_and_route(task, _completed_state(),
        PhaseOutcome('solar', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.REPLAN
    assert task.status != 'completed'


def test_mission_disagreement_after_prepared_report_requires_action_or_specific_question():
    from alphonse.agent_v2.core.intelligence.v3.planner import _validate_inspection_handoff
    task = _task()
    task.metadata['prepared_user_response'] = {'message': 'The existing record has been verified.'}
    context, _ = _context()

    class Unresolved(_SystemOne):
        def recommend_act(self, **kwargs):
            return SystemOneActRecommendation('complete', .99, True, 'Complete', answers={
                'mission_outcome_is_fulfilled': .66, 'mission_requires_persisted_effect': .54,
            })

    context.system_one = Unresolved(SystemOneReviewResult(({
        'criterion_id': 'ac-1', 'status': 'satisfied', 'evidence_refs': ['tactical-action:edit'],
    },), ()))
    _, decision = V3OuterController().review_and_route(task, _completed_state(),
        PhaseOutcome('solar', PhaseStatus.PHASE_COMPLETE), context)
    assert decision.action == StrategicAction.REPLAN
    directive = task.metadata['act_directive']
    assert directive['inspection_resolution']['route'] == 'decision_required'
    report = PhasePlan('repeat-report', 'Repeat report', (PhaseSubgoal(
        'report', 'Repeat report', 'response', allowed_capabilities=('user_response',),
        allowed_side_effects=(SideEffectClass.USER_RESPONSE,),
    ),), authorized_capabilities=('user_response',))
    with pytest.raises(ValueError, match='inspection_handoff_requires_action_or_question'):
        _validate_inspection_handoff(report, directive)
