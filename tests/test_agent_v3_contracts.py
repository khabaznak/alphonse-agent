from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import CompletionCondition
from alphonse.agent_v2.core.intelligence.v3 import MutationScope
from alphonse.agent_v2.core.intelligence.v3 import PhaseEvidence
from alphonse.agent_v2.core.intelligence.v3 import PhaseLimits
from alphonse.agent_v2.core.intelligence.v3 import PhaseOutcome
from alphonse.agent_v2.core.intelligence.v3 import PhasePlan
from alphonse.agent_v2.core.intelligence.v3 import PhaseStatus
from alphonse.agent_v2.core.intelligence.v3 import PhaseSubgoal
from alphonse.agent_v2.core.intelligence.v3 import SideEffectClass
from alphonse.agent_v2.core.intelligence.v3 import TacticalAction
from alphonse.agent_v2.core.intelligence.v3 import TacticalState
from alphonse.agent_v2.core.intelligence.v3 import new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.contracts import legacy_call_phase


def _solar_phase() -> PhasePlan:
    return PhasePlan(
        phase_id="complete-solar",
        objective="Locate, complete, and verify the solar project",
        criterion_ids=("ac-1", "ac-2"),
        authorized_capabilities=("project_search", "exact_text_mutation", "project_read"),
        mutation_scope=MutationScope(allowed_paths=("mejoras_hogar/backlog.md",)),
        limits=PhaseLimits(max_tool_calls=5, max_duration_seconds=30),
        subgoals=(
            PhaseSubgoal(
                subgoal_id="locate",
                objective="Locate the authoritative record",
                required_output_type="record_reference",
                allowed_capabilities=("project_search",),
                completion=CompletionCondition(kind="output_present", output_type="record_reference"),
            ),
            PhaseSubgoal(
                subgoal_id="update",
                objective="Update the exact record",
                required_output_type="verified_mutation",
                depends_on=("locate",),
                allowed_capabilities=("exact_text_mutation",),
                allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,),
                completion=CompletionCondition(kind="field_equals", field="verification.status", expected="verified"),
            ),
            PhaseSubgoal(
                subgoal_id="verify",
                objective="Verify final state",
                required_output_type="record_observation",
                depends_on=("update",),
                allowed_capabilities=("project_read",),
                completion=CompletionCondition(kind="output_present", output_type="record_observation"),
            ),
        ),
    )


def test_phase_plan_round_trips_with_nested_contracts() -> None:
    phase = _solar_phase()

    restored = PhasePlan.from_dict(json.loads(json.dumps(phase.to_dict())))

    assert restored == phase
    assert restored.mutation_scope.allowed_paths == ("mejoras_hogar/backlog.md",)
    assert restored.subgoals[1].allowed_side_effects == (SideEffectClass.PROJECT_MUTATION,)


def test_phase_rejects_duplicate_and_forward_dependencies() -> None:
    first = PhaseSubgoal("one", "First", "record")
    duplicate = PhaseSubgoal("one", "Again", "record")
    with pytest.raises(ValueError, match="ids_duplicate"):
        PhasePlan("bad", "Bad", (first, duplicate))

    forward = PhaseSubgoal("one", "First", "record", depends_on=("later",))
    later = PhaseSubgoal("later", "Later", "record")
    with pytest.raises(ValueError, match="dependency_invalid"):
        PhasePlan("bad", "Bad", (forward, later))


def test_phase_rejects_subgoal_capability_outside_phase_authorization() -> None:
    subgoal = PhaseSubgoal(
        "ocr", "Read image", "text", allowed_capabilities=("attachment_ocr",)
    )

    with pytest.raises(ValueError, match="capability_unauthorized"):
        PhasePlan("bad", "Bad", (subgoal,), authorized_capabilities=("project_search",))


@pytest.mark.parametrize("path", ["/etc/passwd", "../outside.md", "safe/../../outside.md", ""])
def test_mutation_scope_rejects_non_project_relative_paths(path: str) -> None:
    with pytest.raises(ValueError, match="mutation_scope_path_invalid"):
        MutationScope(allowed_paths=(path,))


def test_tactical_state_round_trip_preserves_full_evidence_and_binds_state() -> None:
    phase = _solar_phase()
    evidence = PhaseEvidence()
    evidence.append({"evidence_ref": "action:search", "result": {"path": "mejoras_hogar/backlog.md"}})
    state = TacticalState(
        phase=phase,
        active_subgoal_id="update",
        status=PhaseStatus.RUNNING,
        completed_subgoal_ids=["locate"],
        bindings={"record_reference": {"path": "mejoras_hogar/backlog.md"}},
        revealed_capabilities=["exact_text_mutation"],
        revealed_tool_ids=["native.exact_text_edit"],
        actions=[TacticalAction("search", "locate", "native.search", {"query": "solar"}, status="success")],
        evidence=evidence,
        remaining_tool_calls=4,
        deadline_at="2026-09-20T12:00:00+00:00",
    )

    restored = TacticalState.from_dict(json.loads(json.dumps(state.to_dict())))

    assert restored.to_dict() == state.to_dict()
    assert restored.bindings["record_reference"]["path"] == "mejoras_hogar/backlog.md"
    assert restored.evidence.entries[0]["evidence_ref"] == "action:search"


def test_tactical_state_enforces_transitions_and_tool_budget() -> None:
    state = TacticalState(_solar_phase(), active_subgoal_id="locate")
    state.transition(PhaseStatus.RUNNING)
    state.consume_tool_call()
    state.transition(PhaseStatus.SUBGOAL_COMPLETE)
    state.transition(PhaseStatus.RUNNING)

    assert state.remaining_tool_calls == 4
    with pytest.raises(ValueError, match="transition_invalid"):
        state.transition(PhaseStatus.PHASE_COMPLETE)


def test_tactical_state_binds_only_declared_output_type() -> None:
    state = TacticalState(_solar_phase(), active_subgoal_id="locate")
    state.bind_subgoal_output("locate", "record_reference", {"path": "backlog.md"})

    assert state.bindings["record_reference"]["subgoal_id"] == "locate"
    with pytest.raises(ValueError, match="output_type_invalid"):
        state.bind_subgoal_output("locate", "ocr_text", "wrong")


def test_new_tactical_state_sets_absolute_deadline_and_imports_prior_evidence() -> None:
    state = new_tactical_state(
        _solar_phase(),
        cumulative_evidence=[{"evidence_ref": "tool-call:prior"}],
        now=datetime(2026, 9, 20, 12, 0, tzinfo=timezone.utc),
    )

    assert state.deadline_at == "2026-09-20T12:00:30+00:00"
    assert state.evidence.entries == [{"source": "prior_task_evidence", "evidence_ref": "tool-call:prior"}]


def test_tactical_state_rejects_naive_or_invalid_deadline() -> None:
    with pytest.raises(ValueError, match="tactical_deadline_invalid"):
        TacticalState(_solar_phase(), active_subgoal_id="locate", deadline_at="2026-09-20T12:00:00")


def test_waiting_state_can_resume_but_completed_phase_is_immutable() -> None:
    waiting = TacticalState(_solar_phase(), active_subgoal_id="locate")
    waiting.transition(PhaseStatus.RUNNING)
    waiting.transition(PhaseStatus.WAITING_USER)
    waiting.transition(PhaseStatus.RUNNING)
    assert waiting.status == PhaseStatus.RUNNING

    completed = TacticalState(_solar_phase(), active_subgoal_id="verify", status=PhaseStatus.SUBGOAL_COMPLETE)
    completed.transition(PhaseStatus.PHASE_COMPLETE)
    with pytest.raises(ValueError, match="phase_status_terminal"):
        completed.transition(PhaseStatus.RUNNING)


def test_prompt_projection_is_bounded_without_mutating_full_evidence() -> None:
    evidence = PhaseEvidence()
    for index in range(20):
        evidence.append({"evidence_ref": f"action:{index}", "result": "x" * 1000})
    state = TacticalState(_solar_phase(), active_subgoal_id="locate", evidence=evidence)

    rendered = state.prompt_projection(max_evidence_entries=2, max_chars=4000)

    assert len(rendered) <= 4000
    assert len(state.evidence.entries) == 20
    assert "action:19" in rendered
    assert "action:0" not in rendered


def test_phase_outcome_requires_terminal_or_waiting_status() -> None:
    outcome = PhaseOutcome("complete-solar", PhaseStatus.PHASE_COMPLETE, evidence_refs=("action:verify",))
    assert PhaseOutcome.from_dict(outcome.to_dict()) == outcome

    with pytest.raises(ValueError, match="nonterminal"):
        PhaseOutcome("complete-solar", PhaseStatus.RUNNING)


def test_legacy_call_adapter_creates_only_one_subgoal() -> None:
    phase = legacy_call_phase(
        {"id": "call-1", "tool_id": "native.bash", "internal_state": "Inspect one file"}
    )

    assert len(phase.subgoals) == 1
    assert phase.subgoals[0].subgoal_id == "call-1"
    assert phase.originating_decision == "legacy_one_call_adapter"


def test_task_state_checkpoints_v3_state_without_changing_v2_default() -> None:
    state = TaskState()
    assert state.intelligence_engine == "tactical_v2"
    assert state.intelligence_schema_version == 2

    state.intelligence_engine = "hierarchical_v3"
    state.intelligence_schema_version = 3
    state.hierarchical_state = TacticalState(_solar_phase(), active_subgoal_id="locate").to_dict()
    restored = TaskState.from_dict(state.to_checkpoint_dict())

    assert restored.intelligence_engine == "hierarchical_v3"
    assert restored.intelligence_schema_version == 3
    assert TacticalState.from_dict(restored.hierarchical_state).phase.phase_id == "complete-solar"


def test_legacy_task_state_dictionary_loads_with_v2_defaults() -> None:
    restored = TaskState.from_dict({"goal": "legacy"})

    assert restored.intelligence_engine == "tactical_v2"
    assert restored.intelligence_schema_version == 2
    assert restored.hierarchical_state == {}
