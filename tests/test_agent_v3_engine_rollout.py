from __future__ import annotations

from pathlib import Path

from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingResult, StateSnapshot, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRouter, ModelProfile, StubInferenceProvider
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import CompletionCondition
from alphonse.agent_v2.core.intelligence.v3 import EngineRoutingProcessor, HierarchicalCAPDProcessor
from alphonse.agent_v2.core.intelligence.v3 import PhaseLimits, PhaseOutcome, PhasePlan, PhaseStatus, PhaseSubgoal
from alphonse.agent_v2.core.intelligence.v3 import new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.processor import _deduplicated_history_evidence
from alphonse.agent_v2.core.messages import CommunicationChannel, InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
from alphonse.agent_v2.intelligence_engine_settings import HIERARCHICAL_V3, TACTICAL_V2
from alphonse.agent_v2.intelligence_engine_settings import IntelligenceEngineSettings
from alphonse.agent_v2.intelligence_engine_settings import SQLiteIntelligenceEngineSettingsStore
from alphonse.agent_v2.system_one import SystemOneReviewResult
from alphonse.agent_v2.system_one import SystemOneTacticalReview
from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection


def test_engine_settings_default_to_v3_and_allow_v2_rollback(tmp_path: Path) -> None:
    store = SQLiteIntelligenceEngineSettingsStore(tmp_path / "settings.sqlite3")
    assert store.get().default_engine == HIERARCHICAL_V3

    saved = store.save(IntelligenceEngineSettings(default_engine=TACTICAL_V2, v3_project_ids=("home",)))

    assert saved.engine_for("home") == HIERARCHICAL_V3
    assert saved.engine_for("health") == TACTICAL_V2


def test_channel_stamps_engine_at_ingestion_so_later_setting_changes_do_not_move_task() -> None:
    selected = {"engine": HIERARCHICAL_V3}
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue, intelligence_engine_provider=lambda project_id: selected["engine"])

    queued = channel.queue_message(prompt="Do work", user="alex", project_id="home")
    selected["engine"] = TACTICAL_V2
    task = TaskState.from_queued_message(queued)

    assert task.intelligence_engine == HIERARCHICAL_V3
    assert task.intelligence_schema_version == 3


def test_channel_defaults_new_messages_to_v3_without_a_settings_provider() -> None:
    queued = CommunicationChannel(InMemoryMessageQueue()).queue_message(
        prompt="Do work", user="alex", project_id="home"
    )

    assert queued.message.metadata["intelligence_engine"] == HIERARCHICAL_V3
    assert queued.message.metadata["intelligence_schema_version"] == 3


def test_v3_cumulative_phase_evidence_is_deduplicated_by_reference() -> None:
    repeated = {"evidence_ref": "action:one", "status": "success", "result": {"value": 1}}
    later = {"evidence_ref": "action:two", "status": "success", "result": {"value": 2}}
    history = [
        {"evidence": {"entries": [repeated]}},
        {"evidence": {"entries": [repeated, later]}},
        {"evidence": {"entries": [repeated, later]}},
    ]

    assert _deduplicated_history_evidence(history) == [repeated, later]


def test_v3_phase_history_stores_only_current_phase_evidence() -> None:
    phase = PhasePlan(
        "current-phase",
        "Complete current work",
        (PhaseSubgoal(
            "work",
            "Complete work",
            "result",
            completion=CompletionCondition("output_present", output_type="result"),
        ),),
        limits=PhaseLimits(1, 10),
    )
    state = new_tactical_state(
        phase,
        cumulative_evidence=[{"evidence_ref": "prior", "phase_id": "prior-phase", "status": "success"}],
    )
    state.evidence.append({"evidence_ref": "current", "phase_id": "current-phase", "status": "success"})
    task = TaskState(goal="Do work", user="alex", project_id="home")

    HierarchicalCAPDProcessor._append_history(
        task,
        state,
        PhaseOutcome("current-phase", PhaseStatus.PHASE_COMPLETE).to_dict(),
    )

    assert task.metadata["v3_phase_history"][0]["evidence"]["entries"] == [
        {"evidence_ref": "current", "phase_id": "current-phase", "status": "success"}
    ]


class _Processor:
    def __init__(self, label: str) -> None:
        self.label = label
        self.calls = 0

    def process(self, task, context):
        self.calls += 1
        return ProcessingResult(snapshot=StateSnapshot(metadata={"engine": self.label}))


def test_engine_router_uses_task_stamp_not_current_global_setting() -> None:
    v2, v3 = _Processor("v2"), _Processor("v3")
    router = EngineRoutingProcessor(v2=v2, v3=v3)
    context = CoreLoopContext(messages=InMemoryMessageQueue())

    result = router.process(TaskState(intelligence_engine=HIERARCHICAL_V3), context)

    assert result.snapshot.metadata["engine"] == "v3"
    assert (v2.calls, v3.calls) == (0, 1)


def test_hierarchical_processor_completes_one_phase_without_v2_tool_cycles() -> None:
    provider = StubInferenceProvider(
        markdown_by_purpose={
            InferencePurpose.ACCEPTANCE_CRITERIA: "1.- [ ] The project record was found",
            InferencePurpose.FINAL_RESPONSE: "Encontré el registro del proyecto.",
        },
        json_by_purpose={
            InferencePurpose.PHASE_PLANNING: {
                "schema_version": 3,
                "phase_id": "locate-project",
                "objective": "Locate the project record",
                "criterion_ids": ["ac-1"],
                "authorized_capabilities": ["project_record_search"],
                "mutation_scope": {"allowed_paths": [], "allow_external_effects": False},
                "limits": {"max_tool_calls": 2, "max_duration_seconds": 20},
                "originating_decision": "initial",
                "subgoals": [{
                    "subgoal_id": "locate",
                    "objective": "Locate the record",
                    "required_output_type": "record_reference",
                    "depends_on": [],
                    "allowed_capabilities": ["project_record_search"],
                    "allowed_side_effects": ["read_only"],
                    "limits": {"max_tool_calls": 1, "max_duration_seconds": 10},
                    "completion": {"kind": "output_present", "output_type": "record_reference"},
                    "failure_policy": "stop",
                }],
            },
            InferencePurpose.TACTICAL_ACTION: {"tool_id": "native.project_search", "arguments": {"query": "solar"}},
            InferencePurpose.PHASE_REVIEW: {"updates": [{
                "criterion_id": "ac-1", "status": "satisfied",
                "evidence_refs": [], "reason": "placeholder",
            }]},
        },
    )
    # The evidence reference is generated dynamically; adapt the test provider at review time.
    original_generate_json = provider.generate_json

    def generate_json(request):
        if request.purpose == InferencePurpose.PHASE_REVIEW:
            import json
            marker = "tactical-action:"
            start = request.prompt.find(marker)
            end = request.prompt.find('"', start)
            provider.json_by_purpose[InferencePurpose.PHASE_REVIEW]["updates"][0]["evidence_refs"] = [request.prompt[start:end]]
        return original_generate_json(request)

    provider.generate_json = generate_json  # type: ignore[method-assign]
    inference = InferenceRouter(provider=provider, default_profile=ModelProfile("test", "test", "default"))
    registry = InMemoryToolRegistry()
    registry.register(ToolDefinition(
        descriptor=ToolDescriptor(
            "native.project_search", "project_search", ToolKind.NATIVE,
            metadata={"v3_capabilities": ["project_record_search"]}, read_only=True,
        ),
        callable=lambda arguments: {"path": "backlog.md", "query": arguments["query"]},
    ))
    task = TaskState(goal="Find solar", user="alex", project_id="home", intelligence_engine=HIERARCHICAL_V3, intelligence_schema_version=3)

    result = HierarchicalCAPDProcessor().process(
        task, CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=inference)
    )

    assert result.status.value == "completed"
    assert task.status == "completed"
    assert task.metadata["v3_route"] == "respond_and_end"
    assert task.metadata["prepared_user_response"]["message"] == "Encontré el registro del proyecto."
    assert InferencePurpose.TOOL_PLANNING not in [item.purpose for item in provider.requests]


def test_hierarchical_processor_plans_one_stage_and_jev_selects_respond_for_greeting(tmp_path: Path) -> None:
    provider = StubInferenceProvider(
        markdown_by_purpose={
            InferencePurpose.ACCEPTANCE_CRITERIA: "1.- [ ] Alex receives a warm greeting",
        },
        json_by_purpose={
            InferencePurpose.PHASE_PLANNING: {
                "schema_version": 3,
                "phase_id": "reply-to-requester",
                "objective": "Reply directly to Alex with a warm greeting",
                "criterion_ids": ["ac-1"],
                "authorized_capabilities": ["user_response"],
                "mutation_scope": {"allowed_paths": [], "allow_external_effects": True},
                "limits": {"max_tool_calls": 1, "max_duration_seconds": 10},
                "originating_decision": "initial",
                "subgoals": [{
                    "subgoal_id": "reply",
                    "objective": "Reply directly to Alex with a warm greeting",
                    "required_output_type": "user_response",
                    "depends_on": [],
                    "allowed_capabilities": ["user_response"],
                    "allowed_side_effects": ["user_response"],
                    "limits": {"max_tool_calls": 1, "max_duration_seconds": 10},
                    "completion": {"kind": "output_present", "output_type": "user_response"},
                    "failure_policy": "stop",
                }],
            },
            InferencePurpose.TACTICAL_ACTION: {
                "tool_id": "native.respond",
                "arguments": {"message": "¡Hola, Alex! Qué gusto saludarte.", "tone": "warm"},
            },
        },
    )
    inference = InferenceRouter(provider=provider, default_profile=ModelProfile("test", "test", "default"))

    class SystemOne:
        def select_plan_tools(self, **values):
            assert values["goal"] == "Hola Alphonse!"
            assert {tool.tool_id for tool in values["tools"]} == {"native.respond", "artifact.medical"}
            return SystemOneToolRegistrySelection(
                selected_tool_ids=("native.respond",),
                rejected_tool_ids=("artifact.medical",),
                probabilities={"native.respond": 0.99, "artifact.medical": 0.01},
                model="jev-latest",
            )

        def evaluate_tactical_progress(self, **_values):
            return SystemOneTacticalReview(True, 0.99, True, model="jev-latest")

        def evaluate(self, **values):
            evidence_ref = values["evidence"]["entries"][-1]["evidence_ref"]
            return SystemOneReviewResult(
                updates=({
                    "criterion_id": "ac-1", "status": "satisfied",
                    "evidence_refs": [evidence_ref], "reason": "The response was produced.",
                },),
                ambiguous_criterion_ids=(),
                recommended_route="complete",
                route_confidence=0.99,
                route_confident=True,
                model="jev-latest",
            )

    registry = InMemoryToolRegistry()
    registry.register(build_respond_tool_definition())
    registry.register(ToolDefinition(
        descriptor=ToolDescriptor(
            "artifact.medical", "medical", ToolKind.ARTIFACT,
            description="Read or update the authorized family medical database.",
            metadata={"v3_capabilities": ["project_artifact_query"]},
        ),
        callable=lambda _arguments: (_ for _ in ()).throw(AssertionError("irrelevant artifact must not run")),
    ))

    task = TaskState(
        task_id="greeting-task", goal="Hola Alphonse!", user="alex", project_id="home",
        intelligence_engine=HIERARCHICAL_V3, intelligence_schema_version=3,
    )
    activity = []
    question_store = SQLiteQuestionStore(tmp_path / "questions.sqlite3")
    result = HierarchicalCAPDProcessor().process(
        task,
        CoreLoopContext(
            messages=InMemoryMessageQueue(),
            tools=registry,
            inference=inference,
            system_one=SystemOne(),
            question_store=question_store,
            activity_sink=activity.append,
        ),
    )

    assert result.status.value == "completed"
    assert task.metadata["v3_route"] == "respond_and_end"
    assert task.metadata["prepared_user_response"]["message"] == "¡Hola, Alex! Qué gusto saludarte."
    assert task.metadata["prepared_user_response"]["source"] == "native.respond"
    assert task.acceptance_criteria_all_complete()
    purposes = [item.purpose for item in provider.requests]
    assert InferencePurpose.ACCEPTANCE_CRITERIA in purposes
    assert InferencePurpose.PHASE_PLANNING in purposes
    assert InferencePurpose.TACTICAL_ACTION in purposes
    assert InferencePurpose.FINAL_RESPONSE not in purposes
    planning_request = next(item for item in provider.requests if item.purpose == InferencePurpose.PHASE_PLANNING)
    assert '"required": ["kind"]' in planning_request.prompt
    assert '"enum": ["read_only", "user_response", "project_mutation"' in planning_request.prompt
    assert "do not declare it unavailable" in planning_request.prompt
    acceptance_request = next(item for item in provider.requests if item.purpose == InferencePurpose.ACCEPTANCE_CRITERIA)
    assert "All criteria are conjunctive" in acceptance_request.prompt
    assert question_store.load_task_checkpoint("greeting-task") is not None
    assert "tactical action" in [event.label for event in activity]


def test_hierarchical_processor_fails_invalid_phase_once_with_controlled_error() -> None:
    provider = StubInferenceProvider(
        markdown_by_purpose={InferencePurpose.ACCEPTANCE_CRITERIA: "1.- [ ] The record is updated"},
        json_by_purpose={
            InferencePurpose.PHASE_PLANNING: {
                "schema_version": 3,
                "phase_id": "invalid",
                "objective": "Update record",
                "criterion_ids": ["ac-1"],
                "authorized_capabilities": ["project_record_search"],
                "mutation_scope": {"allowed_paths": []},
                "limits": {"max_tool_calls": 1, "max_duration_seconds": 20},
                "originating_decision": "initial",
                "subgoals": [{
                    "subgoal_id": "missing-completion",
                    "objective": "Locate record",
                    "required_output_type": "record",
                    "allowed_capabilities": ["project_record_search"],
                }],
            }
        },
    )
    inference = InferenceRouter(provider=provider, default_profile=ModelProfile("test", "test", "default"))
    task = TaskState(
        goal="Update the record", user="alex", project_id="home",
        intelligence_engine=HIERARCHICAL_V3, intelligence_schema_version=3,
    )

    result = HierarchicalCAPDProcessor().process(
        task, CoreLoopContext(messages=InMemoryMessageQueue(), inference=inference)
    )

    assert result.status.value == "failed"
    assert result.error is not None and result.error.startswith("v3_task_failed:v3_phase_plan_invalid:")
    assert len([item for item in provider.requests if item.purpose == InferencePurpose.PHASE_PLANNING]) == 1


def test_phase_budget_failure_is_persisted_as_terminal_checkpoint(tmp_path: Path) -> None:
    phase = PhasePlan(
        "only-phase",
        "Attempt one phase",
        (PhaseSubgoal(
            "work",
            "Attempt work",
            "result",
            completion=CompletionCondition("output_present", output_type="result"),
        ),),
        limits=PhaseLimits(1, 10),
    )
    state = new_tactical_state(phase)
    state.status = PhaseStatus.PHASE_COMPLETE
    task = TaskState(
        task_id="budget-task",
        goal="Complete work",
        user="alex",
        project_id="home",
        intelligence_engine=HIERARCHICAL_V3,
        intelligence_schema_version=3,
    )
    task.set_acceptance_contract_from_markdown("1.- [ ] Work is complete")
    question_store = SQLiteQuestionStore(tmp_path / "questions.sqlite3")
    processor = HierarchicalCAPDProcessor(max_phases=1)
    processor._new_state = lambda _task, _context: state  # type: ignore[method-assign]

    class Executor:
        @staticmethod
        def run(_task, _state, _context):
            return PhaseOutcome("only-phase", PhaseStatus.PHASE_COMPLETE)

    class Outer:
        @staticmethod
        def review_and_route(current_task, _state, _outcome, _context):
            current_task.metadata["v3_route"] = "plan_next_phase"
            return object(), object()

    processor.executor = Executor()  # type: ignore[assignment]
    processor.outer = Outer()  # type: ignore[assignment]

    result = processor.process(
        task,
        CoreLoopContext(messages=InMemoryMessageQueue(), question_store=question_store),
    )

    restored = question_store.load_task_checkpoint("budget-task")
    assert result.status.value == "failed"
    assert restored is not None
    assert restored.status == "failed"
    assert restored.outcome == {
        "status": "failure",
        "reason": "V3 phase budget exhausted without a terminal outcome.",
    }
