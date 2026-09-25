from __future__ import annotations

from pathlib import Path

from alphonse.agent_v2.core.core import CoreLoopContext, ImprovementPhase, ProcessingResult, StateSnapshot, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult, InferenceRouter, ModelProfile, StubInferenceProvider
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import CompletionCondition
from alphonse.agent_v2.core.intelligence.v3 import EngineRoutingProcessor, HierarchicalCAPDProcessor
from alphonse.agent_v2.core.intelligence.v3 import PhaseOutcome, PhasePlan, PhaseStatus, PhaseSubgoal
from alphonse.agent_v2.core.intelligence.v3 import new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.processor import _deduplicated_history_evidence
from alphonse.agent_v2.core.intelligence.v3 import processor as v3_processor_module
from alphonse.agent_v2.core.messages import CommunicationChannel, InMemoryMessageQueue
from alphonse.agent_v2.core.projects import ProjectStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.core.tools.registry.native.exact_text_edit import build_exact_text_edit_tool_definition
from alphonse.agent_v2.core.tools.registry.native.project_files import build_project_read_tool_definition
from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
from alphonse.agent_v2.intelligence_engine_settings import HIERARCHICAL_V3, TACTICAL_V2
from alphonse.agent_v2.intelligence_engine_settings import IntelligenceEngineSettings
from alphonse.agent_v2.intelligence_engine_settings import SQLiteIntelligenceEngineSettingsStore
from alphonse.agent_v2.system_one import SystemOneReviewResult
from alphonse.agent_v2.system_one import SystemOneActRecommendation
from alphonse.agent_v2.system_one import SystemOneTacticalReview
from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection


class _TestJev:
    def curate_request_tools(self, *, tools, **_values):
        return SystemOneToolRegistrySelection(selected_tool_ids=tuple(item.tool_id for item in tools))

    def select_plan_tools(self, *, goal, phase, tools):
        _ = goal, phase
        return SystemOneToolRegistrySelection(selected_tool_ids=tuple(item.tool_id for item in tools))

    def evaluate(self, *, contract, phase, evidence):
        _ = phase
        entries = evidence.get("entries") or []
        refs = [str(item.get("evidence_ref")) for item in entries if item.get("status") == "success"]
        updates = tuple({"criterion_id": str(item.get("id")), "status": "satisfied", "evidence_refs": refs[-1:]}
                        for item in contract.get("criteria") or [] if item.get("status") != "satisfied" and refs)
        return SystemOneReviewResult(updates, (), model="test-jev")

    def evaluate_tactical_progress(self, *, questions=None, **_values):
        return SystemOneTacticalReview(True, 0.99, True, model="test-jev", answers={
            str(item.get("question_id")): 0.99 for item in questions or []
        })

    def recommend_act(self, *, state):
        action = "complete" if state.get("check_verdict") == "success" else "continue"
        return SystemOneActRecommendation(action, 0.99, True, f"Jev recommends {action}.", answers={
            "continuation_is_worthwhile": 0.99,
            "user_input_can_unblock": 0.1,
            "closure_explanation_is_warranted": 0.99,
        })


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
                "acceptance_criteria": ["The project record was found"],
                "phase_id": "locate-project",
                "objective": "Locate the project record",
                "criterion_ids": ["ac-1"],
                "authorized_capabilities": ["project_record_search"],
                "mutation_scope": {"allowed_paths": [], "allow_external_effects": False},
                "originating_decision": "initial",
                "subgoals": [{
                    "subgoal_id": "locate",
                    "objective": "Locate the record",
                    "required_output_type": "record_reference",
                    "depends_on": [],
                    "allowed_capabilities": ["project_record_search"],
                    "allowed_side_effects": ["read_only"],
                    "completion": {"kind": "output_present", "output_type": "record_reference"},
                    "failure_policy": "stop",
                }],
            },
            InferencePurpose.TACTICAL_ACTION: {
                "tool_id": "native.project_search",
                "arguments": {"query": "solar"},
                "acceptance_questions": [{
                    "question_id": "record-found",
                    "type": "noul",
                    "instructions": "Did the search find the requested project record?",
                    "criteria": {"true": "The record was found.", "false": "The record was not found."},
                }],
            },
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
        task, CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=inference, system_one=_TestJev())
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
                "acceptance_criteria": ["Alex receives a warm greeting"],
                "phase_id": "reply-to-requester",
                "objective": "Reply directly to Alex with a warm greeting",
                "criterion_ids": ["ac-1"],
                "authorized_capabilities": ["user_response"],
                "mutation_scope": {"allowed_paths": [], "allow_external_effects": True},
                "originating_decision": "initial",
                "subgoals": [{
                    "subgoal_id": "reply",
                    "objective": "Reply directly to Alex with a warm greeting",
                    "required_output_type": "user_response",
                    "depends_on": [],
                    "allowed_capabilities": ["user_response"],
                    "allowed_side_effects": ["user_response"],
                    "completion": {"kind": "output_present", "output_type": "user_response"},
                    "failure_policy": "stop",
                }],
            },
            InferencePurpose.TACTICAL_ACTION: {
                "tool_id": "native.respond",
                "arguments": {"message": "¡Hola, Alex! Qué gusto saludarte.", "tone": "warm"},
                "acceptance_questions": [{
                    "question_id": "greeting-produced",
                    "type": "noul",
                    "instructions": "Was the requested greeting produced?",
                    "criteria": {"true": "A greeting was produced.", "false": "No greeting was produced."},
                }],
            },
        },
    )
    inference = InferenceRouter(provider=provider, default_profile=ModelProfile("test", "test", "default"))

    class SystemOne:
        def curate_request_tools(self, *, tools, **_values):
            return SystemOneToolRegistrySelection(selected_tool_ids=tuple(item.tool_id for item in tools))

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

        def recommend_act(self, *, state):
            action = "complete" if state.get("check_verdict") == "success" else "continue"
            return SystemOneActRecommendation(action, 0.99, True, "Verified greeting.", answers={
                "continuation_is_worthwhile": 0.99, "user_input_can_unblock": 0.1,
                "closure_explanation_is_warranted": 0.99,
            })

        def recommend_act(self, *, state):
            from alphonse.agent_v2.system_one import SystemOneActRecommendation
            return SystemOneActRecommendation(
                "complete", 0.99, True, "Acceptance is verified.",
                answers={"continuation_is_worthwhile": 0.1, "user_input_can_unblock": 0.1, "closure_explanation_is_warranted": 0.9},
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
    assert "use native.artifact_metadata_update for the mutation" in planning_request.prompt
    assert "editing a README or program is not a substitute" in planning_request.prompt
    assert "plan the necessary duplicate/context inspection, the requested mutation, and verification as one coherent end-to-end phase" in planning_request.prompt
    acceptance_request = next(item for item in provider.requests if item.purpose == InferencePurpose.ACCEPTANCE_CRITERIA)
    assert "All criteria are conjunctive" in acceptance_request.prompt
    assert question_store.load_task_checkpoint("greeting-task") is not None
    assert "tactical action" in [event.label for event in activity]


def test_every_v3_plan_pass_emits_a_plan_activity(monkeypatch) -> None:
    task = TaskState(goal="Continue the task", user="alex", project_id="home")
    task.metadata["v3_route"] = "strategic_replan"
    events = []
    context = CoreLoopContext(messages=InMemoryMessageQueue(), activity_sink=events.append)
    phase = PhasePlan(
        "replanned", "Continue with the next execution phase",
        (PhaseSubgoal(
            "next", "Do the next step", "result",
            completion=CompletionCondition("output_present", output_type="result"),
        ),),
    )
    monkeypatch.setattr(v3_processor_module, "plan_phase", lambda _task, _context: phase)

    state = HierarchicalCAPDProcessor._new_state(task, context)

    assert state.phase.phase_id == "replanned"
    assert len(events) == 1
    assert events[0].phase == ImprovementPhase.PLAN
    assert events[0].label == "planning phase"
    assert events[0].progress["route"] == "strategic_replan"


def test_hierarchical_processor_fails_invalid_phase_once_with_controlled_error() -> None:
    provider = StubInferenceProvider(
        markdown_by_purpose={InferencePurpose.ACCEPTANCE_CRITERIA: "1.- [ ] The record is updated"},
        json_by_purpose={
            InferencePurpose.PHASE_PLANNING: {
                "schema_version": 3,
                "acceptance_criteria": ["The record is updated"],
                "phase_id": "invalid",
                "objective": "Update record",
                "criterion_ids": ["ac-1"],
                "authorized_capabilities": ["project_record_search"],
                "mutation_scope": {"allowed_paths": []},
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
        task, CoreLoopContext(messages=InMemoryMessageQueue(), inference=inference, system_one=_TestJev())
    )

    assert result.status.value == "failed"
    assert result.error is not None and result.error.startswith("v3_task_failed:v3_phase_plan_invalid:")
    assert len([item for item in provider.requests if item.purpose == InferencePurpose.PHASE_PLANNING]) == 1


def test_v3_phase_planner_receives_latest_user_tool_hint() -> None:
    provider = StubInferenceProvider(
        json_by_purpose={
            InferencePurpose.PHASE_PLANNING: {
                "schema_version": 3,
                "acceptance_criteria": ["The device temperature is reported"],
                "phase_id": "read-device-temperature",
                "objective": "Use the requested device-status tool to read the studio temperature",
                "criterion_ids": ["ac-1"],
                "authorized_capabilities": ["device_control"],
                "mutation_scope": {"allowed_paths": [], "allow_external_effects": False},
                "originating_decision": "The user named the tool in the latest answer.",
                "subgoals": [{
                    "subgoal_id": "read-temperature",
                    "objective": "Query the named device-status tool for the current studio temperature",
                    "required_output_type": "temperature",
                    "depends_on": [],
                    "allowed_capabilities": ["device_control"],
                    "allowed_side_effects": ["read_only"],
                    "completion": {"kind": "output_present", "output_type": "temperature"},
                    "failure_policy": "stop",
                }],
            }
        },
    )
    inference = InferenceRouter(provider=provider, default_profile=ModelProfile("test", "test", "default"))
    registry = InMemoryToolRegistry()
    registry.register(ToolDefinition(
        descriptor=ToolDescriptor(
            "native.device_status", "device_status", ToolKind.NATIVE,
            description="Query authorized device status.",
            metadata={"v3_capabilities": ["device_control"]}, read_only=True,
        ),
        callable=lambda _arguments: {"temperature_c": 23},
    ))
    task = TaskState(
        goal="Qué temperatura tenemos en el estudio?",
        user="alex",
        project_id="home",
        recent_conversation_md=(
            '- alex: "Qué temperatura tenemos en el estudio?"\n'
            '- alex: "usa la herramienta device status"'
        ),
    )
    task.set_acceptance_contract_from_markdown("1.- [ ] La temperatura del estudio se obtiene")

    state = HierarchicalCAPDProcessor._new_state(
        task,
        CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=inference, system_one=_TestJev()),
    )

    request = next(item for item in provider.requests if item.purpose == InferencePurpose.PHASE_PLANNING)
    assert 'usa la herramienta device status' in request.prompt
    assert '"capability": "device_control"' in request.prompt
    assert "Read or control an authorized device" in request.prompt
    assert state.phase.phase_id == "read-device-temperature"


def test_v3_phase_planner_receives_project_context_and_durable_memory(tmp_path: Path) -> None:
    provider = StubInferenceProvider(
        json_by_purpose={
            InferencePurpose.PHASE_PLANNING: {
                "schema_version": 3,
                "acceptance_criteria": ["The journal is updated"],
                "phase_id": "read-known-journal",
                "objective": "Read the known journal before updating it",
                "criterion_ids": ["ac-1"],
                "authorized_capabilities": ["project_file_inspection"],
                "mutation_scope": {"allowed_paths": [], "allow_external_effects": False},
                "originating_decision": "The journal path is established in durable project memory.",
                "subgoals": [{
                    "subgoal_id": "read-journal",
                    "objective": "Read calisthenics_journal.md",
                    "required_output_type": "journal_contents",
                    "depends_on": [],
                    "allowed_capabilities": ["project_file_inspection"],
                    "allowed_side_effects": ["read_only"],
                    "completion": {"kind": "output_present", "output_type": "journal_contents"},
                    "failure_policy": "stop",
                }],
            }
        },
    )
    inference = InferenceRouter(provider=provider, default_profile=ModelProfile("test", "test", "default"))
    projects = ProjectStore(":memory:")
    root = tmp_path / "calisthenics"
    project = projects.create_project(
        name="Calisthenics", description="Workout coaching and tracking",
        root_path=str(root), owner_user_id="alex",
    )
    projects.write_project_context(
        project.project_id, "Keep a dated workout journal.", requester_user_id="alex",
    )
    task = TaskState(
        goal="Update today's workout",
        user="alex",
        project_id=project.project_id,
        conversation_history_md=(
            "# Durable Project Memory\n"
            f"Primary journal path: {root / 'calisthenics_journal.md'}\n"
            "Never duplicate a same-date entry.\n"
            + ("archived context " * 900)
            + "\n# Recent Session Events\nLatest workout correction remains active."
        ),
        recent_conversation_md='- alex: "I completed 5 sets of push-ups"',
    )
    task.set_acceptance_contract_from_markdown("1.- [ ] Today's workout is recorded")

    state = HierarchicalCAPDProcessor._new_state(
        task,
        CoreLoopContext(messages=InMemoryMessageQueue(), project_store=projects, inference=inference, system_one=_TestJev()),
    )

    request = next(item for item in provider.requests if item.purpose == InferencePurpose.PHASE_PLANNING)
    assert f"Project Directory: {root}" in request.prompt
    assert "Keep a dated workout journal." in request.prompt
    assert str(root / "calisthenics_journal.md") in request.prompt
    assert "Latest workout correction remains active." in request.prompt
    assert "context truncated" in request.prompt
    assert 'I completed 5 sets of push-ups' in request.prompt
    assert "never mutation targets" in request.prompt
    assert "Do not ask the requester for a file location already present" in request.prompt
    assert state.phase.mutation_scope.allowed_paths == ()


def test_v3_known_journal_path_flows_from_memory_to_read_and_verified_edit(tmp_path: Path) -> None:
    root = tmp_path / "project"
    root.mkdir()
    journal = root / "workout_journal.md"
    journal.write_text("# Workout Journal\n\n- Push-ups: 3 sets\n", encoding="utf-8")
    projects = ProjectStore(":memory:")
    project = projects.create_project(name="Workout", root_path=str(root), owner_user_id="alex")

    read_phase = {
        "schema_version": 3,
        "acceptance_criteria": ["The workout entry is recorded"],
        "phase_id": "read-known-journal",
        "objective": "Read the journal path established by project memory",
        "criterion_ids": ["ac-1"],
        "authorized_capabilities": ["project_file_inspection"],
        "mutation_scope": {"allowed_paths": [], "allow_external_effects": False},
        "originating_decision": "Durable memory establishes workout_journal.md as a read candidate.",
        "subgoals": [{
            "subgoal_id": "read-journal",
            "objective": "Read workout_journal.md before editing it",
            "required_output_type": "journal_contents",
            "depends_on": [],
            "allowed_capabilities": ["project_file_inspection"],
            "allowed_side_effects": ["read_only"],
            "completion": {"kind": "output_present", "output_type": "journal_contents"},
            "failure_policy": "stop",
        }],
    }
    edit_phase = {
        "schema_version": 3,
        "acceptance_criteria": [],
        "phase_id": "update-known-journal",
        "objective": "Append the verified workout entry using an exact edit",
        "criterion_ids": ["ac-1"],
        "authorized_capabilities": ["exact_text_mutation"],
        "mutation_scope": {"allowed_paths": ["workout_journal.md"], "allow_external_effects": False},
        "originating_decision": "The successful read established the exact file and anchor text.",
        "subgoals": [{
            "subgoal_id": "update-journal",
            "objective": "Append today's workout and verify the write",
            "required_output_type": "verified_mutation",
            "depends_on": [],
            "allowed_capabilities": ["exact_text_mutation"],
            "allowed_side_effects": ["project_mutation"],
            "completion": {"kind": "field_equals", "field": "verification.status", "expected": "verified"},
            "failure_policy": "stop",
        }],
    }

    class SequencedProvider:
        def __init__(self):
            self.requests = []
            self.phases = [read_phase, edit_phase]
            self.actions = [
                {"tool_id": "native.read_project_file", "arguments": {"path": "workout_journal.md"}},
                {
                    "tool_id": "native.exact_text_edit",
                    "arguments": {
                        "path": "workout_journal.md",
                        "expected_text": "- Push-ups: 3 sets\n",
                        "replacement_text": "- Push-ups: 3 sets\n\n## 2026-09-21\n- Push-ups: 5 sets x 10 reps\n",
                    },
                },
            ]

        def generate_json(self, request):
            self.requests.append(request)
            values = self.phases if request.purpose == InferencePurpose.PHASE_PLANNING else self.actions
            return InferenceResult(json_value=values.pop(0), model_profile=request.model_profile)

        def generate_markdown(self, request):
            self.requests.append(request)
            return InferenceResult(content="Workout journal updated.", model_profile=request.model_profile)

        def plan_tool_call(self, request):
            raise AssertionError(f"Unexpected tool-planning request: {request.purpose}")

    class SystemOne:
        def curate_request_tools(self, *, tools, **_values):
            return SystemOneToolRegistrySelection(selected_tool_ids=tuple(item.tool_id for item in tools))

        def select_plan_tools(self, **values):
            selected = (
                "native.read_project_file" if values["phase"]["phase_id"] == "read-known-journal"
                else "native.exact_text_edit"
            )
            return SystemOneToolRegistrySelection(selected_tool_ids=(selected,))

        def evaluate_tactical_progress(self, **_values):
            return SystemOneTacticalReview(True, 0.99, True)

        def evaluate(self, **values):
            evidence_ref = values["evidence"]["entries"][-1]["evidence_ref"]
            edited = values["phase"]["phase_id"] == "update-known-journal"
            return SystemOneReviewResult(
                updates=({
                    "criterion_id": "ac-1",
                    "status": "satisfied" if edited else "pending",
                    "evidence_refs": [evidence_ref] if edited else [],
                    "reason": "Verified edit" if edited else "Read completed; mutation remains.",
                },),
                ambiguous_criterion_ids=(),
                recommended_route="complete" if edited else "continue",
                route_confidence=0.99,
                route_confident=True,
            )

        def recommend_act(self, *, state):
            action = "complete" if state.get("check_verdict") == "success" else "continue"
            return SystemOneActRecommendation(action, 0.99, True, f"Jev recommends {action}.", answers={
                "continuation_is_worthwhile": 0.99, "user_input_can_unblock": 0.1,
                "closure_explanation_is_warranted": 0.99,
            })

    provider = SequencedProvider()
    inference = InferenceRouter(provider=provider, default_profile=ModelProfile("test", "test", "default"))
    registry = InMemoryToolRegistry()
    registry.register(build_project_read_tool_definition())
    registry.register(build_exact_text_edit_tool_definition())
    task = TaskState(
        task_id="journal-update",
        goal="Record today's workout",
        user="alex",
        project_id=project.project_id,
        intelligence_engine=HIERARCHICAL_V3,
        intelligence_schema_version=3,
        conversation_history_md=(
            "# Durable Project Memory\nPrimary journal path: " + str(journal)
        ),
    )
    task.set_acceptance_contract_from_markdown("1.- [ ] Today's workout is recorded in the journal")

    result = HierarchicalCAPDProcessor().process(
        task,
        CoreLoopContext(
            messages=InMemoryMessageQueue(), tools=registry, inference=inference,
            project_store=projects, system_one=SystemOne(),
        ),
    )

    assert result.status.value == "completed"
    assert "## 2026-09-21" in journal.read_text(encoding="utf-8")
    tool_ids = [
        entry["tool_id"]
        for phase in task.metadata["v3_phase_history"]
        for entry in phase["evidence"]["entries"]
    ]
    assert tool_ids == ["native.read_project_file", "native.exact_text_edit"]
    assert "native.ask_question" not in tool_ids


def test_v3_answered_question_replans_with_named_tool_instead_of_repeating_parked_subgoal() -> None:
    waiting_phase = PhasePlan(
        "ask-source",
        "Ask for a temperature source",
        (PhaseSubgoal(
            "ask",
            "Ask Alex for a source",
            "source",
            allowed_capabilities=("user_interaction",),
            completion=CompletionCondition("output_present", output_type="source"),
        ),),
        authorized_capabilities=("user_interaction",),
    )
    waiting_state = new_tactical_state(waiting_phase)
    waiting_state.status = PhaseStatus.WAITING_USER
    resumed_phase = PhasePlan(
        "use-device-status",
        "Use the device-status tool named in the answer",
        (PhaseSubgoal(
            "read",
            "Read the device status",
            "temperature",
            allowed_capabilities=("device_control",),
            completion=CompletionCondition("output_present", output_type="temperature"),
        ),),
        authorized_capabilities=("device_control",),
    )
    resumed_state = new_tactical_state(resumed_phase)
    task = TaskState(
        task_id="temperature-task",
        goal="Qué temperatura tenemos en el estudio?",
        user="alex",
        project_id="home",
        recent_conversation_md='- alex: "usa la herramienta device status"',
        hierarchical_state=waiting_state.to_dict(),
        intelligence_engine=HIERARCHICAL_V3,
        intelligence_schema_version=3,
    )
    task.set_acceptance_contract_from_markdown("1.- [ ] La temperatura del estudio se obtiene")
    seen_phase_ids: list[str] = []
    processor = HierarchicalCAPDProcessor()
    processor._new_state = lambda _task, _context: resumed_state  # type: ignore[method-assign]

    class Executor:
        @staticmethod
        def run(_task, state, _context):
            seen_phase_ids.append(state.phase.phase_id)
            state.status = PhaseStatus.PHASE_COMPLETE
            return PhaseOutcome(state.phase.phase_id, PhaseStatus.PHASE_COMPLETE)

    class Outer:
        @staticmethod
        def review_and_route(current_task, _state, _outcome, _context):
            current_task.status = "completed"
            current_task.metadata["v3_route"] = "respond_and_end"
            return object(), object()

    processor.executor = Executor()  # type: ignore[assignment]
    processor.outer = Outer()  # type: ignore[assignment]

    processor.process(task, CoreLoopContext(messages=InMemoryMessageQueue()))

    assert seen_phase_ids == ["use-device-status"]


def test_processor_continues_until_a_real_terminal_state(tmp_path: Path) -> None:
    phase = PhasePlan(
        "only-phase",
        "Attempt one phase",
        (PhaseSubgoal(
            "work",
            "Attempt work",
            "result",
            completion=CompletionCondition("output_present", output_type="result"),
        ),),
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
    processor = HierarchicalCAPDProcessor()
    processor._new_state = lambda _task, _context: new_tactical_state(phase)  # type: ignore[method-assign]
    phase_count = 0

    class Executor:
        @staticmethod
        def run(_task, _state, _context):
            nonlocal phase_count
            phase_count += 1
            return PhaseOutcome("only-phase", PhaseStatus.PHASE_COMPLETE)

    class Outer:
        @staticmethod
        def review_and_route(current_task, _state, _outcome, _context):
            if phase_count < 10:
                current_task.metadata["v3_route"] = "plan_next_phase"
            else:
                current_task.status = "completed"
                current_task.outcome = {"status": "success", "reason": "done"}
                current_task.metadata["v3_route"] = "respond_and_end"
            return object(), object()

    processor.executor = Executor()  # type: ignore[assignment]
    processor.outer = Outer()  # type: ignore[assignment]

    result = processor.process(
        task,
        CoreLoopContext(messages=InMemoryMessageQueue(), question_store=question_store),
    )

    restored = question_store.load_task_checkpoint("budget-task")
    assert result.status.value == "completed"
    assert restored is not None
    assert restored.status == "completed"
    assert restored.outcome == {"status": "success", "reason": "done"}
    assert phase_count == 10
