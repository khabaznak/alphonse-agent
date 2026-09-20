from __future__ import annotations

from pathlib import Path

from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingResult, StateSnapshot, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRouter, ModelProfile, StubInferenceProvider
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import EngineRoutingProcessor, HierarchicalCAPDProcessor
from alphonse.agent_v2.core.messages import CommunicationChannel, InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.intelligence_engine_settings import HIERARCHICAL_V3, TACTICAL_V2
from alphonse.agent_v2.intelligence_engine_settings import IntelligenceEngineSettings
from alphonse.agent_v2.intelligence_engine_settings import SQLiteIntelligenceEngineSettingsStore


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
