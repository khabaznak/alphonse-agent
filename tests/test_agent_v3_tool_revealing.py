from __future__ import annotations

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import Capability
from alphonse.agent_v2.core.intelligence.v3 import CompletionCondition
from alphonse.agent_v2.core.intelligence.v3 import MutationScope
from alphonse.agent_v2.core.intelligence.v3 import PhasePlan
from alphonse.agent_v2.core.intelligence.v3 import PhaseSubgoal
from alphonse.agent_v2.core.intelligence.v3 import SideEffectClass
from alphonse.agent_v2.core.intelligence.v3 import ToolRevealPolicy
from alphonse.agent_v2.core.intelligence.v3 import new_tactical_state


def _descriptor(
    tool_id: str,
    capability: Capability,
    *,
    read_only: bool = True,
    schema_size: int = 0,
    integration_id: str = "",
) -> ToolDescriptor:
    metadata = {"v3_capabilities": [capability.value]}
    if integration_id:
        metadata["integration_id"] = integration_id
    return ToolDescriptor(
        tool_id=tool_id,
        name=tool_id,
        kind=ToolKind.ARTIFACT if tool_id.startswith("artifact.") else ToolKind.NATIVE,
        argument_schema={"description": "x" * schema_size},
        metadata=metadata,
        read_only=read_only,
    )


def _state(subgoal: PhaseSubgoal, *, mutation_paths=()):
    phase = PhasePlan(
        "phase", "Do work", (subgoal,),
        authorized_capabilities=subgoal.allowed_capabilities,
        mutation_scope=MutationScope(tuple(mutation_paths)),
    )
    return new_tactical_state(phase)


def test_solar_record_discovery_hides_ocr_home_assistant_and_lg() -> None:
    subgoal = PhaseSubgoal(
        "locate", "Locate solar record", "record_reference",
        allowed_capabilities=(Capability.PROJECT_RECORD_SEARCH.value, Capability.MEMORY_RECALL.value),
    )
    state = _state(subgoal)
    available = (
        _descriptor("native.project_search", Capability.PROJECT_RECORD_SEARCH),
        _descriptor("native.search_memory", Capability.MEMORY_RECALL),
        _descriptor("native.analyze_image", Capability.ATTACHMENT_ANALYSIS),
        _descriptor("artifact.home_assistant", Capability.HOME_AUTOMATION),
        _descriptor("artifact.lg", Capability.DEVICE_CONTROL),
    )

    result = ToolRevealPolicy().reveal(TaskState(project_id="home"), state, subgoal, available)

    assert {item.tool_id for item in result.tools} == {"native.project_search", "native.search_memory"}


def test_attachment_analysis_requires_attachment_or_discovered_document() -> None:
    subgoal = PhaseSubgoal(
        "read", "Read prescription", "text",
        allowed_capabilities=(Capability.ATTACHMENT_ANALYSIS.value,),
    )
    state = _state(subgoal)
    tool = _descriptor("native.analyze_image", Capability.ATTACHMENT_ANALYSIS)

    hidden = ToolRevealPolicy().reveal(TaskState(project_id="health"), state, subgoal, (tool,))
    visible = ToolRevealPolicy().reveal(
        TaskState(project_id="health", metadata={"attachments": [{"asset_id": "rx"}]}),
        state, subgoal, (tool,),
    )

    assert hidden.tools == ()
    assert hidden.decisions[0].reason == "attachment_required"
    assert [item.tool_id for item in visible.tools] == ["native.analyze_image"]


def test_plain_markdown_record_does_not_satisfy_ocr_prerequisite() -> None:
    subgoal = PhaseSubgoal(
        "read", "Read prescription", "text",
        allowed_capabilities=(Capability.ATTACHMENT_ANALYSIS.value,),
    )
    state = _state(subgoal)
    state.bindings["record_reference"] = {"value": {"path": "Receta.md"}}

    result = ToolRevealPolicy().reveal(
        TaskState(project_id="health"), state, subgoal,
        (_descriptor("native.analyze_image", Capability.ATTACHMENT_ANALYSIS),),
    )

    assert result.tools == ()


def test_memory_recall_requires_current_project() -> None:
    subgoal = PhaseSubgoal(
        "recall", "Recall treatment", "memory_matches",
        allowed_capabilities=(Capability.MEMORY_RECALL.value,),
    )
    tool = _descriptor("native.search_memory", Capability.MEMORY_RECALL)

    hidden = ToolRevealPolicy().reveal(TaskState(), _state(subgoal), subgoal, (tool,))
    visible = ToolRevealPolicy().reveal(TaskState(project_id="health"), _state(subgoal), subgoal, (tool,))

    assert hidden.tools == ()
    assert [item.tool_id for item in visible.tools] == ["native.search_memory"]


def test_exact_mutation_requires_scope_and_subgoal_authorization() -> None:
    tool = _descriptor("native.exact_text_edit", Capability.EXACT_TEXT_MUTATION, read_only=False)
    read_only_subgoal = PhaseSubgoal(
        "update", "Update", "mutation",
        allowed_capabilities=(Capability.EXACT_TEXT_MUTATION.value,),
    )
    writable_subgoal = PhaseSubgoal(
        "update", "Update", "mutation",
        allowed_capabilities=(Capability.EXACT_TEXT_MUTATION.value,),
        allowed_side_effects=(SideEffectClass.PROJECT_MUTATION,),
        completion=CompletionCondition("output_present", output_type="mutation"),
    )

    no_scope = ToolRevealPolicy().reveal(TaskState(project_id="home"), _state(writable_subgoal), writable_subgoal, (tool,))
    no_write = ToolRevealPolicy().reveal(
        TaskState(project_id="home"), _state(read_only_subgoal, mutation_paths=("a.md",)), read_only_subgoal, (tool,)
    )
    visible = ToolRevealPolicy().reveal(
        TaskState(project_id="home"), _state(writable_subgoal, mutation_paths=("a.md",)), writable_subgoal, (tool,)
    )

    assert no_scope.decisions[0].reason == "mutation_scope_required"
    assert no_write.decisions[0].reason == "project_mutation_not_allowed"
    assert [item.tool_id for item in visible.tools] == ["native.exact_text_edit"]


def test_integration_tool_requires_installed_integration_id() -> None:
    subgoal = PhaseSubgoal(
        "temperature", "Read temperature", "temperature",
        allowed_capabilities=(Capability.DEVICE_CONTROL.value,),
    )
    tool = _descriptor("artifact.lg", Capability.DEVICE_CONTROL, integration_id="lg-client")
    state = _state(subgoal)

    hidden = ToolRevealPolicy().reveal(TaskState(project_id="home"), state, subgoal, (tool,))
    visible = ToolRevealPolicy().reveal(
        TaskState(project_id="home", metadata={"available_integration_ids": ["lg-client"]}), state, subgoal, (tool,)
    )

    assert hidden.decisions[0].reason == "integration_unavailable"
    assert [item.tool_id for item in visible.tools] == ["artifact.lg"]


def test_reveal_enforces_tool_count_and_schema_budgets() -> None:
    subgoal = PhaseSubgoal(
        "search", "Search", "result",
        allowed_capabilities=(Capability.PROJECT_RECORD_SEARCH.value,),
    )
    tools = tuple(
        _descriptor(f"native.search_{index}", Capability.PROJECT_RECORD_SEARCH, schema_size=120)
        for index in range(4)
    )

    result = ToolRevealPolicy(max_tools=2, max_schema_chars=300).reveal(
        TaskState(project_id="home"), _state(subgoal), subgoal, tools
    )

    assert len(result.tools) == 2
    assert any(item.reason in {"tool_count_budget", "schema_budget"} for item in result.decisions if not item.revealed)


def test_capability_catalog_is_compact_and_subgoal_scoped() -> None:
    subgoal = PhaseSubgoal(
        "locate", "Locate", "record",
        allowed_capabilities=(Capability.PROJECT_RECORD_SEARCH.value, Capability.MEMORY_RECALL.value),
    )

    catalog = ToolRevealPolicy().capability_catalog(subgoal)

    assert [item["capability"] for item in catalog] == [
        Capability.PROJECT_RECORD_SEARCH.value,
        Capability.MEMORY_RECALL.value,
    ]
    assert all(item["description"] for item in catalog)
