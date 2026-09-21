"""Capability catalog and deterministic progressive tool revealing."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseSubgoal, SideEffectClass, TacticalState

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import ToolDescriptor
    from alphonse.agent_v2.core.intelligence.task_state import TaskState


class Capability(str, Enum):
    PROJECT_RECORD_SEARCH = "project_record_search"
    PROJECT_FILE_INSPECTION = "project_file_inspection"
    PROJECT_ARTIFACT_QUERY = "project_artifact_query"
    MEMORY_RECALL = "memory_recall"
    ATTACHMENT_ANALYSIS = "attachment_analysis"
    DOCUMENT_EXTRACTION = "document_extraction"
    EXACT_TEXT_MUTATION = "exact_text_mutation"
    COMMUNICATION = "communication"
    SCHEDULING = "scheduling"
    HOME_AUTOMATION = "home_automation"
    DEVICE_CONTROL = "device_control"
    USER_INTERACTION = "user_interaction"
    USER_RESPONSE = "user_response"


@dataclass(frozen=True)
class ToolRevealDecision:
    tool_id: str
    revealed: bool
    capabilities: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class ToolRevealResult:
    tools: tuple["ToolDescriptor", ...]
    decisions: tuple[ToolRevealDecision, ...]
    catalog: tuple[dict[str, str], ...] = ()


@dataclass
class ToolRevealPolicy:
    max_tools: int | None = None
    max_schema_chars: int | None = None
    capability_descriptions: dict[str, str] = field(default_factory=lambda: {
        Capability.PROJECT_RECORD_SEARCH.value: "Find records and indexes inside the authorized project.",
        Capability.PROJECT_FILE_INSPECTION.value: "Inspect authorized project files and local process state.",
        Capability.PROJECT_ARTIFACT_QUERY.value: "Query a registered project artifact through its native client.",
        Capability.MEMORY_RECALL.value: "Search bounded archived memory for the current project.",
        Capability.ATTACHMENT_ANALYSIS.value: "Analyze an image attached to the current task.",
        Capability.DOCUMENT_EXTRACTION.value: "Extract information from a discovered document.",
        Capability.EXACT_TEXT_MUTATION.value: "Atomically change one exact string in an authorized project file.",
        Capability.COMMUNICATION.value: "Deliver a message to another registered user.",
        Capability.SCHEDULING.value: "Create or manage a requested time-based action.",
        Capability.HOME_AUTOMATION.value: "Read or control an authorized home-automation service.",
        Capability.DEVICE_CONTROL.value: "Read or control an authorized device through its native client.",
        Capability.USER_INTERACTION.value: "Ask the requester for information that is necessary to continue.",
        Capability.USER_RESPONSE.value: "Return the final user-visible response.",
    })

    def capability_catalog(self, subgoal: PhaseSubgoal) -> tuple[dict[str, str], ...]:
        return tuple(
            {"capability": capability, "description": self.capability_descriptions.get(capability, capability)}
            for capability in subgoal.allowed_capabilities
        )

    def reveal(
        self,
        task: "TaskState",
        state: TacticalState,
        subgoal: PhaseSubgoal,
        available: tuple["ToolDescriptor", ...],
    ) -> ToolRevealResult:
        allowed = set(subgoal.allowed_capabilities)
        candidates: list[tuple[int, "ToolDescriptor", tuple[str, ...]]] = []
        decisions: list[ToolRevealDecision] = []
        for descriptor in available:
            capabilities = tool_capabilities(descriptor)
            direct_compatibility = descriptor.tool_id in allowed
            matched = set(capabilities) & allowed
            if not matched and not direct_compatibility:
                decisions.append(ToolRevealDecision(descriptor.tool_id, False, capabilities, "capability_not_allowed"))
                continue
            reason = _prerequisite_failure(task, state, subgoal, descriptor, capabilities)
            if reason:
                decisions.append(ToolRevealDecision(descriptor.tool_id, False, capabilities, reason))
                continue
            score = (20 if matched else 0) + (5 if descriptor.read_only else 0) + len(matched)
            candidates.append((score, descriptor, capabilities))
        candidates.sort(key=lambda item: (-item[0], item[1].tool_id))
        selected: list["ToolDescriptor"] = []
        schema_chars = 0
        selected_ids: set[str] = set()
        for _, descriptor, capabilities in candidates:
            size = len(json.dumps(descriptor.argument_schema, ensure_ascii=False))
            if self.max_tools is not None and len(selected) >= max(1, self.max_tools):
                decisions.append(ToolRevealDecision(descriptor.tool_id, False, capabilities, "tool_count_budget"))
                continue
            if self.max_schema_chars is not None and selected and schema_chars + size > max(1, self.max_schema_chars):
                decisions.append(ToolRevealDecision(descriptor.tool_id, False, capabilities, "schema_budget"))
                continue
            selected.append(descriptor)
            selected_ids.add(descriptor.tool_id)
            schema_chars += size
            decisions.append(ToolRevealDecision(descriptor.tool_id, True, capabilities, "relevant_and_authorized"))
        decisions.sort(key=lambda item: item.tool_id)
        return ToolRevealResult(
            tools=tuple(selected),
            decisions=tuple(decisions),
            catalog=self.capability_catalog(subgoal),
        )


def tool_capabilities(descriptor: "ToolDescriptor") -> tuple[str, ...]:
    explicit = tuple(str(item).strip() for item in descriptor.metadata.get("v3_capabilities", ()) if str(item).strip())
    if explicit:
        return explicit
    mapped = _TOOL_CAPABILITIES.get(descriptor.tool_id)
    if mapped:
        return mapped
    raw = set(descriptor.capabilities) | set(descriptor.tags)
    derived: set[str] = set()
    if "memory" in raw:
        derived.add(Capability.MEMORY_RECALL.value)
    if "ocr" in raw or "vision" in raw or "attachments" in raw:
        derived.add(Capability.ATTACHMENT_ANALYSIS.value)
    if "communication" in raw or "delivery" in raw:
        derived.add(Capability.COMMUNICATION.value)
    if "schedule" in raw or "scheduling" in raw:
        derived.add(Capability.SCHEDULING.value)
    if descriptor.kind.value == "artifact":
        derived.add(Capability.PROJECT_ARTIFACT_QUERY.value)
    return tuple(sorted(derived))


def _prerequisite_failure(
    task: "TaskState",
    state: TacticalState,
    subgoal: PhaseSubgoal,
    descriptor: "ToolDescriptor",
    capabilities: tuple[str, ...],
) -> str:
    if descriptor.tool_id == "native.bash":
        return "unbounded_shell_disabled_in_v3"
    capability_set = set(capabilities)
    if Capability.ATTACHMENT_ANALYSIS.value in capability_set and not _has_analyzable_attachment(task, state):
        return "attachment_required"
    if Capability.MEMORY_RECALL.value in capability_set and not str(task.project_id or "").strip():
        return "authorized_project_required"
    if Capability.EXACT_TEXT_MUTATION.value in capability_set:
        if not state.phase.mutation_scope.allowed_paths:
            return "mutation_scope_required"
        if SideEffectClass.PROJECT_MUTATION not in subgoal.allowed_side_effects:
            return "project_mutation_not_allowed"
    if not descriptor.read_only and not subgoal.allowed_side_effects:
        return "side_effect_not_allowed"
    required_integration = str(descriptor.metadata.get("integration_id") or "").strip()
    if required_integration:
        available = set(str(item) for item in task.metadata.get("available_integration_ids") or [])
        if required_integration not in available:
            return "integration_unavailable"
    return ""


def _has_analyzable_attachment(task: "TaskState", state: TacticalState) -> bool:
    if task.metadata.get("attachments") or task.metadata.get("asset_ids"):
        return True
    for binding in state.bindings.values():
        rendered = json.dumps(binding, ensure_ascii=False).lower()
        if any(extension in rendered for extension in (".png", ".jpg", ".jpeg", ".webp", ".pdf")):
            return True
    return False


_TOOL_CAPABILITIES: dict[str, tuple[str, ...]] = {
    "native.bash": (Capability.PROJECT_FILE_INSPECTION.value,),
    "native.project_search": (Capability.PROJECT_RECORD_SEARCH.value, Capability.PROJECT_FILE_INSPECTION.value),
    "native.read_project_file": (Capability.PROJECT_FILE_INSPECTION.value,),
    "native.search_memory": (Capability.MEMORY_RECALL.value,),
    "native.analyze_image": (Capability.ATTACHMENT_ANALYSIS.value,),
    "native.exact_text_edit": (Capability.EXACT_TEXT_MUTATION.value,),
    "native.deliver_message": (Capability.COMMUNICATION.value,),
    "native.scheduled_task": (Capability.SCHEDULING.value,),
    "native.ask_question": (Capability.USER_INTERACTION.value,),
    "native.respond": (Capability.USER_RESPONSE.value,),
}
