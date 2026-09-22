"""Strategic phase planning for hierarchical CAPD."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.intelligence.v3.contracts import FailurePolicy, PhasePlan, SideEffectClass
from alphonse.agent_v2.core.intelligence.v3.contracts import SUPPORTED_COMPLETION_KINDS, V3_SCHEMA_VERSION
from alphonse.agent_v2.core.intelligence.v3.revealing import tool_capabilities
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.system_one import SystemOneUnavailableError

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState


class V3PhasePlanValidationError(ValueError):
    """System Two returned a phase that violates the durable V3 contract."""


def plan_phase(task: "TaskState", context: "CoreLoopContext") -> PhasePlan:
    if context.inference is None:
        raise RuntimeError("v3_phase_planning_inference_unavailable")
    tools = tuple(context.tools.list()) if context.tools is not None else ()
    catalog = sorted({capability for tool in tools for capability in tool_capabilities(tool)})
    capability_descriptions = ToolRevealPolicy().capability_descriptions
    capability_catalog = [
        {"capability": capability, "description": capability_descriptions.get(capability, capability)}
        for capability in catalog
    ]
    project_context = _project_context(task, context)
    durable_memory = _bounded_context_text(task.conversation_history_md, 9000)
    prepared_response = task.metadata.get("prepared_user_response")
    contract_schema = _phase_plan_json_schema(catalog)
    prompt = (
        "Plan one bounded strategic execution phase. Return one JSON object matching the PhasePlan contract. "
        "Use meaningful subgoals, not one outer CAPD cycle per tool. "
        "When replying to the requester is itself part or all of the goal, include a user_response subgoal; "
        "a purely conversational request can be a single-stage response phase. "
        "Plan toward the requested outcome; do not declare it unavailable merely because Plan sees abstract "
        "capability identifiers rather than concrete tools. Tool curation will use the complete tool registry and "
        "this complete phase plan after you return it. Do not create a user_response-only phase that claims inability unless prior verified evidence "
        "shows that relevant retrieval or action tools were attempted and failed. "
        "Authorize only capabilities and project-relative mutation paths needed in this phase. "
        "Never target .alphonse, memory ledgers, prompts, plans, acceptance criteria, or other agent-internal state. "
        "Project context and durable memory are trusted contextual evidence for choosing read candidates, but are never mutation targets. "
        "Convert any absolute path under Project Directory to its project-relative form before calling a project tool. "
        "A mutation path must already be established by the user or prior verified read evidence; otherwise plan a read-only discovery phase first. "
        "Do not ask the requester for a file location already present in project context, durable memory, recent conversation, or prior verified evidence. "
        "If a prepared user response already exists, do not generate the same response again; plan only work that can add missing evidence. "
        "Use native.project_search and native.read_project_file for bounded file discovery; native.bash is unavailable in V3.\n\n"
        f"Goal: {task.goal}\n"
        f"Project context:\n{_bounded_text(project_context, 5000)}\n"
        f"Durable project/session memory (context only; never mutate it):\n{durable_memory}\n"
        f"Recent conversation (newest steering and answers are authoritative):\n{_bounded_text(task.recent_conversation_md, 6000)}\n"
        f"Known task facts:\n{_bounded_text(task.facts_md, 4000)}\n"
        f"Prepared user response already exists: {'yes' if isinstance(prepared_response, dict) else 'no'}\n"
        f"Immutable acceptance contract: {json.dumps(task.ensure_acceptance_contract(), ensure_ascii=False)}\n"
        f"Prior V3 phase history: {json.dumps(task.metadata.get('v3_phase_history') or [], ensure_ascii=False)}\n"
        f"Available capability catalog: {json.dumps(capability_catalog, ensure_ascii=False)}\n"
        "Return an object conforming exactly to this JSON Schema. Do not omit required nested fields "
        "or invent enum values. Use the user_response side effect for a subgoal whose effect is replying "
        f"to the requester.\nPhasePlan JSON Schema: {json.dumps(contract_schema, ensure_ascii=False, sort_keys=True)}"
    )
    result = context.inference.generate_json(
        InferenceRequest(
            prompt=prompt,
            purpose=InferencePurpose.PHASE_PLANNING,
            project_id=task.project_id,
            user=task.user,
            task_id=task.task_id,
            tools=(),
            cancel_checker=context.is_cancelled if context.cancellation_checker is not None else None,
        )
    )
    if not isinstance(result.json_value, dict):
        raise ValueError("v3_phase_plan_missing")
    try:
        phase = PhasePlan.from_dict(result.json_value)
    except (TypeError, ValueError) as exc:
        raise V3PhasePlanValidationError(f"v3_phase_plan_invalid:{exc}") from exc
    return _curate_phase_tools(task, phase, tools, context)


def _curate_phase_tools(task, phase: PhasePlan, tools, context) -> PhasePlan:
    if context.system_one is None or not hasattr(context.system_one, "select_plan_tools"):
        raise SystemOneUnavailableError("tool_curation_unavailable")
    try:
        selection = context.system_one.select_plan_tools(
            goal=task.goal, phase=phase.to_dict(), tools=tools,
        )
    except Exception as exc:
        raise SystemOneUnavailableError(f"tool_curation:{type(exc).__name__}") from exc
    else:
        metadata = {"status": "used", **selection.to_metadata()}
        selected = tuple(dict.fromkeys((*selection.selected_tool_ids, *selection.ambiguous_tool_ids)))
        curated = replace(phase, curated_tool_ids=selected, tool_curation_status="used")
    _record_tool_curation(task, phase, metadata)
    context.emit_telemetry({
        "event": "system_one_tool_registry_selection", "task_id": task.task_id,
        "phase_id": phase.phase_id, **metadata,
    })
    return curated


def _record_tool_curation(task, phase: PhasePlan, metadata: dict) -> None:
    history = task.metadata.setdefault("system_one_tool_registry_selections", [])
    if not isinstance(history, list):
        history = []
        task.metadata["system_one_tool_registry_selections"] = history
    history.append({"phase_id": phase.phase_id, **metadata})
    if len(history) > 20:
        del history[:-20]


def _project_context(task: "TaskState", context: "CoreLoopContext") -> str:
    if context.project_store is None or not str(task.project_id or "").strip():
        return "- (none)"
    render = getattr(context.project_store, "render_project_context", None)
    if not callable(render):
        return "- (none)"
    try:
        return str(render(task.project_id, requester_user_id=task.user) or "").strip() or "- (none)"
    except (KeyError, OSError, PermissionError):
        return "- (none)"


def _bounded_text(value: object, max_chars: int) -> str:
    rendered = str(value or "").strip() or "- (none)"
    if len(rendered) <= max_chars:
        return rendered
    return rendered[-max(1, max_chars - 18):].lstrip() + "\n... truncated"


def _bounded_context_text(value: object, max_chars: int) -> str:
    """Keep durable baseline facts from the front and recent events from the end."""
    rendered = str(value or "").strip() or "- (none)"
    if len(rendered) <= max_chars:
        return rendered
    marker = "\n... context truncated ...\n"
    available = max(2, max_chars - len(marker))
    front = available // 2
    return rendered[:front].rstrip() + marker + rendered[-(available - front):].lstrip()


def _phase_plan_json_schema(capabilities: list[str]) -> dict[str, object]:
    limits = {
        "type": "object",
        "additionalProperties": False,
        "required": ["max_tool_calls", "max_duration_seconds"],
        "properties": {
            "max_tool_calls": {"type": "integer", "minimum": 1},
            "max_duration_seconds": {"type": "number", "exclusiveMinimum": 0},
        },
    }
    capability = {"type": "string", "enum": capabilities}
    completion = {
        "type": "object",
        "additionalProperties": False,
        "required": ["kind"],
        "properties": {
            "kind": {"type": "string", "enum": list(SUPPORTED_COMPLETION_KINDS)},
            "output_type": {"type": "string"},
            "field": {"type": "string"},
            "expected": {},
        },
    }
    subgoal = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "subgoal_id", "objective", "required_output_type", "depends_on",
            "allowed_capabilities", "allowed_side_effects", "limits", "completion", "failure_policy",
        ],
        "properties": {
            "subgoal_id": {"type": "string", "minLength": 1},
            "objective": {"type": "string", "minLength": 1},
            "required_output_type": {"type": "string", "minLength": 1},
            "depends_on": {"type": "array", "items": {"type": "string"}},
            "allowed_capabilities": {"type": "array", "items": capability},
            "allowed_side_effects": {
                "type": "array", "minItems": 1,
                "items": {"type": "string", "enum": [item.value for item in SideEffectClass]},
            },
            "limits": limits,
            "completion": completion,
            "failure_policy": {"type": "string", "enum": [item.value for item in FailurePolicy]},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "phase_id", "objective", "subgoals", "criterion_ids", "limits",
            "authorized_capabilities", "mutation_scope", "originating_decision",
        ],
        "properties": {
            "schema_version": {"const": V3_SCHEMA_VERSION},
            "phase_id": {"type": "string", "minLength": 1},
            "objective": {"type": "string", "minLength": 1},
            "subgoals": {"type": "array", "minItems": 1, "items": subgoal},
            "criterion_ids": {"type": "array", "items": {"type": "string"}},
            "limits": limits,
            "authorized_capabilities": {"type": "array", "items": capability},
            "mutation_scope": {
                "type": "object", "additionalProperties": False,
                "required": ["allowed_paths", "allow_external_effects"],
                "properties": {
                    "allowed_paths": {"type": "array", "items": {"type": "string"}},
                    "allow_external_effects": {"type": "boolean"},
                },
            },
            "originating_decision": {"type": "string"},
        },
    }
