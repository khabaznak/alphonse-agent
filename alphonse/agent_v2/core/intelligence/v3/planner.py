"""Strategic phase planning for hierarchical CAPD."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.intelligence.v3.contracts import FailurePolicy, PhasePlan, SideEffectClass
from alphonse.agent_v2.core.intelligence.v3.contracts import SUPPORTED_COMPLETION_KINDS, V3_SCHEMA_VERSION
from alphonse.agent_v2.core.intelligence.v3.revealing import tool_capabilities
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy

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
    contract_schema = _phase_plan_json_schema(catalog)
    prompt = (
        "Plan one bounded strategic execution phase. Return one JSON object matching the PhasePlan contract. "
        "Use meaningful subgoals, not one outer CAPD cycle per tool. "
        "When replying to the requester is itself part or all of the goal, include a user_response subgoal; "
        "a purely conversational request can be a single-stage response phase. "
        "Plan toward the requested outcome; do not declare it unavailable merely because Plan sees abstract "
        "capability identifiers rather than concrete tools. Jev evaluates the complete tool registry after this phase "
        "is produced. Do not create a user_response-only phase that claims inability unless prior verified evidence "
        "shows that relevant retrieval or action tools were attempted and failed. "
        "Authorize only capabilities and project-relative mutation paths needed in this phase. "
        "Never target .alphonse, memory ledgers, prompts, plans, acceptance criteria, or other agent-internal state. "
        "A mutation path must already be established by the user or prior verified evidence; otherwise plan a read-only discovery phase first. "
        "Use native.project_search and native.read_project_file for bounded file discovery; native.bash is unavailable in V3.\n\n"
        f"Goal: {task.goal}\n"
        f"Recent conversation (newest steering and answers are authoritative):\n{_bounded_text(task.recent_conversation_md, 6000)}\n"
        f"Known task facts:\n{_bounded_text(task.facts_md, 4000)}\n"
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
        return PhasePlan.from_dict(result.json_value)
    except (TypeError, ValueError) as exc:
        raise V3PhasePlanValidationError(f"v3_phase_plan_invalid:{exc}") from exc


def _bounded_text(value: object, max_chars: int) -> str:
    rendered = str(value or "").strip() or "- (none)"
    if len(rendered) <= max_chars:
        return rendered
    return rendered[-max(1, max_chars - 18):].lstrip() + "\n... truncated"


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
