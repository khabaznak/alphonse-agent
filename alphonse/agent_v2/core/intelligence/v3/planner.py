"""Strategic phase planning for hierarchical CAPD."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import TYPE_CHECKING

from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.intelligence.acceptance_contract import normalize_contract
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
    _ingest_relevant_messages(task, context)
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
    act_directive = task.metadata.get("act_directive")
    act_directive = act_directive if isinstance(act_directive, dict) else {}
    contract_schema = _phase_plan_json_schema(catalog)
    prompt = (
        "Plan one bounded strategic execution phase and define the mission acceptance criteria in the same response. "
        "Return one JSON object matching the PhasePlan contract. "
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
        f"Act directive (authoritative resilience instruction): {json.dumps(act_directive, ensure_ascii=False)}\n"
        "If Act requires a final response/closure, plan exactly one bounded user_response subgoal that uses the user-response capability, "
        "and do no further mission work in that phase. If Act requests user input, plan an ask-question subgoal. "
        f"Existing acceptance contract (preserve its definitions unless new user steering requires a justified revision): {json.dumps(task.ensure_acceptance_contract(), ensure_ascii=False)}\n"
        "If no acceptance contract exists yet, create a concise complete set of measurable acceptance_criteria; "
        "use stable IDs ac-1, ac-2, ... and make criterion_ids refer to the criteria needed in this phase. "
        "If a contract already exists, return acceptance_criteria as an empty array and preserve it. "
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
    if act_directive.get("response_required"):
        response_subgoals = [
            subgoal for subgoal in phase.subgoals
            if SideEffectClass.USER_RESPONSE in subgoal.allowed_side_effects
        ]
        if len(response_subgoals) != 1 or len(phase.subgoals) != 1:
            raise V3PhasePlanValidationError("v3_act_response_directive_not_isolated")
    if not task.acceptance_contract:
        if not phase.acceptance_criteria:
            raise V3PhasePlanValidationError("v3_phase_plan_acceptance_criteria_missing")
        markdown = "\n".join(f"- [ ] {item}" for item in phase.acceptance_criteria)
        task.acceptance_contract = normalize_contract({}, fallback_markdown=markdown, source_message_id=str(task.message_id or ""))
        if not task.acceptance_contract:
            raise V3PhasePlanValidationError("v3_phase_plan_acceptance_criteria_invalid")
        task.sync_acceptance_criteria_view()
        task.append_update("Plan established the mission acceptance contract alongside the strategic phase.")
    return _curate_phase_tools(task, phase, tools, context)


def _ingest_relevant_messages(task: "TaskState", context: "CoreLoopContext") -> None:
    """Jev classifies eligible queued context before strategic planning."""
    list_pending = getattr(context.messages, "list_pending", None)
    if not callable(list_pending):
        raise SystemOneUnavailableError("message_intake_unavailable")
    ignored = task.metadata.setdefault("v3_intake_ignored_message_ids", [])
    if not isinstance(ignored, list):
        ignored = []
        task.metadata["v3_intake_ignored_message_ids"] = ignored
    candidates = []
    for queued in list_pending(limit=1000):
        message = queued.message
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        if queued.message_id in ignored or metadata.get("source") in {"scheduled_task", "event_automation"}:
            continue
        disposition = str(metadata.get("routing_disposition") or "")
        eligible = (
            disposition == "steering" and message.user == task.user and message.project_id == task.project_id
        )
        question_id = str(metadata.get("answered_question_id") or "")
        if disposition == "correlated_response" and task.correlation_id and message.correlation_id == task.correlation_id and question_id:
            question = context.question_store.get_question(question_id) if context.question_store is not None else None
            eligible = bool(
                question is not None and question.status == "answered"
                and question.task_id == task.task_id
                and question.respondent_user_id == message.user
            )
        if eligible:
            candidates.append({
                "message_id": queued.message_id,
                "sender": message.user,
                "text": message.prompt,
                "question_id": question_id or None,
            })
    if not candidates:
        return
    triage = getattr(context.system_one, "triage_plan_messages", None) if context.system_one is not None else None
    if not callable(triage):
        raise SystemOneUnavailableError("message_intake_jev_unavailable")
    plan = task.hierarchical_state.get("phase") if isinstance(task.hierarchical_state, dict) else None
    task_view = {
        "task_id": task.task_id,
        "goal": task.goal,
        "acceptance_contract": task.ensure_acceptance_contract(),
        "strategic_plan": plan,
    }
    try:
        selected_ids = set(triage(task=task_view, candidates=candidates))
    except Exception as exc:
        raise SystemOneUnavailableError(f"message_intake_jev:{type(exc).__name__}") from exc
    eligible_ids = {item["message_id"] for item in candidates}
    ignored.extend(sorted(eligible_ids - selected_ids))
    for message_id in (item["message_id"] for item in candidates if item["message_id"] in selected_ids):
        queued = context.consume_message(MessageSelector(message_id=message_id))
        if queued is None:
            continue
        task.append_conversation_message(queued.message.user, queued.message.prompt)
        if queued.message.user == task.user:
            task.merge_attachments(queued.message.metadata)
    context.emit_telemetry({
        "event": "v3_plan_message_intake", "task_id": task.task_id,
        "candidate_count": len(candidates), "selected_count": len(selected_ids & eligible_ids),
    })


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
            "authorized_capabilities", "mutation_scope", "originating_decision", "acceptance_criteria",
        ],
        "properties": {
            "schema_version": {"const": V3_SCHEMA_VERSION},
            "phase_id": {"type": "string", "minLength": 1},
            "objective": {"type": "string", "minLength": 1},
            "subgoals": {"type": "array", "minItems": 1, "items": subgoal},
            "criterion_ids": {"type": "array", "items": {"type": "string"}},
            "acceptance_criteria": {"type": "array", "items": {"type": "string", "minLength": 1}},
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
