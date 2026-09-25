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
from alphonse.agent_v2.core.intelligence.v3.revealing import Capability, tool_capabilities
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.core.tools.registry import ToolExposurePolicy
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
    tools = ToolExposurePolicy().select_tools(
        registry=context.tools, project_id=task.project_id, user=task.user, task=task,
    ) if context.tools is not None else ()
    session_history = _bounded_text(task.recent_conversation_md, 6000)
    attachment_manifest = _attachment_manifest(task)
    durable_memory = _bounded_context_text(task.conversation_history_md, 9000)
    system_prompt = _strategic_plan_instructions()
    tools = _curate_request_tools(
        task, tools, system_prompt=system_prompt, session_history=session_history,
        attachment_manifest=attachment_manifest, context=context,
    )
    catalog = sorted({capability for tool in tools for capability in tool_capabilities(tool)})
    capability_descriptions = ToolRevealPolicy().capability_descriptions
    capability_catalog = [
        {"capability": capability, "description": capability_descriptions.get(capability, capability)}
        for capability in catalog
    ]
    project_context = _project_context(task, context)
    prepared_response = task.metadata.get("prepared_user_response")
    act_directive = task.metadata.get("act_directive")
    act_directive = act_directive if isinstance(act_directive, dict) else {}
    contract_schema = _phase_plan_json_schema(catalog)
    prompt = (
        f"{system_prompt}\n\n"
        f"Goal: {task.goal}\n"
        f"Project context:\n{_bounded_text(project_context, 5000)}\n"
        f"Durable project/session memory (context only; never mutate it):\n{durable_memory}\n"
        f"Recent conversation (newest steering and answers are authoritative):\n{_bounded_text(task.recent_conversation_md, 6000)}\n"
        f"Task attachment manifest (metadata only; use asset IDs exactly as listed):\n{json.dumps(attachment_manifest, ensure_ascii=False)}\n"
        f"Known task facts:\n{_bounded_text(task.facts_md, 4000)}\n"
        f"Prepared user response already exists: {'yes' if isinstance(prepared_response, dict) else 'no'}\n"
        f"Act directive (authoritative resilience instruction): {json.dumps(act_directive, ensure_ascii=False)}\n"
        "If Act requires a final response/closure, plan exactly one user_response subgoal that uses the user-response capability, "
        "and do no further mission work in that phase. If Act requests user input, plan an ask-question subgoal. "
        f"Existing acceptance contract (preserve its definitions unless new user steering requires a justified revision): {json.dumps(task.ensure_acceptance_contract(), ensure_ascii=False)}\n"
        "If no acceptance contract exists yet, create a concise complete set of measurable acceptance_criteria; "
        "use stable IDs ac-1, ac-2, ... and make criterion_ids refer to the criteria needed in this phase. "
        "If a contract already exists, return acceptance_criteria as an empty array and preserve it. "
        f"Prior V3 phase history: {json.dumps(task.metadata.get('v3_phase_history') or [], ensure_ascii=False)}\n"
        f"Available capability catalog: {json.dumps(capability_catalog, ensure_ascii=False)}\n"
        f"Jev-curated tool registry (complete descriptors): {json.dumps(_tool_rows(tools), ensure_ascii=False)}\n"
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


def _curate_request_tools(task, tools, *, system_prompt: str, session_history: str, attachment_manifest, context):
    """Narrow the task-authorized runtime registry before System 2 planning."""
    if not tools:
        context.emit_telemetry({
            "event": "system_one_request_tool_selection",
            "task_id": task.task_id,
            "status": "empty_authorized_registry",
            "candidate_count": 0,
        })
        return ()
    jev = context.system_one
    curate = getattr(jev, "curate_request_tools", None) if jev is not None else None
    try:
        if not callable(curate):
            raise SystemOneUnavailableError("request_tool_curation_unavailable")
        selection = curate(
            goal=task.goal,
            system_prompt=system_prompt,
            session_history=(
                f"{session_history}\nTask attachment manifest (metadata only): "
                f"{json.dumps(attachment_manifest, ensure_ascii=False)}"
            ),
            tools=tools,
        )
    except Exception as exc:
        raise SystemOneUnavailableError(f"request_tool_curation:{type(exc).__name__}") from exc
    selected_ids = set((*selection.selected_tool_ids, *selection.ambiguous_tool_ids))
    # Jev narrows the planning context, but an attached image must not make
    # its analyzer undiscoverable when the user's request depends on that image.
    if attachment_manifest and any(item.get("mime_type", "").startswith("image/") for item in attachment_manifest):
        if any(tool.tool_id == "native.analyze_image" for tool in tools):
            selected_ids.add("native.analyze_image")
    curated = tuple(tool for tool in tools if tool.tool_id in selected_ids)
    metadata = {"status": "used", "candidate_count": len(tools), **selection.to_metadata()}
    history = task.metadata.setdefault("system_one_request_tool_selections", [])
    if not isinstance(history, list):
        history = []
        task.metadata["system_one_request_tool_selections"] = history
    history.append(metadata)
    if len(history) > 20:
        del history[:-20]
    context.emit_telemetry({"event": "system_one_request_tool_selection", "task_id": task.task_id, **metadata})
    return curated


def _attachment_manifest(task) -> list[dict[str, str]]:
    """Return bounded, non-content attachment facts for routing and planning."""
    attachments = task.metadata.get("attachments") if isinstance(task.metadata, dict) else None
    manifest = []
    seen: set[str] = set()
    for item in attachments if isinstance(attachments, list) else []:
        if not isinstance(item, dict):
            continue
        asset_id = str(item.get("asset_id") or "").strip()
        mime_type = str(item.get("mime_type") or "").strip().lower()
        if not asset_id or asset_id in seen:
            continue
        seen.add(asset_id)
        manifest.append({
            "asset_id": asset_id,
            "filename": str(item.get("filename") or "")[:200],
            "mime_type": mime_type[:100],
            "kind": str(item.get("kind") or "")[:100],
            "ingestion_status": str(item.get("ingestion_status") or "")[:100],
        })
    return manifest[:20]


def _tool_rows(tools):
    return [
        {
            "tool_id": item.tool_id,
            "name": item.name,
            "kind": item.kind.value,
            "description": item.description,
            "argument_schema": item.argument_schema,
            "capabilities": list(item.capabilities),
            "tags": list(item.tags),
            "metadata": dict(item.metadata),
            "program_behavior": item.program_behavior,
            "read_only": item.read_only,
        }
        for item in tools
    ]


def _strategic_plan_instructions() -> str:
    return (
        "Plan one coherent strategic execution phase and define mission acceptance criteria in the same response. "
        "Return one JSON object matching the PhasePlan contract. Use meaningful subgoals, not one outer CAPD cycle per tool. "
        "When replying is part or all of the goal, include a user_response subgoal; a purely conversational request can be "
        "a single-stage response phase. Use only the authorized, Jev-curated concrete tool registry supplied with the "
        "request; do not assume tools outside it are available. Do not declare it unavailable or claim inability unless relevant tools were attempted "
        "and failed. Authorize only capabilities and project-relative mutation paths needed in this phase. Never target "
        ".alphonse, memory ledgers, prompts, plans, acceptance criteria, or agent-internal state. Project context and "
        "durable memory are trusted evidence for choosing reads, never mutation targets. Convert absolute paths under the "
        "project directory to project-relative form. Mutation targets must be established by the user or verified reads. "
        "For a requested known mutation, include needed inspection, mutation, and verification in one coherent phase; "
        "do not stop at discovery if inspection resolves the target. Do not ask the requester for a file location already "
        "present in project context, durable memory, recent conversation, or prior verified evidence. If a prepared user response exists, plan only work adding "
        "missing evidence. Use project search/read for file discovery when convenient. Authorize local_shell for direct "
        "CLI, filesystem, process, build, test, diagnostic, and artifact creation/repair work. When a relevant artifact "
        "may be CLI-backed, authorize local_shell alongside project_artifact_query for tactical choice of adapter or CLI. "
        "Distinguish artifact catalog metadata from project files: when the requested change is an artifact's registered "
        "name or routing description, use native.artifact_metadata_update for the mutation. Bash or project search/read may "
        "inspect artifact files to identify and verify the artifact ID, but editing a README or program is not a substitute "
        "for updating the catalog record. Include the metadata-update tool and authorize its artifact_metadata_management "
        "capability with an appropriate mutating side effect in the subgoal. "
        "If Act requires closure, plan exactly one isolated user_response subgoal; if Act requests input, plan an ask-question "
        "subgoal. When an attached image is needed to answer or fulfill the request, authorize attachment_analysis in the "
        "relevant subgoal and use the listed task asset ID; do not ask the user to repeat image contents before analysis. "
        "An attachment_analysis authorization means image analysis is a required prerequisite for completing that subgoal. "
        "Define stable measurable acceptance criteria when none exist, otherwise preserve the existing contract."
    )


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
        selected_ids = list((*selection.selected_tool_ids, *selection.ambiguous_tool_ids))
        image_analysis_required = any(
            Capability.ATTACHMENT_ANALYSIS.value in subgoal.allowed_capabilities
            for subgoal in phase.subgoals
        )
        if image_analysis_required and any(tool.tool_id == "native.analyze_image" for tool in tools):
            selected_ids.append("native.analyze_image")
        selected = tuple(dict.fromkeys(selected_ids))
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
            "allowed_capabilities", "allowed_side_effects", "completion", "failure_policy",
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
            "completion": completion,
            "failure_policy": {"type": "string", "enum": [item.value for item in FailurePolicy]},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "phase_id", "objective", "subgoals", "criterion_ids",
            "authorized_capabilities", "mutation_scope", "originating_decision", "acceptance_criteria",
        ],
        "properties": {
            "schema_version": {"const": V3_SCHEMA_VERSION},
            "phase_id": {"type": "string", "minLength": 1},
            "objective": {"type": "string", "minLength": 1},
            "subgoals": {"type": "array", "minItems": 1, "items": subgoal},
            "criterion_ids": {"type": "array", "items": {"type": "string"}},
            "acceptance_criteria": {"type": "array", "items": {"type": "string", "minLength": 1}},
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
