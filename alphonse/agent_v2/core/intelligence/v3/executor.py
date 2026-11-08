"""Bounded tactical execution inside one hierarchical CAPD Do phase."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from alphonse.agent_v2.core.context_inventory import source, prompt_receipt

import json
import time
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError

from alphonse.agent_v2.core.core import ImprovementPhase, ToolKind
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseOutcome, FailurePolicy
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseStatus
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseSubgoal
from alphonse.agent_v2.core.intelligence.v3.contracts import SideEffectClass
from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalAction
from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalState
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy, action_compatibility_error
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.core.intelligence.v3.projections import enabled as projections_enabled
from alphonse.agent_v2.core.intelligence.v3.context import selected_skill_guidance
from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision
from alphonse.agent_v2.core.intelligence.v3.revealing import Capability
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.core.tools.registry.native.bash import BASH_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native.respond import RESPOND_TOOL_ID
from alphonse.agent_v2.system_one import SystemOneUnavailableError, system_one_error_code
from alphonse.agent_v2.core.intelligence.v3.response_state import work_evidence_refs
from alphonse.agent_v2.core.intelligence.v3.task_understanding import receipt as task_understanding_receipt
from alphonse.agent_v2.core.tools.registry import ToolExposurePolicy
from alphonse.agent_v2.core.memory.workflow import MEMORY_WORKFLOW
from alphonse.agent_v2.core.memory.retry_context import retry_continuity_context
from .tactical_contract import TACTICAL_SCHEMA, valid_tactical_response
from .feasibility import FEASIBILITY_GUIDANCE
from ..routine_autonomy import ROUTINE_AUTONOMY_GUIDANCE, ordinary_workspace_context

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor
    from alphonse.agent_v2.core.intelligence.task_state import TaskState

ActionSelector = Callable[[TacticalState, PhaseSubgoal, tuple["ToolDescriptor", ...]], dict[str, Any] | None]
TACTICAL_RETRY_BASE_SECONDS = 1.0
MAX_UPDATE_INSPECTION_ACTIONS = 6
MAX_UNCERTAIN_PROGRESS_ACTIONS = 6
MAX_BROWSER_OBSERVATIONS_PER_SUBGOAL = 8
MAX_INDEPENDENT_READ_ACTIONS = 3


class ScheduledHandoff(Exception):
    """A durable checkpoint is ready for the daemon to yield its queue lease."""


class PhaseExecutor:
    """Run tactical actions until a phase reaches a real outcome."""

    def __init__(self, *, action_selector: ActionSelector | None = None, reveal_policy: ToolRevealPolicy | None = None) -> None:
        self._action_selector = action_selector
        self._reveal_policy = reveal_policy

    def run(self, task: "TaskState", state: TacticalState, context: "CoreLoopContext") -> PhaseOutcome:
        if state.status == PhaseStatus.PLANNED:
            state.transition(PhaseStatus.RUNNING)
            self._checkpoint(task, state, context)
            context.emit_activity(
                phase=ImprovementPhase.DO,
                label="phase started",
                message=state.phase.objective,
                progress={"phase_id": state.phase.phase_id, "subgoal_id": state.active_subgoal_id},
            )
        if task.metadata.get("source") == "scheduled_task" and state.status == PhaseStatus.RUNNING:
            recorded = {str(item.get("evidence_ref") or "") for item in state.evidence.entries}
            uncertain = next((action for action in state.actions
                              if action.tool_id in {"native.deliver_message", "native.send_attachment"}
                              and action.status in {"planned", "running"}
                              and f"tactical-action:{action.action_id}" not in recorded), None)
            if uncertain is not None:
                # A crash can occur after provider dispatch but before the
                # result checkpoint. Never send again on that ambiguity.
                task.metadata["v3_unconfirmed_delivery_action_id"] = uncertain.action_id
                state.transition(PhaseStatus.BLOCKED)
                self._checkpoint(task, state, context)
                return self._outcome(state, "The scheduled delivery outcome is unconfirmed.",
                                     blockers=("scheduled_delivery_outcome_unavailable",))
        # Ephemeral only: a restart discards proposals that have not run.
        pending_reads: list[dict[str, Any]] = []
        while state.status == PhaseStatus.RUNNING:
            interruption = self._interruption(task, state, context)
            if interruption is not None:
                self._checkpoint(task, state, context)
                return interruption
            stage_handoff = getattr(context.messages, "stage_scheduled_handoff", None)
            if (context.scheduled_handoff_enabled and callable(stage_handoff) and context.question_store is not None and context.schedule_store is not None
                    and stage_handoff(task, context.question_store, context.schedule_store)):
                raise ScheduledHandoff()
            subgoal = self._active_subgoal(state)
            tools = self._revealed_tools(task, state, subgoal, context)
            retry = task.metadata.pop("v3_tactical_delivery_retry", None)
            selected = (retry.get("action") if isinstance(retry, dict)
                        and retry.get("phase_id") == state.phase.phase_id
                        and retry.get("subgoal_id") == subgoal.subgoal_id else None)
            from_inference = False
            if isinstance(selected, dict):
                pending_reads.clear()
            elif pending_reads:
                selected = pending_reads.pop(0)
            else:
                selected = self._select_action(task, state, subgoal, tools, context)
                from_inference = self._action_selector is None
            interruption = self._interruption(task, state, context)
            if interruption is not None:
                self._checkpoint(task, state, context)
                return interruption
            if selected is None:
                state.transition(PhaseStatus.BLOCKED)
                self._checkpoint(task, state, context)
                return self._outcome(state, "No valid tactical action was available.", blockers=("tactical_action_unavailable",))
            # An outstanding fill has only one safe local recovery path: obtain
            # a subsequent observation, then reconcile it. Selection remains
            # subject to the ordinary reveal, phase, grant and cancellation checks.
            required = _pending_browser_fill_recovery(task)
            reacquire = False
            if required and selected.get("tool_id") == "native.browser_workflow_reacquire":
                try:
                    service = context.browser_workflow_service
                    reacquire = service is not None and not service.projection(task).get("session_available")
                except Exception:
                    reacquire = False
            if required and selected.get("tool_id") != required and not reacquire:
                state.transition(PhaseStatus.BLOCKED)
                self._checkpoint(task, state, context)
                return self._outcome(state, "The browser effect needs independent reconciliation.",
                                     blockers=("browser_workflow_reconciliation_required",))
            is_sequence = "actions" in selected
            is_browser_sequence = "browser_fill_sequence" in selected
            selected_id = selected.get("tool_id")
            sensitive_action = is_sequence or is_browser_sequence or (isinstance(selected_id, str) and selected_id.startswith("native.browser_workflow_")) or any(
                tool.tool_id == selected_id and tool.metadata.get("sensitive_content") is True for tool in tools)
            try:
                if from_inference and not valid_tactical_response(selected):
                    raise ValueError("tactical_response_schema_invalid")
                if is_browser_sequence:
                    proposed = _validate_browser_fill_sequence(self, state, subgoal, selected, tools)
                    selected, pending_reads = proposed[0], proposed[1:]
                elif is_sequence:
                    proposed = _validate_independent_reads(self, state, subgoal, selected, tools)
                    selected, pending_reads = proposed[0], proposed[1:]
                    sensitive_action = False
                action = self._validate_action(state, subgoal, selected, tools)
            except (ValueError, PermissionError) as exc:
                pending_reads.clear()
                error = str(exc)
                context.emit_telemetry({"event": "v3_tactical_response_failure", "category":
                    "schema" if error == "tactical_response_schema_invalid" else "action_validation"})
                # All rejected provider output can contain sensitive content,
                # including under unknown keys or a forged tool identifier.
                safe_error = error.split(":", 1)[0]
                if not safe_error.startswith(("tactical_", "browser_fill_sequence_", "independent_reads_", "legacy_")):
                    safe_error = "tactical_action_invalid"
                from . import browser_workflow
                sensitive_rejection = (sensitive_action or browser_workflow.KEY in task.metadata
                                       or browser_workflow._legacy_browser_state(task))
                fingerprint = hashlib.sha256(json.dumps({
                    "error": "sensitive_action_invalid" if sensitive_rejection else safe_error,
                    "capabilities": sorted(subgoal.allowed_capabilities),
                    "effects": sorted(item.value for item in subgoal.allowed_side_effects),
                }, sort_keys=True).encode()).hexdigest()
                seen = task.metadata.setdefault("v3_rejected_action_fingerprints", [])
                repeated = fingerprint in seen
                exhausted = any(item.get("status") == "rejected" and
                                item.get("subgoal_id") == subgoal.subgoal_id for item in state.evidence.entries)
                if not repeated:
                    seen.append(fingerprint)
                state.evidence.append({
                    "evidence_ref": f"tactical-rejection:{uuid4()}",
                    "phase_id": state.phase.phase_id, "subgoal_id": subgoal.subgoal_id,
                    "status": "rejected", "error": "sensitive_action_invalid" if sensitive_rejection else safe_error,
                    "rejected_action": {"tool_id": "native.browser_workflow" if sensitive_rejection else "redacted", "arguments": {"redacted": True}},
                    "validation_feedback": "No tool ran. Correct this action once using the same revealed tools, "
                        "capabilities, effects and mutation scope; preserve completed work. Do not broaden authorization.",
                    "fingerprint": fingerprint, "repeated": repeated,
                })
                self._checkpoint(task, state, context)
                if not repeated and not exhausted:
                    continue
                state.transition(PhaseStatus.BLOCKED)
                self._checkpoint(task, state, context)
                reason = ("tactical_planning_error_repeated:" if repeated else "tactical_correction_exhausted:") + ("sensitive_action_invalid" if sensitive_rejection else safe_error)
                return self._outcome(state, reason, blockers=(reason,))
            if sensitive_action:
                action = TacticalAction(action_id=f"action-{uuid4()}", subgoal_id=action.subgoal_id,
                    tool_id=action.tool_id, arguments=action.arguments, acceptance_questions=action.acceptance_questions)
            if (action.tool_id in {"native.browser_workflow_observe", "native.browser_workflow_observe_result"}
                    and sum(prior.subgoal_id == action.subgoal_id and prior.tool_id in {
                        "native.browser_workflow_observe", "native.browser_workflow_observe_result"}
                            for prior in state.actions) >= MAX_BROWSER_OBSERVATIONS_PER_SUBGOAL):
                state.transition(PhaseStatus.BLOCKED)
                self._checkpoint(task, state, context)
                return self._outcome(state, "Browser observation budget exhausted; review the existing evidence.",
                                     blockers=("browser_observation_budget_exhausted",))
            recorded_arguments = {"redacted": True} if sensitive_action else action.arguments
            recorded_questions = [] if sensitive_action else action.acceptance_questions
            if _scheduled_delivery_may_repeat(task, state, action, context):
                state.transition(PhaseStatus.BLOCKED)
                self._checkpoint(task, state, context)
                return self._outcome(state, "A similar scheduled message may already have been sent.",
                                     blockers=("scheduled_delivery_repetition_unresolved",))
            recovery = (task.metadata.get("act_directive") or {}).get("outcome_review") or {}
            verified_refs = set(recovery.get("verified_evidence_refs") or ())
            repeated_verified = any(
                entry.get("evidence_ref") in verified_refs and entry.get("tool_id") == action.tool_id
                and (entry.get("arguments") or {}) == recorded_arguments
                for entry in state.evidence.entries
            )
            redacted_observation = action.tool_id in {"native.browser_workflow_observe", "native.browser_workflow_observe_result"}
            if repeated_verified and not redacted_observation and not _browser_fill_has_protected_identity(task, action, context):
                blocker = "act_recovery_repeats_verified_work"
                state.transition(PhaseStatus.BLOCKED)
                self._checkpoint(task, state, context)
                return self._outcome(state, blocker, blockers=(blocker,))
            retry_count = 0
            retry_interrupted = False
            reconciliation_step = False
            while True:
                if retry_count:
                    interruption = self._interruption(task, state, context)
                    if interruption is not None:
                        self._checkpoint(task, state, context)
                        return interruption
                state.actions.append(TacticalAction(action_id=action.action_id, subgoal_id=action.subgoal_id,
                    tool_id=action.tool_id, arguments=recorded_arguments, acceptance_questions=recorded_questions))
                self._checkpoint(task, state, context)
                context.emit_activity(
                    phase=ImprovementPhase.DO,
                    label="tactical action",
                    message=subgoal.objective,
                    progress={
                        "phase_id": state.phase.phase_id,
                        "subgoal_id": subgoal.subgoal_id,
                        "action_id": action.action_id,
                        "tool_id": action.tool_id,
                        "retry_attempt": retry_count,
                    },
                )
                # Inference and retry waits can outlive a package update or ACL
                # change. Revalidate pins before native dispatch, not just when
                # composing the prompt. Existing effect/phase checks still apply.
                selected_skill_guidance(task, context)
                from .event_evidence import project_event_evidence
                project_event_evidence(task, context)
                interruption = self._interruption(task, state, context)
                if interruption is not None:
                    state.actions.pop()  # This action never reached native dispatch.
                    self._checkpoint(task, state, context)
                    return interruption
                understanding = task_understanding_receipt(task)
                outcome = ToolInvocationService(context=context, task=task).invoke(
                    action.tool_id, action.arguments, call_id=action.action_id
                )
                completed = TacticalAction(
                    action_id=action.action_id,
                    subgoal_id=action.subgoal_id,
                    tool_id=action.tool_id,
                    arguments=recorded_arguments,
                    status=str(outcome.get("status") or "failed"),
                    result={"redacted": True} if sensitive_action else outcome.get("result"),
                    error=_error_message(outcome.get("error")),
                    acceptance_questions=recorded_questions,
                )
                state.actions[-1] = completed
                state.evidence.append(
                    {
                        "evidence_ref": f"tactical-action:{completed.action_id}",
                        "invocation": {
                            "call_id": outcome["call_id"],
                            "read_only": next((tool.read_only for tool in tools if tool.tool_id == completed.tool_id), False),
                            "task_understanding": understanding,
                        },
                        "error_code": (outcome.get("error") or {}).get("code"),
                        "phase_id": state.phase.phase_id,
                        "subgoal_id": subgoal.subgoal_id,
                        "tool_id": completed.tool_id,
                        "arguments": completed.arguments,
                        "status": completed.status,
                        "result": completed.result,
                        "error": completed.error,
                        "retry_attempt": retry_count,
                        "acceptance_questions": [dict(item) for item in completed.acceptance_questions],
                    }
                )
                if task.metadata.get("source") == "scheduled_task" and completed.status == "success":
                    from .delivery_evidence import delivery_receipt
                    receipt = delivery_receipt(task, state.evidence.entries[-1], context)
                    if isinstance(receipt, dict) and receipt.get("status") in {"pending", "claimed", "retry_wait", "unconfirmed", "failed"}:
                        result = completed.result if isinstance(completed.result, dict) else {}
                        task.metadata["v3_tactical_delivery_retry"] = {
                            "phase_id": state.phase.phase_id, "subgoal_id": subgoal.subgoal_id,
                            "action": {"tool_id": completed.tool_id, "arguments": completed.arguments,
                                       "acceptance_questions": list(completed.acceptance_questions)},
                        }
                        self._checkpoint(task, state, context)
                        from .processor import _await_scheduled_delivery
                        if not _await_scheduled_delivery(task, context, {
                            "outbox_message_id": result.get("outbox_message_id"),
                            "recipient_user_id": result.get("recipient_user_id"),
                        }):
                            state.transition(PhaseStatus.BLOCKED)
                            self._checkpoint(task, state, context)
                            return self._outcome(state, "Scheduled delivery could not be confirmed.",
                                                 blockers=("scheduled_delivery_outcome_unavailable",))
                        task.metadata.pop("v3_tactical_delivery_retry", None)
                        # The provider completed during staging; preserve the
                        # original result while attaching the verified receipt.
                        refreshed = delivery_receipt(task, state.evidence.entries[-1], context)
                        if refreshed is not None:
                            state.evidence.entries[-1]["result"] = {**result, "delivery_receipt": refreshed}
                interruption = self._interruption(task, state, context)
                if interruption is not None:
                    self._checkpoint(task, state, context)
                    return interruption
                if completed.status == "waiting":
                    state.transition(PhaseStatus.WAITING_USER)
                    self._checkpoint(task, state, context)
                    return self._outcome(state, "A tactical action requires user input.")
                if completed.tool_id == "native.browser_workflow_fill" and completed.status == "success":
                    # A return is only an attempt. Continue solely to the fenced
                    # observation/reconciliation path, never to completion.
                    if _pending_browser_fill_recovery(task) == "native.browser_workflow_observe":
                        self._checkpoint(task, state, context)
                        reconciliation_step = True
                        break
                if completed.tool_id in {"native.browser_workflow_fill", "native.browser_workflow_submit"}:
                    # A transport return cannot verify the form effect. Until a
                    # fresh-observation reconciliation exists, do not retry or
                    # let semantic review promote this into phase completion.
                    state.transition(PhaseStatus.BLOCKED)
                    self._checkpoint(task, state, context)
                    return self._outcome(state, "The browser effect needs independent reconciliation.",
                                         blockers=("browser_workflow_reconciliation_required",))
                if completed.tool_id == "native.browser_workflow_observe" and required == completed.tool_id:
                    if completed.status == "success":
                        self._checkpoint(task, state, context)
                        reconciliation_step = True
                        break
                    state.transition(PhaseStatus.BLOCKED)
                    self._checkpoint(task, state, context)
                    return self._outcome(state, "Browser reconciliation remains unavailable.",
                                         blockers=("browser_workflow_reconciliation_required",))
                if completed.tool_id == "native.browser_workflow_reconcile" and required == completed.tool_id and completed.status == "success":
                    if not _pending_browser_fill_recovery(task):
                        self._checkpoint(task, state, context)
                        reconciliation_step = True
                        break
                if completed.tool_id in {"native.browser_workflow_reconcile", "native.browser_workflow_verify_submission"}:
                    state.transition(PhaseStatus.BLOCKED)
                    self._checkpoint(task, state, context)
                    reason = ("Review the recorded browser reconciliation before continuing." if completed.status == "success"
                              else "Browser reconciliation remains unavailable.")
                    return self._outcome(state, reason,
                                         blockers=("browser_workflow_reconciliation_review",))
                self._checkpoint(task, state, context)
                completion_met, retry_approved, progress_review = self._completion_review(
                    task, state, subgoal, completed, context
                )
                state.evidence.entries[-1]["progress_review"] = {
                    "complete": completion_met,
                    "confident": bool(progress_review.get("confident")),
                    "confidence": progress_review.get("confidence"),
                    "confidence_kind": progress_review.get("confidence_kind"),
                    "retry_approved": retry_approved,
                    "context_complete": progress_review.get("context_complete", True),
                    "context_coverage": progress_review.get("context_coverage", {}),
                    "answers": dict(progress_review.get("answers") or {}),
                }
                if not completion_met and completed.status == "success":
                    _queue_uncertainty_recovery(state, subgoal, context)
                if not completion_met and _is_update_inspection(state, subgoal, progress_review.get("answers") or {},
                    float(getattr(getattr(context.system_one, "settings", None), "yes_threshold", 0.8))):
                    count = int(task.metadata.get("v3_update_inspection_actions") or 0) + 1
                    task.metadata["v3_update_inspection_actions"] = count
                    threshold = float(getattr(getattr(context.system_one, "settings", None), "yes_threshold", 0.8))
                    answers = progress_review.get("answers") or {}
                    ready = answers.get("runtime_inspection_ready_for_action", 0.0)
                    missing = answers.get("runtime_inspection_requires_user_input", 0.0)
                    route = None
                    if completed.status == "success" and progress_review.get("context_complete", True):
                        if _confident_probability(ready, threshold) and _confident_negative(missing, threshold):
                            route = "ready_for_action"
                        elif _confident_probability(missing, threshold) and _confident_negative(ready, threshold):
                            route = "requires_user_input"
                    if route is None and not completion_met and count >= MAX_UPDATE_INSPECTION_ACTIONS:
                        route = "decision_required"
                    if route is not None:
                        task.metadata["v3_inspection_resolution"] = {
                            "route": route, "phase_id": state.phase.phase_id,
                            "evidence_refs": [entry["evidence_ref"] for entry in state.evidence.entries
                                              if entry.get("status") == "success" and entry.get("evidence_ref")],
                        }
                        blocker = "inspection_" + route
                        state.evidence.entries[-1]["inspection_resolution"] = route
                        state.transition(PhaseStatus.BLOCKED)
                        self._checkpoint(task, state, context)
                        return self._outcome(state, "Inspection must now resolve into the authorized operation or a specific missing-information question; do not restart inspection.", blockers=(blocker,))
                settings = getattr(context.system_one, "settings", None)
                yes_threshold = float(getattr(settings, "yes_threshold", 0.8))
                no_threshold = float(getattr(settings, "no_threshold", 0.2))
                if pending_reads and (completion_met or completed.status != "success"
                                      or not _independent_read_can_continue(progress_review, yes_threshold)):
                    pending_reads.clear()
                if not completion_met and _consecutive_no_progress(state, subgoal.subgoal_id, no_threshold) >= 3:
                    blocker = "tactical_stage_no_new_progress"
                    state.evidence.entries[-1]["no_progress_blocker"] = blocker
                    state.transition(PhaseStatus.BLOCKED)
                    self._checkpoint(task, state, context)
                    return self._outcome(state, "Repeated attempts added no useful evidence; a different strategy is required.", blockers=(blocker,))
                if not completion_met and _consecutive_uncertain_progress(
                    state, subgoal.subgoal_id, no_threshold, yes_threshold
                ) >= MAX_UNCERTAIN_PROGRESS_ACTIONS:
                    blocker = "tactical_progress_uncertain"
                    state.evidence.entries[-1]["no_progress_blocker"] = blocker
                    state.transition(PhaseStatus.BLOCKED)
                    self._checkpoint(task, state, context)
                    return self._outcome(state, "Progress remained uncertain after repeated attempts; review the evidence and choose a different strategy.", blockers=(blocker,))
                repeated_evidence = (
                    _matching_prior_incomplete_result(state, completed)
                    if not completion_met and (completed.status == "success" or (completed.status == "failed" and not retry_approved))
                    else None
                )
                if repeated_evidence is not None:
                    blocker = f"tactical_action_repeated_without_progress:{repeated_evidence}"
                    state.evidence.entries[-1]["semantic_duplicate_of"] = repeated_evidence
                    state.evidence.entries[-1]["no_progress_blocker"] = blocker
                    state.transition(PhaseStatus.BLOCKED)
                    self._checkpoint(task, state, context)
                    return self._outcome(state, blocker, blockers=(blocker,))
                self._checkpoint(task, state, context)
                retry_available = (
                    completed.status == "failed"
                    and retry_approved
                    and _retry_is_safe(completed, context)
                )
                if not retry_available:
                    break
                delay = TACTICAL_RETRY_BASE_SECONDS * (2 ** retry_count)
                if not _wait_for_retry(delay, task, state, context):
                    retry_interrupted = True
                    break
                retry_count += 1
                action = TacticalAction(
                    action_id=f"action-{uuid4()}",
                    subgoal_id=action.subgoal_id,
                    tool_id=action.tool_id,
                    arguments=dict(action.arguments),
                    acceptance_questions=action.acceptance_questions,
                )
            if reconciliation_step:
                continue
            if retry_interrupted:
                interruption = self._interruption(task, state, context)
                if interruption is not None:
                    self._checkpoint(task, state, context)
                    return interruption
            if completed.status != "success":
                failure = self._handle_failure(state, subgoal, completed)
                self._checkpoint(task, state, context)
                if failure is not None:
                    return failure
                continue
            if not completion_met:
                continue
            if completed.tool_id == RESPOND_TOOL_ID and isinstance(completed.result, dict):
                message = str(completed.result.get("message") or "").strip()
                if message:
                    task.metadata["prepared_user_response"] = {
                        "source": RESPOND_TOOL_ID,
                        "tool_call_id": completed.action_id,
                        "message": message,
                        "work_evidence_refs": work_evidence_refs(state.evidence.entries),
                    }
                    task.metadata.pop("v3_response_refresh_required", None)
            state.bind_subgoal_output(subgoal.subgoal_id, subgoal.required_output_type, completed.result)
            state.completed_subgoal_ids.append(subgoal.subgoal_id)
            state.transition(PhaseStatus.SUBGOAL_COMPLETE)
            context.emit_activity(
                phase=ImprovementPhase.DO,
                label="subgoal completed",
                message=subgoal.objective,
                progress={"phase_id": state.phase.phase_id, "subgoal_id": subgoal.subgoal_id},
            )
            next_subgoal = _next_subgoal(state)
            if next_subgoal is None:
                state.transition(PhaseStatus.PHASE_COMPLETE)
                self._checkpoint(task, state, context)
                return self._outcome(state, "All phase subgoals completed with recorded evidence.")
            state.active_subgoal_id = next_subgoal.subgoal_id
            state.revealed_capabilities = []
            state.revealed_tool_ids = []
            state.transition(PhaseStatus.RUNNING)
            self._checkpoint(task, state, context)
        return self._outcome(state, "The phase ended outside the running state.")

    def _select_action(
        self,
        task: "TaskState",
        state: TacticalState,
        subgoal: PhaseSubgoal,
        tools: tuple["ToolDescriptor", ...],
        context: "CoreLoopContext",
    ) -> dict[str, Any] | None:
        skill_guidance, loaded_skill_ids = selected_skill_guidance(task, context)
        if self._action_selector is not None:
            selected = self._action_selector(state, subgoal, tools)
            if isinstance(selected, dict) and not selected.get("acceptance_questions") and "browser_fill_sequence" not in selected:
                selected = dict(selected)
                selected["acceptance_questions"] = [{
                    "question_id": "stage_result_satisfies_completion",
                    "type": "noul",
                    "instructions": "Does the tool result satisfy the current stage completion condition?",
                    "criteria": {
                        "true": "The observed result establishes the stage completion condition.",
                        "false": "The result is missing, failed, ambiguous, or does not establish the condition.",
                    },
                }]
            return selected
        if context.inference is None:
            return None
        skill_guidance, loaded_skill_ids = ("", []) if projections_enabled() else selected_skill_guidance(task, context)
        skill_selection = task.metadata.get("v3_skill_selection")
        if isinstance(skill_selection, dict) and not projections_enabled():
            skill_selection["loaded_skill_ids"] = loaded_skill_ids
        if not projections_enabled():
            from .task_understanding import render
            skill_guidance = skill_guidance + retry_continuity_context(task, context) + render(task)
        directive = task.metadata.get("act_directive") or {}
        recovery = directive.get("outcome_review") or {}
        if recovery.get("remaining_work"):
            skill_guidance = skill_guidance + source(
                "\nOutcome recovery: " + json.dumps(recovery, ensure_ascii=False)
                + "\nAddress the specific remaining work within the authorized phase. Preserve the verified work. "
                "For a question, ask for the specific missing fact and choose the answer form that fits it.",
                'task:' + str(task.task_id) + ':outcome_recovery', 'outcome_recovery',
            )
        resolution = directive.get("inspection_resolution")
        if resolution:
            skill_guidance = skill_guidance + source(
                "\nInspection handoff: " + json.dumps(resolution, ensure_ascii=False)
                + "\nUse the existing evidence to carry out the authorized operation or ask the specific missing-information "
                "question in this phase. Do not repeat broad database inspection or independently re-extract an already "
                "reviewed and confirmed source. Missing or conflicting essential fields still require a question; never guess them.",
                'task:' + str(task.task_id) + ':inspection_handoff', 'inspection_handoff',
            )
        skill_guidance = skill_guidance + "\n" + MEMORY_WORKFLOW
        required_browser_step = _pending_browser_fill_recovery(task)
        if required_browser_step in {"native.browser_workflow_observe", "native.browser_workflow_reconcile"}:
            skill_guidance += source(
                "\nA previous browser fill has an unresolved effect. The next permitted local recovery step is "
                + required_browser_step + " with empty arguments. Do not repeat the fill or perform another "
                "mutation until the independent reconciliation succeeds. Normal authorization and cancellation checks apply.",
                'task:' + str(task.task_id) + ':browser_reconciliation', 'browser_reconciliation',
            )
        selection = task.metadata.get("v3_context_selection") or {}
        task_context = task_context_for_decision(task, context, purpose="execution",
                                                selected_context_ids=selection.get("selected_context_ids", []))
        recovery = _consume_uncertainty_recovery(state, subgoal)
        if recovery is not None:
            # Reserve this bounded recovery opportunity before inference so a
            # restart cannot repeatedly spend it. The next ordinary action-composition
            # call carries the question and evidence; it cannot declare success.
            self._checkpoint(task, state, context)
            context.emit_telemetry({"event": "v3_tactical_uncertainty_recovery", "task_id": task.task_id,
                                    "phase_id": state.phase.phase_id, "subgoal_id": subgoal.subgoal_id,
                                    "evidence_ref": recovery["evidence_ref"], "reason": recovery["reason"]})
            skill_guidance += source(
                "\nTargeted uncertainty recovery: " + json.dumps(recovery, ensure_ascii=False)
                + "\nUse the referenced execution evidence and coverage gaps to select the next safe action. "
                  "Alongside the action, include uncertainty_assessment with question_id, answer "
                  "(supported, not_supported, or uncertain), and evidence_refs referencing only successful "
                  "execution-log entries. Use uncertain if the evidence does not resolve the question. "
                  "The assessment is advisory and cannot establish completion; obtain a targeted observation when needed. "
                  "Do not repeat a mutation with an uncertain effect. All normal authorization and verification apply.",
                'phase:' + state.phase.phase_id + ':uncertainty_recovery', 'uncertainty_recovery',
            )
        prompt = _tactical_prompt(
            state,
            subgoal,
            tools,
            goal="" if projections_enabled() else task.goal,
            task_project_root=_task_project_root(task, context),
            attachment_manifest=_task_attachment_manifest(task),
            skill_guidance=skill_guidance,
            task_context=task_context + "\n" + ordinary_workspace_context(task, context),
        )
        # Providers which cannot enforce an output schema still receive the
        # identical contract in the prompt; local validation is authoritative.
        prompt += "\nTactical response JSON Schema: " + json.dumps(TACTICAL_SCHEMA, sort_keys=True)
        from .context_manifests import event_input_metadata
        images = context.image_inputs(task)
        request = InferenceRequest(
            prompt=prompt,
            purpose=InferencePurpose.TACTICAL_ACTION,
            context_manifests=prompt_receipt(prompt, [task_context, skill_guidance,
                source('Strategic plan: ' + json.dumps(state.phase.to_dict(), ensure_ascii=False), 'phase:' + state.phase.phase_id, 'strategic_plan'),
                source('Complete tool execution log (every entry, no omissions or truncation):\n' + state.prompt_projection(max_evidence_entries=None, max_chars=None), 'phase:' + state.phase.phase_id + ':execution', 'execution_evidence'),
                *([source('Goal: ' + task.goal, 'request:' + str(task.message_id or task.task_id), 'current_request')] if not projections_enabled() else []),
            ], purpose='execution'),
            project_id=task.project_id,
            user=task.user,
            task_id=task.task_id,
            images=images,
            tools=tools,
            metadata={"phase_id": state.phase.phase_id, "subgoal_id": subgoal.subgoal_id,
                      **event_input_metadata(task, "execution")},
            cancel_checker=context.is_cancelled if context.cancellation_checker is not None else None,
        )
        try:
            result = _safe_tactical_generate(context.inference, request)
        except ValueError as exc:
            # A malformed provider response contains no executable action. A
            # single fresh attempt can recover without replaying that response,
            # any browser mutation, or sensitive output into the next prompt.
            category = {"openai_codex_invalid_json": "encoding",
                        "openai_codex_invalid_tactical_schema": "schema"}.get(str(exc), "provider")
            context.emit_telemetry({"event": "v3_tactical_response_failure", "category": category})
            if "v3_browser_workflow" not in task.metadata or str(exc) not in {"openai_codex_invalid_json", "openai_codex_invalid_tactical_schema"}:
                raise
            if context.is_cancelled():
                return None
            try:
                result = _safe_tactical_generate(context.inference, replace(request, prompt=request.prompt +
                    "\nThe previous response could not be parsed. Return one valid JSON object using the action contract."))
            except ValueError as retry_exc:
                if str(retry_exc) not in {"openai_codex_invalid_json", "openai_codex_invalid_tactical_schema"}:
                    raise
                return None
        if recovery is not None and valid_tactical_response(result.json_value):
            _record_uncertainty_assessment(state, recovery, result.json_value)
        if not isinstance(result.json_value, dict):
            context.emit_telemetry({"event": "v3_tactical_response_failure", "category": "schema"})
            return None
        if images:
            state.evidence.append({
                "evidence_ref": f"inference-images:{uuid4()}",
                "phase_id": state.phase.phase_id,
                "subgoal_id": subgoal.subgoal_id,
                "kind": "inference_image_input",
                "status": "provided",
                "asset_ids": [image.asset_id for image in images],
                "model": result.model_profile.model if result.model_profile else "",
            })
        return dict(result.json_value)

    def _revealed_tools(
        self, task: "TaskState", state: TacticalState, subgoal: PhaseSubgoal, context: "CoreLoopContext"
    ) -> tuple["ToolDescriptor", ...]:
        if context.tools is None:
            return ()
        # Re-evaluate current runtime exposure even for legacy Jev-curated
        # checkpoints. Historical relevance omissions must not hide capabilities.
        candidates = ToolExposurePolicy().select_tools(
            registry=context.tools, project_id=task.project_id, user=task.user, task=task,
        )
        if self._reveal_policy is not None:
            reveal = self._reveal_policy.reveal(task, state, subgoal, candidates)
            state.revealed_capabilities = [item["capability"] for item in reveal.catalog]
            state.revealed_tool_ids = [item.tool_id for item in reveal.tools]
            context.emit_ui_event(
                "tactical_tools_revealed",
                {
                    "phase_id": state.phase.phase_id,
                    "subgoal_id": subgoal.subgoal_id,
                    "capabilities": list(state.revealed_capabilities),
                    "tool_ids": list(state.revealed_tool_ids),
                    "decisions": [item.__dict__ for item in reveal.decisions],
                },
            )
            return reveal.tools
        allowed_ids = set(state.revealed_tool_ids)
        if not allowed_ids:
            # Stage 2 compatibility: use a fixed shortlist chosen by the caller/phase.
            # Stage 3 replaces this with progressive reveal policy.
            allowed_ids = {item.tool_id for item in candidates if item.tool_id in state.phase.authorized_capabilities}
            state.revealed_tool_ids = sorted(allowed_ids)
        return tuple(item for item in candidates if item.tool_id in allowed_ids)

    def _completion_review(
        self,
        task: "TaskState",
        state: TacticalState,
        subgoal: PhaseSubgoal,
        action: TacticalAction,
        context: "CoreLoopContext",
    ) -> tuple[bool, bool, dict[str, Any]]:
        deterministic = _condition_met(subgoal, action.result)
        if context.system_one is None or not hasattr(context.system_one, "evaluate_tactical_progress"):
            raise SystemOneUnavailableError("tactical_acceptance_unavailable")
        from .task_understanding import render
        try:
            review = context.system_one.evaluate_tactical_progress(
                goal=render(task),
                phase=state.phase.to_dict(),
                subgoal={
                    "subgoal_id": subgoal.subgoal_id,
                    "objective": subgoal.objective,
                    "required_output_type": subgoal.required_output_type,
                    "depends_on": list(subgoal.depends_on),
                    "allowed_capabilities": list(subgoal.allowed_capabilities),
                    "completion": subgoal.completion.__dict__,
                    "allowed_side_effects": [item.value for item in subgoal.allowed_side_effects],
                    "failure_policy": subgoal.failure_policy.value,
                },
                action=action.to_dict(),
                questions=list(action.acceptance_questions),
                execution_log=[dict(item) for item in state.evidence.entries],
                task_context=task_context_for_decision(
                    task, context, purpose="progress_review",
                    selected_context_ids=(
                        task.metadata.get("v3_context_selection", {}).get("selected_context_ids", [])
                        if isinstance(task.metadata.get("v3_context_selection"), dict) else []
                    ),
                ),
            )
        except Exception as exc:
            raise SystemOneUnavailableError(f"tactical_acceptance:{type(exc).__name__}:{system_one_error_code(exc)}") from exc
        else:
            metadata = {"status": "used" if review.confident else "ambiguous_fallback", **review.to_metadata()}
            result = review.complete if review.confident else deterministic
            stage_answer = review.answers.get("runtime_declared_subgoal_complete")
            response_only = (action.tool_id == RESPOND_TOOL_ID
                             and set(subgoal.allowed_side_effects) == {SideEffectClass.USER_RESPONSE})
            if isinstance(stage_answer, (int, float)) and not response_only:
                threshold = float(getattr(getattr(context.system_one, "settings", None), "yes_threshold", 0.8))
                # A generic output-present predicate cannot turn an ambiguous
                # judgment of an operational stage into completion. Preparing a
                # response retains ambiguous fallback; outer acceptance review
                # still verifies whether its content fulfills the request.
                result = result and stage_answer >= threshold
            if not getattr(review, 'context_complete', True):
                result = False
                metadata['completion_blocker'] = 'targeted_observation_required'
            retry_approved = bool(getattr(review, "retry_approved", False)) and getattr(review, 'context_complete', True)
            metadata["retry_approved"] = retry_approved
        if (
            Capability.ATTACHMENT_ANALYSIS.value in subgoal.allowed_capabilities
            and _has_task_image(task)
            and not _image_inputs_provided(state, subgoal.subgoal_id)
        ):
            # Successful inference transport is required; a filename or a
            # shell lookup alone is not a model image input. Jev still reviews
            # whether the resulting action satisfies the requested outcome.
            result = False
            metadata["completion_blocker"] = "required_attachment_analysis_not_completed"
        if action.status != "success":
            result = False
            metadata["completion_blocker"] = "tool_execution_failed"
        _record_system_one_tactical_review(task, state, subgoal, action, metadata)
        context.emit_telemetry({
            "event": "system_one_tactical_review", "task_id": task.task_id,
            "phase_id": state.phase.phase_id, "subgoal_id": subgoal.subgoal_id,
            "action_id": action.action_id, **metadata,
        })
        return result, retry_approved, metadata

    @staticmethod
    def _validate_action(
        state: TacticalState,
        subgoal: PhaseSubgoal,
        selected: dict[str, Any],
        tools: tuple["ToolDescriptor", ...],
    ) -> TacticalAction:
        tool_id = str(selected.get("tool_id") or "").strip()
        arguments = selected.get("arguments")
        descriptor = next((item for item in tools if item.tool_id == tool_id), None)
        if descriptor is None:
            raise ValueError(f"tactical_tool_not_revealed:{tool_id or '(missing)'}")
        if not isinstance(arguments, dict):
            raise ValueError("tactical_tool_arguments_invalid")
        try:
            Draft202012Validator.check_schema(descriptor.argument_schema or {})
        except SchemaError:
            raise ValueError("tactical_tool_schema_invalid") from None
        try:
            ToolInvocationService._validate(descriptor.argument_schema, arguments)
        except ValueError:
            # Validation messages may contain private argument values.
            raise ValueError("tactical_tool_arguments_schema_invalid") from None
        compatibility_error = action_compatibility_error(subgoal, descriptor)
        if compatibility_error:
            raise PermissionError(compatibility_error)
        if tool_id == "native.exact_text_edit":
            path = str(arguments.get("path") or "").strip().replace("\\", "/")
            if path not in set(state.phase.mutation_scope.allowed_paths):
                raise PermissionError(f"tactical_mutation_path_not_authorized:{path or '(missing)'}")
        if tool_id == BASH_TOOL_ID:
            identity_keys = ("command", "skill_id", "artifact_id", "cwd", "runtime")
            identity = tuple(str(arguments.get(key) or "").strip() for key in identity_keys)
            for prior in state.actions:
                prior_subgoal = next((item for item in state.phase.subgoals if item.subgoal_id == prior.subgoal_id), None)
                mutation = prior_subgoal is not None and any(
                    effect != SideEffectClass.READ_ONLY for effect in prior_subgoal.allowed_side_effects
                )
                if (prior.tool_id == BASH_TOOL_ID and mutation
                        and tuple(str(prior.arguments.get(key) or "").strip() for key in identity_keys) == identity):
                    raise ValueError("tactical_shell_mutation_replay_requires_reconciliation: inspect prior effects before a new write")
        raw_questions = selected.get("acceptance_questions")
        if not isinstance(raw_questions, list) or not raw_questions:
            raise ValueError("tactical_acceptance_questions_required")
        questions = []
        seen_questions: set[str] = set()
        for item in raw_questions:
            if not isinstance(item, dict) or item.get("type") != "noul":
                raise ValueError("tactical_acceptance_question_invalid")
            question_id = str(item.get("question_id") or "").strip()
            criteria = item.get("criteria")
            if (
                not question_id or question_id in seen_questions
                or not str(item.get("instructions") or "").strip()
                or not isinstance(criteria, dict)
                or not str(criteria.get("true") or "").strip()
                or not str(criteria.get("false") or "").strip()
            ):
                raise ValueError("tactical_acceptance_question_invalid")
            seen_questions.add(question_id)
            questions.append(dict(item))
        return TacticalAction(
            action_id=str(selected.get("action_id") or f"action-{uuid4()}"),
            subgoal_id=subgoal.subgoal_id,
            tool_id=tool_id,
            arguments=dict(arguments),
            acceptance_questions=tuple(questions),
        )

    def _handle_failure(
        self, state: TacticalState, subgoal: PhaseSubgoal, action: TacticalAction
    ) -> PhaseOutcome | None:
        if (subgoal.failure_policy == FailurePolicy.LOCAL_FALLBACK
                and set(subgoal.allowed_side_effects) == {SideEffectClass.READ_ONLY}):
            failures = sum(item.status == "failed" and item.subgoal_id == subgoal.subgoal_id for item in state.actions)
            if failures < 3:
                # System Two can inspect the recorded failure and compose an
                # alternative within the same read-only stage. Writes and unknown
                # partial external effects still require outer recovery.
                return None
        state.transition(PhaseStatus.BLOCKED)
        return self._outcome(state, action.error or "A tactical action failed.", blockers=(action.error or "tactical_action_failed",))

    @staticmethod
    def _interruption(task: "TaskState", state: TacticalState, context: "CoreLoopContext") -> PhaseOutcome | None:
        if context.is_cancelled():
            state.cancellation_pending = True
            state.transition(PhaseStatus.CANCELLED)
            return PhaseExecutor._outcome(state, "Execution was cancelled.")
        if _steering_is_pending(task, context):
            state.steering_pending = True
            state.transition(PhaseStatus.BLOCKED)
            return PhaseExecutor._outcome(state, "New steering requires outer contract review.", blockers=("steering_pending",))
        return None

    @staticmethod
    def _active_subgoal(state: TacticalState) -> PhaseSubgoal:
        return next(item for item in state.phase.subgoals if item.subgoal_id == state.active_subgoal_id)

    @staticmethod
    def _checkpoint(task: "TaskState", state: TacticalState, context: "CoreLoopContext") -> None:
        task.intelligence_engine = "hierarchical_v3"
        task.intelligence_schema_version = 3
        task.hierarchical_state = state.to_dict()
        if context.question_store is not None:
            if "v3_browser_workflow" in task.metadata:
                if task.status == "running":
                    context.question_store.save_active_browser_checkpoint(task, expected_workflow=task.metadata["v3_browser_workflow"])
                else:
                    status = "done" if task.status == "completed" else task.status
                    context.question_store.save_browser_transition(task, status=status)
            else:
                context.question_store.save_task_checkpoint(task, status=task.status)

    @staticmethod
    def _outcome(state: TacticalState, reason: str, blockers: tuple[str, ...] = ()) -> PhaseOutcome:
        return PhaseOutcome(
            phase_id=state.phase.phase_id,
            status=state.status,
            reason=reason,
            evidence_refs=tuple(
                str(item.get("evidence_ref")) for item in state.evidence.entries if item.get("evidence_ref")
            ),
            blockers=blockers,
        )


def _condition_met(subgoal: PhaseSubgoal, result: Any) -> bool:
    condition = subgoal.completion
    if condition.kind in {"output_present", "tool_call_terminal"}:
        return result is not None
    if condition.kind == "field_equals":
        current = result
        for part in condition.field.split("."):
            if not isinstance(current, dict) or part not in current:
                return False
            current = current[part]
        return current == condition.expected
    return False


def _next_subgoal(state: TacticalState) -> PhaseSubgoal | None:
    completed = set(state.completed_subgoal_ids)
    for item in state.phase.subgoals:
        if item.subgoal_id not in completed and set(item.depends_on).issubset(completed):
            return item
    return None


def _steering_is_pending(task: "TaskState", context: "CoreLoopContext") -> bool:
    list_pending = getattr(context.messages, "list_pending", None)
    if not callable(list_pending):
        return False
    ignored = set(task.metadata.get("v3_intake_ignored_message_ids") or [])
    for pending in list_pending(limit=1000):
        if pending.message_id in ignored:
            continue
        message = pending.message
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        # Automation messages are separate tasks, never amendments to the
        # active task. Match planner intake so they cannot cause a review loop.
        if metadata.get("source") in {"scheduled_task", "event_automation"}:
            continue
        disposition = str(metadata.get("routing_disposition") or "pdca_task")
        if (
            disposition in {"steering", "pdca_task"}
            and message.user == task.user
            and not str(message.prompt or "").lstrip().startswith("/")
        ):
            return True
        if disposition == "correlated_response" and task.correlation_id and message.correlation_id == task.correlation_id:
            question_id = str(metadata.get("answered_question_id") or "")
            question = context.question_store.get_question(question_id) if context.question_store is not None and question_id else None
            if question is not None and question.status == "answered" and question.task_id == task.task_id and question.respondent_user_id == message.user:
                return True
    return False


def _scheduled_delivery_may_repeat(task: "TaskState", state: TacticalState, action: TacticalAction, context: "CoreLoopContext") -> bool:
    """Stop on a likely or unresolved semantic repeat; relevance never authorizes a send."""
    if task.metadata.get("source") != "scheduled_task" or action.tool_id != "native.deliver_message":
        return False
    prior = [entry for entry in state.evidence.entries
             if entry.get("tool_id") == "native.deliver_message" and entry.get("status") == "success"
             and isinstance(entry.get("result"), dict) and entry["result"].get("outbox_message_id")
             and isinstance(entry.get("arguments"), dict) and isinstance(entry["arguments"].get("message"), str)]
    if not prior:
        return False
    if not callable(context.delivery_sink):
        return True
    try:
        resolved = context.delivery_sink({"event_type": "communication.resolve_recipient", "recipient": action.arguments.get("recipient")})
        recipient = resolved.get("recipient_user_id") if isinstance(resolved, dict) and resolved.get("status") == "resolved" else None
    except Exception:
        return True
    if not recipient:
        return True
    matching = [entry for entry in prior if entry["result"].get("recipient_user_id") == recipient]
    if not matching:
        return False
    draft = str(action.arguments.get("message") or "")
    # A byte-identical retry in the same subgoal has a deterministic outbox key.
    matching = [entry for entry in matching if not (
        entry.get("phase_id") == state.phase.phase_id and entry.get("subgoal_id") == action.subgoal_id
        and entry["arguments"].get("message") == draft
    )]
    if not matching:
        return False
    review = getattr(context.system_one, "scheduled_delivery_similarity", None)
    if not callable(review):
        return True
    try:
        probabilities = review(goal=task.goal, draft=draft, prior_messages=[entry["arguments"]["message"] for entry in matching])
        threshold = float(getattr(getattr(context.system_one, "settings", None), "no_threshold", 0.2))
        return (len(probabilities) != len(matching) or any(
            type(value) not in {int, float} or not 0 <= value <= threshold for value in probabilities
        ))
    except Exception:
        return True


def _tactical_prompt(
    state: TacticalState,
    subgoal: PhaseSubgoal,
    tools: tuple["ToolDescriptor", ...],
    *,
    goal: str = "",
    task_project_root: str = "",
    attachment_manifest: list[dict[str, str]] | None = None,
    skill_guidance: str = "",
    task_context: str = "",
) -> str:
    tool_rows = [
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
    return (
        FEASIBILITY_GUIDANCE + ROUTINE_AUTONOMY_GUIDANCE +
        "Select one concrete tactical action for the current subgoal and define acceptance questions for its result. "
        "For a read-only subgoal only, you may instead return {actions:[...]} with two or three independent read-only "
        "actions, each in the same format, only when all arguments are already known and none depends on another result. "
        "For a browser fill whose field, target and private input are already known, you may instead return "
        "{browser_fill_sequence:[fill, observe, reconcile]} with those three native browser workflow actions in that "
        "exact order and the normal action format. The sequence stops on failure, uncertainty, cancellation, or changed "
        "authorization; it does not establish task completion. Do not sequence any other write, browser action, "
        "private input request, or user interaction. "
        "The runtime rechecks every action and stops the sequence on failure, uncertainty or completion. "
        "Do not change the objective, acceptance criteria, or mutation scope. "
        "Each action has tool_id, arguments, and a non-empty acceptance_questions array. Each question must be a Noul "
        "shaped as {question_id, type:'noul', instructions, criteria:{true, false}}. Questions must evaluate the tool result "
        "against the stage goal and relevant acceptance criteria. Treat successful prior action results as authoritative inputs "
        "for the next action even when they did not complete the subgoal. Do not repeat a successful discovery/list action when "
        "its result already supplies the identifier needed by another operation in the same tool schema; compose that follow-up "
        "operation with the discovered identifier.\n\n"
        "Rejected entries contain precise validation feedback. Correct the action within the unchanged phase; "
        "never repeat an equivalent rejected action or replay completed work. "
        "If progress_review.context_complete is false, the current result exceeded the bounded judgment view. "
        "The original result remains below. Obtain a targeted source expansion or narrower observation to verify the "
        "stage; do not repeat the broad search or re-execute a mutation whose effects may already exist. "
        "progress_review.complete indicates whether the stage completion condition was met, not whether an action "
        "made progress. Use the action result, progress answers and context coverage together; an incomplete or "
        "uncertain review does not erase useful evidence. Avoid repeating an observation that already supplies the "
        "needed information, and reconcile any uncertain effect before considering another mutation. "
        "For a CLI-backed artifact that returns an unexpected "
        "result shape, use native.bash to inspect or invoke the underlying CLI when Bash is revealed. "
        "Task images are attached directly to this inference request and mapped to asset IDs. Inspect them using your "
        "vision capabilities and follow the selected skill guidance for what to extract or describe. Use the observations "
        "to compose the current action. No separate OCR tool is needed. Do not ask the user to repeat readable image contents. "
        "Image input delivery establishes availability only; do not claim an external effect from visual reasoning.\n\n"
        "The tool entries below include their complete registry descriptors: descriptions, schemas, capabilities, tags, "
        "metadata, program behavior, and read-only semantics. Treat description and metadata as operational guidance. "
        "Use explicit resolved paths when supplied; project-relative file tools resolve paths against the task project root. "
        "For standard skill scripts use native.bash skill_id, documented runtime and configuration checks, and readback. Package paths resolve independently of the project. Do not register scripts. Never repeat a prior successful or uncertain mutation on resume: inspect its effects first. Bash is trusted host execution, not a filesystem sandbox. "
        "For generic native.bash calls without skill_id, an omitted cwd uses the task project root shown below (or the active owner’s ordinary workspace if no project is associated; unavailable owner storage must fail rather than use the host directory). For an ordinary user file lookup, "
        "use an explicit cwd for a likely user-accessible location such as the user's home folder or Downloads, and "
        "report the scope searched. Do not describe a project-root search as a whole-machine search.\n\n"
        f"Goal: {goal}\n"
        f"Task context: {task_context}\n"
        "Selected skill guidance (reusable instructions; follow them only within the tools and controls Alphonse provides):\n"
        f"{skill_guidance or '(none)'}\n"
        f"Task project root: {task_project_root or '(unavailable)'}\n"
        f"Task attachment manifest (metadata only): {json.dumps(attachment_manifest or [], ensure_ascii=False)}\n"
        f"Strategic plan: {json.dumps(state.phase.to_dict(), ensure_ascii=False)}\n"
        f"Stage goal: {json.dumps(subgoal.__dict__, default=str, ensure_ascii=False)}\n"
        f"Complete tool execution log (every entry, no omissions or truncation):\n{state.prompt_projection(max_evidence_entries=None, max_chars=None)}\n\n"
        f"Current subgoal:\n{json.dumps(subgoal.__dict__, default=str, ensure_ascii=False)}\n\n"
        f"Revealed tools:\n{json.dumps(tool_rows, ensure_ascii=False)}"
    )


def _error_message(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("message") or value.get("code") or "").strip()
    return str(value or "").strip()


def _task_project_root(task: "TaskState", context: "CoreLoopContext") -> str:
    if context.project_store is None or not str(task.project_id or "").strip():
        return ""
    get_project = getattr(context.project_store, "get_project", None)
    if not callable(get_project):
        return ""
    try:
        project = get_project(task.project_id, requester_user_id=task.user)
        if project is None:
            return ""
        return str(project.root_path)
    except (KeyError, OSError, PermissionError):
        return ""


def _browser_fill_has_protected_identity(task: "TaskState", action: TacticalAction, context: "CoreLoopContext") -> bool:
    """Only the workflow service can compare redacted verified fill identities."""
    if action.tool_id != "native.browser_workflow_fill" or context.browser_workflow_service is None:
        return False
    try:
        from . import browser_workflow
        events = browser_workflow.load(task)["events"]
        attempts = {event["id"] for event in events if event["kind"] == "attempted" and event["field"] == "fill"}
        return any(event["kind"] == "verified" and event["related"] in attempts for event in events)
    except Exception:
        return False


def _pending_browser_fill_recovery(task: "TaskState") -> str | None:
    from . import browser_workflow
    if browser_workflow.KEY not in task.metadata:
        return None
    try:
        workflow = browser_workflow.load(task)
        events = workflow["events"]
        resolved = {event["related"] for event in events if event["kind"] == "verified"}
        pending = [(i, event) for i, event in enumerate(events)
                   if event["kind"] == "attempted" and event["id"] not in resolved]
        if not pending:
            return None
        if len(pending) != 1:
            return "browser_reconciliation_unavailable"
        if pending[0][1]["field"] != "fill" or pending[0][1]["session"] != workflow["session"]:
            # Submission and cross-session recovery follow their existing
            # strategic recovery paths, including explicit reacquisition.
            return None
        index = pending[0][0]
        observed = any(event["kind"] == "observed" and event["field"] == "form"
                       and event["session"] == workflow["session"] for event in events[index + 1:])
        return "native.browser_workflow_reconcile" if observed else "native.browser_workflow_observe"
    except Exception:
        return "browser_reconciliation_unavailable"


def _matching_prior_incomplete_result(state: TacticalState, completed: TacticalAction) -> str | None:
    """Find equivalent prior evidence that Jev already judged incomplete."""
    if (completed.tool_id in {"native.browser_workflow_observe", "native.browser_workflow_observe_result"}
            and completed.result == {"redacted": True}):
        # Identical placeholders cannot establish identical page observations.
        # A separate deterministic per-subgoal budget bounds repeated reads.
        return None
    for evidence in reversed(state.evidence.entries[:-1]):
        review = evidence.get("progress_review")
        if not isinstance(review, dict) or review.get("complete") is not False:
            continue
        if (
            str(evidence.get("subgoal_id") or "") == completed.subgoal_id
            and str(evidence.get("tool_id") or "") == completed.tool_id
            and evidence.get("arguments") == completed.arguments
            and evidence.get("result") == completed.result
        ):
            action_ref = str(evidence.get("evidence_ref") or "").removeprefix("tactical-action:")
            return action_ref or "unknown"
    return None


def _confident_probability(value, threshold: float) -> bool:
    import math
    return type(value) in {int, float} and math.isfinite(value) and threshold <= value <= 1.0


def _confident_negative(value, threshold: float) -> bool:
    import math
    return type(value) in {int, float} and math.isfinite(value) and 0.0 <= value <= 1.0 - threshold


def _is_update_inspection(state: TacticalState, subgoal: PhaseSubgoal, answers: dict, threshold: float) -> bool:
    return (set(subgoal.allowed_side_effects) == {SideEffectClass.READ_ONLY}
            or _confident_probability(answers.get("runtime_action_is_inspection"), threshold)) and any(
        set(item.allowed_side_effects) & {
            SideEffectClass.PROJECT_MUTATION, SideEffectClass.EXTERNAL_REVERSIBLE,
            SideEffectClass.EXTERNAL_IRREVERSIBLE, SideEffectClass.LOCAL_SKILL_INSTALLATION,
        }
        for item in state.phase.subgoals
    )


def _validate_browser_fill_sequence(executor: PhaseExecutor, state: TacticalState, subgoal: PhaseSubgoal,
                                    selected: dict[str, Any], tools: tuple["ToolDescriptor", ...]) -> list[dict[str, Any]]:
    """Accept only one fill and its mandatory independent recovery steps.

    The ephemeral proposal confers no authority and is discarded on restart.
    Each member still passes the normal reveal, phase and invocation checks.
    """
    items = selected.get("browser_fill_sequence")
    expected = ["native.browser_workflow_fill", "native.browser_workflow_observe",
                "native.browser_workflow_reconcile"]
    if (set(selected) - {"browser_fill_sequence", "uncertainty_assessment"} or not isinstance(items, list)
            or len(items) != len(expected)
            or any(not isinstance(item, dict) or item.get("tool_id") != name
                   for item, name in zip(items, expected))):
        raise ValueError("browser_fill_sequence_invalid")
    for item in items:
        action = executor._validate_action(state, subgoal, item, tools)
        descriptor = next(tool for tool in tools if tool.tool_id == action.tool_id)
        ToolInvocationService._validate(descriptor.argument_schema, action.arguments)
    return [{key: value for key, value in item.items() if key != "action_id"} for item in items]


def _validate_independent_reads(executor: PhaseExecutor, state: TacticalState, subgoal: PhaseSubgoal,
                                selected: dict[str, Any], tools: tuple["ToolDescriptor", ...]) -> list[dict[str, Any]]:
    """Validate the entire ephemeral sequence before any action can run."""
    items = selected.get("actions")
    if set(subgoal.allowed_side_effects) != {SideEffectClass.READ_ONLY}:
        raise ValueError("independent_reads_require_read_only_subgoal")
    if not isinstance(items, list) or not 2 <= len(items) <= MAX_INDEPENDENT_READ_ACTIONS:
        raise ValueError("independent_reads_invalid_count")
    from alphonse.agent_v2.system_one import _TACTICAL_RETRY_FUSE_QUESTIONS
    reserved = {"runtime_declared_subgoal_complete", "runtime_action_added_progress", "runtime_action_is_inspection",
                "runtime_inspection_ready_for_action", "runtime_inspection_requires_user_input"}
    reserved.update(question["question_id"] for question in _TACTICAL_RETRY_FUSE_QUESTIONS)
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict) or "actions" in item:
            raise ValueError("independent_reads_invalid_action")
        descriptor = next((tool for tool in tools if tool.tool_id == item.get("tool_id")), None)
        if (descriptor is None or not descriptor.read_only or descriptor.metadata.get("sensitive_content") is True
                or descriptor.tool_id.startswith("native.browser_workflow_")
                or descriptor.tool_id in {"native.ask_question", "native.respond"}
                or descriptor.metadata.get("side_effect_class") not in {None, "read_only"}):
            raise ValueError("independent_reads_tool_not_allowed")
        executor._validate_action(state, subgoal, item, tools)
        ToolInvocationService._validate(descriptor.argument_schema, item["arguments"])
        if any(str(question.get("question_id") or "").strip() in reserved for question in item["acceptance_questions"]):
            raise ValueError("independent_reads_reserved_question")
        signature = json.dumps([item.get("tool_id"), item.get("arguments")], sort_keys=True, default=str)
        if signature in seen:
            raise ValueError("independent_reads_duplicate")
        seen.add(signature)
    # Action IDs are runtime-owned; a repeated model-authored ID would alias
    # distinct evidence entries. Generate a fresh one at each actual dispatch.
    return [{key: value for key, value in item.items() if key != "action_id"} for item in items]


def _independent_read_can_continue(review: dict[str, Any], yes_threshold: float) -> bool:
    coverage = review.get("context_coverage") or {}
    return (review.get("confident") is True and review.get("context_complete") is not False
            and not coverage.get("omitted_prior_count", 0)
            and not coverage.get("omitted_prior_result_count", 0) and not coverage.get("omitted_prior_refs")
            and _confident_probability((review.get("answers") or {}).get("runtime_action_added_progress"), yes_threshold))


def _queue_uncertainty_recovery(state: TacticalState, subgoal: PhaseSubgoal, context: "CoreLoopContext") -> None:
    """Queue at most one stage and one other recovery per phase/subgoal."""
    import math
    relevant = [entry for entry in state.evidence.entries
                if entry.get("phase_id") == state.phase.phase_id and entry.get("subgoal_id") == subgoal.subgoal_id]
    if not relevant:
        return
    review = relevant[-1].get("progress_review") or {}
    if review.get("uncertainty_recovery"):
        return
    settings = getattr(context.system_one, "settings", None)
    no = float(getattr(settings, "no_threshold", .2))
    yes = float(getattr(settings, "yes_threshold", .8))
    answers = review.get("answers") or {}
    coverage = review.get("context_coverage") or {}
    incomplete = (review.get("context_complete") is False or coverage.get("omitted_prior_count", 0)
                  or coverage.get("omitted_prior_result_count", 0) or coverage.get("omitted_prior_refs"))
    # Stable question IDs let System Two find the exact question and evidence in
    # the execution log. Give the runtime stage/progress questions precedence.
    keys = [key for key in ("runtime_declared_subgoal_complete", "runtime_action_added_progress") if key in answers]
    authored_ids = {item.get("question_id") for item in relevant[-1].get("acceptance_questions", [])
                    if isinstance(item, dict) and isinstance(item.get("question_id"), str)}
    keys.extend(sorted(key for key in authored_ids if key in answers and key not in keys))
    uncertain = next((key for key in keys if type(answers[key]) in {int, float}
                      and math.isfinite(answers[key]) and no < answers[key] < yes), None)
    if not incomplete and uncertain is None:
        return
    budget_class = "stage" if uncertain == "runtime_declared_subgoal_complete" else "other"
    for entry in relevant:
        previous = (entry.get("progress_review") or {}).get("uncertainty_recovery") or {}
        # Older checkpoints have no budget_class. Derive it from their question.
        previous_class = previous.get("budget_class") or (
            "stage" if previous.get("question_id") == "runtime_declared_subgoal_complete" else "other")
        if previous and previous_class == budget_class:
            return
    review["uncertainty_recovery"] = {"status": "pending", "reason": "incomplete_context" if incomplete else "ambiguous_judgment",
                                      "question_id": uncertain, "budget_class": budget_class}


def _consume_uncertainty_recovery(state: TacticalState, subgoal: PhaseSubgoal) -> dict[str, Any] | None:
    for entry in reversed(state.evidence.entries):
        if entry.get("phase_id") != state.phase.phase_id or entry.get("subgoal_id") != subgoal.subgoal_id:
            continue
        review = entry.get("progress_review") or {}
        recovery = review.get("uncertainty_recovery") or {}
        if recovery.get("status") != "pending":
            continue
        recovery["status"] = "consumed"
        question_id = recovery.get("question_id")
        runtime_questions = {
            "runtime_declared_subgoal_complete": "Does the current action and execution log establish the declared stage goal and completion condition?",
            "runtime_action_added_progress": "Did the latest action add meaningful evidence or enable a concrete next step compared with earlier entries?",
            "runtime_action_is_inspection": "Was the latest action only read-only inspection?",
            "runtime_inspection_ready_for_action": "Does the authorized evidence resolve the details needed to plan the requested operation?",
            "runtime_inspection_requires_user_input": "Is a specific essential missing or conflicting detail dependent on a user answer?",
        }
        authored = next((item.get("instructions") for item in entry.get("acceptance_questions", [])
                         if isinstance(item, dict) and item.get("question_id") == question_id), None)
        return {
            "question": authored or runtime_questions.get(question_id) or "Which targeted observation can resolve the recorded evidence gap?",
            "question_id": question_id,
            "evidence_ref": entry.get("evidence_ref"), "reason": recovery["reason"],
            "probability": (review.get("answers") or {}).get(recovery.get("question_id")),
            "context_complete": review.get("context_complete"),
            "context_coverage": review.get("context_coverage") or {},
        }
    return None


def _record_uncertainty_assessment(state: TacticalState, recovery: dict[str, Any], selected: Any) -> None:
    """Record only a bounded advisory and references to existing successful evidence."""
    target = next((entry for entry in state.evidence.entries
                   if entry.get("evidence_ref") == recovery.get("evidence_ref")), None)
    marker = ((target or {}).get("progress_review") or {}).get("uncertainty_recovery") or {}
    if not marker or marker.get("status") != "consumed":
        return
    assessment = selected.get("uncertainty_assessment") if isinstance(selected, dict) else None
    valid_refs = {entry.get("evidence_ref") for entry in state.evidence.entries
                  if entry.get("status") == "success" and isinstance(entry.get("evidence_ref"), str)}
    if not isinstance(assessment, dict):
        marker["assessment"] = {"status": "unavailable"}
        return
    answer, refs = assessment.get("answer"), assessment.get("evidence_refs")
    if (assessment.get("question_id") != recovery.get("question_id")
            or not isinstance(answer, str) or answer not in {"supported", "not_supported", "uncertain"}
            or not isinstance(refs, list) or len(refs) > 20
            or any(not isinstance(ref, str) or ref not in valid_refs for ref in refs)
            or (answer != "uncertain" and not refs)):
        marker["assessment"] = {"status": "invalid"}
        return
    marker["assessment"] = {"status": "recorded", "answer": answer, "evidence_refs": list(dict.fromkeys(refs))}


def _consecutive_no_progress(state: TacticalState, subgoal_id: str, threshold: float = 0.2) -> int:
    import math
    count = 0
    for entry in reversed(state.evidence.entries):
        if entry.get("phase_id") != state.phase.phase_id or entry.get("subgoal_id") != subgoal_id:
            continue
        review = entry.get("progress_review") or {}
        coverage = review.get("context_coverage") or {}
        if (review.get("context_complete") is False or coverage.get("omitted_prior_count", 0)
                or coverage.get("omitted_prior_result_count", 0) or coverage.get("omitted_prior_refs")):
            break
        answer = (review.get("answers") or {}).get("runtime_action_added_progress")
        if type(answer) not in {int, float} or not math.isfinite(answer) or not 0 <= answer <= threshold:
            break
        count += 1
    return count


def _consecutive_uncertain_progress(state: TacticalState, subgoal_id: str, no_threshold: float, yes_threshold: float) -> int:
    import math
    count = 0
    for entry in reversed(state.evidence.entries):
        if entry.get("phase_id") != state.phase.phase_id or entry.get("subgoal_id") != subgoal_id:
            continue
        answer = (entry.get("progress_review") or {}).get("answers", {}).get("runtime_action_added_progress")
        if type(answer) not in {int, float} or not math.isfinite(answer) or not no_threshold < answer < yes_threshold:
            break
        count += 1
    return count


def _has_task_image(task: "TaskState") -> bool:
    attachments = task.metadata.get("attachments") if isinstance(task.metadata, dict) else None
    return any(
        isinstance(item, dict) and str(item.get("mime_type") or "").lower().startswith("image/")
        for item in (attachments if isinstance(attachments, list) else [])
    )


def _task_attachment_manifest(task: "TaskState") -> list[dict[str, str]]:
    attachments = task.metadata.get("attachments") if isinstance(task.metadata, dict) else None
    manifest = []
    for item in attachments if isinstance(attachments, list) else []:
        if not isinstance(item, dict) or not str(item.get("asset_id") or "").strip():
            continue
        manifest.append({
            "asset_id": str(item.get("asset_id"))[:200],
            "filename": str(item.get("filename") or "")[:200],
            "mime_type": str(item.get("mime_type") or "")[:100],
            "kind": str(item.get("kind") or "")[:100],
        })
        if len(manifest) == 20:
            break
    return manifest


def _image_inputs_provided(state: TacticalState, subgoal_id: str) -> bool:
    return any(
        str(item.get("subgoal_id") or "") == subgoal_id
        and item.get("kind") == "inference_image_input"
        and item.get("status") == "provided"
        and bool(item.get("asset_ids"))
        for item in state.evidence.entries
    )


def _retry_is_safe(action: TacticalAction, context: "CoreLoopContext") -> bool:
    registry = context.tools
    if registry is None:
        return False
    descriptor = registry.get(action.tool_id) if callable(getattr(registry, "get", None)) else None
    if descriptor is None:
        descriptor = next(
            (item for item in registry.list() if str(getattr(item, "tool_id", "")) == action.tool_id),
            None,
        ) if callable(getattr(registry, "list", None)) else None
    if bool(getattr(descriptor, "read_only", False)):
        return True
    result = action.result if isinstance(action.result, dict) else {}
    error = result.get("exception") if isinstance(result.get("exception"), dict) else {}
    return result.get("retry_safe") is True or error.get("retry_safe") is True


def _wait_for_retry(delay: float, task: "TaskState", state: TacticalState, context: "CoreLoopContext") -> bool:
    deadline = time.monotonic() + max(0.0, delay)
    while time.monotonic() < deadline:
        if context.is_cancelled() or _steering_is_pending(task, context):
            return False
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    return True


def _record_system_one_tactical_review(
    task: "TaskState", state: TacticalState, subgoal: PhaseSubgoal,
    action: TacticalAction, metadata: dict[str, Any],
) -> None:
    history = task.metadata.setdefault("system_one_tactical_reviews", [])
    if not isinstance(history, list):
        history = []
        task.metadata["system_one_tactical_reviews"] = history
    history.append({
        "phase_id": state.phase.phase_id,
        "subgoal_id": subgoal.subgoal_id,
        "action_id": action.action_id,
        **metadata,
    })
    if len(history) > 50:
        del history[:-50]


def _safe_tactical_generate(inference: Any, request: InferenceRequest) -> Any:
    """Keep provider-supplied exception text out of terminal errors and traces."""
    try:
        return inference.generate_json(request)
    except Exception as exc:
        if isinstance(exc, ValueError) and str(exc) in {"openai_codex_invalid_json", "openai_codex_invalid_tactical_schema", "inference_cancelled"}:
            raise ValueError(str(exc)) from None
        if isinstance(exc, ValueError):
            raise ValueError("tactical_provider_error") from None
        raise RuntimeError("tactical_provider_error") from None
