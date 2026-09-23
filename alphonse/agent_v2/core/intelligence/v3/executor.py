"""Bounded tactical execution inside one hierarchical CAPD Do phase."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseOutcome
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseStatus
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseSubgoal
from alphonse.agent_v2.core.intelligence.v3.contracts import SideEffectClass
from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalAction
from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalState
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.core.tools.registry.native.respond import RESPOND_TOOL_ID
from alphonse.agent_v2.system_one import SystemOneUnavailableError

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor
    from alphonse.agent_v2.core.intelligence.task_state import TaskState

ActionSelector = Callable[[TacticalState, PhaseSubgoal, tuple["ToolDescriptor", ...]], dict[str, Any] | None]
MAX_TACTICAL_RETRIES = 2
TACTICAL_RETRY_BASE_SECONDS = 1.0


class PhaseExecutor:
    """Run tactical actions until a phase reaches a bounded outcome."""

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
        while state.status == PhaseStatus.RUNNING:
            interruption = self._interruption(task, state, context)
            if interruption is not None:
                self._checkpoint(task, state, context)
                return interruption
            subgoal = self._active_subgoal(state)
            tools = self._revealed_tools(task, state, subgoal, context)
            selected = self._select_action(task, state, subgoal, tools, context)
            if selected is None:
                state.transition(PhaseStatus.BLOCKED)
                self._checkpoint(task, state, context)
                return self._outcome(state, "No valid tactical action was available.", blockers=("tactical_action_unavailable",))
            try:
                action = self._validate_action(state, subgoal, selected, tools)
            except (ValueError, PermissionError) as exc:
                state.evidence.append(
                    {
                        "evidence_ref": f"tactical-rejection:{uuid4()}",
                        "phase_id": state.phase.phase_id,
                        "subgoal_id": subgoal.subgoal_id,
                        "status": "rejected",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                state.transition(PhaseStatus.BLOCKED)
                self._checkpoint(task, state, context)
                return self._outcome(state, str(exc), blockers=(str(exc),))
            retry_count = 0
            retry_interrupted = False
            while True:
                if retry_count:
                    interruption = self._interruption(task, state, context)
                    if interruption is not None:
                        self._checkpoint(task, state, context)
                        return interruption
                state.consume_tool_call()
                state.actions.append(action)
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
                        "remaining_tool_calls": state.remaining_tool_calls,
                    },
                )
                outcome = ToolInvocationService(context=context, task=task).invoke(
                    action.tool_id, action.arguments, call_id=action.action_id
                )
                completed = TacticalAction(
                    action_id=action.action_id,
                    subgoal_id=action.subgoal_id,
                    tool_id=action.tool_id,
                    arguments=action.arguments,
                    status=str(outcome.get("status") or "failed"),
                    result=outcome.get("result"),
                    error=_error_message(outcome.get("error")),
                    acceptance_questions=action.acceptance_questions,
                )
                state.actions[-1] = completed
                state.evidence.append(
                    {
                        "evidence_ref": f"tactical-action:{completed.action_id}",
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
                self._checkpoint(task, state, context)
                completion_met, retry_approved = self._completion_review(task, state, subgoal, completed, context)
                retry_available = (
                    completed.status == "failed"
                    and retry_approved
                    and retry_count < MAX_TACTICAL_RETRIES
                    and int(state.remaining_tool_calls or 0) > 0
                    and self._subgoal_calls(state, subgoal.subgoal_id) < subgoal.limits.max_tool_calls
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
            if retry_interrupted:
                interruption = self._interruption(task, state, context)
                if interruption is not None:
                    self._checkpoint(task, state, context)
                    return interruption
            if completed.status == "waiting":
                state.transition(PhaseStatus.WAITING_USER)
                self._checkpoint(task, state, context)
                return self._outcome(state, "A tactical action requires user input.")
            if completed.status != "success":
                failure = self._handle_failure(state, subgoal, completed)
                self._checkpoint(task, state, context)
                if failure is not None:
                    return failure
                continue
            if not completion_met:
                if self._subgoal_calls(state, subgoal.subgoal_id) >= subgoal.limits.max_tool_calls:
                    state.transition(PhaseStatus.BUDGET_EXHAUSTED)
                    self._checkpoint(task, state, context)
                    return self._outcome(state, "The subgoal call budget was exhausted before its completion condition was met.")
                continue
            if completed.tool_id == RESPOND_TOOL_ID and isinstance(completed.result, dict):
                message = str(completed.result.get("message") or "").strip()
                if message:
                    task.metadata["prepared_user_response"] = {
                        "source": RESPOND_TOOL_ID,
                        "tool_call_id": completed.action_id,
                        "message": message,
                    }
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
        if self._action_selector is not None:
            selected = self._action_selector(state, subgoal, tools)
            if isinstance(selected, dict) and not selected.get("acceptance_questions"):
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
        prompt = _tactical_prompt(state, subgoal, tools, goal=task.goal)
        result = context.inference.generate_json(
            InferenceRequest(
                prompt=prompt,
                purpose=InferencePurpose.TACTICAL_ACTION,
                project_id=task.project_id,
                user=task.user,
                task_id=task.task_id,
                tools=tools,
                metadata={"phase_id": state.phase.phase_id, "subgoal_id": subgoal.subgoal_id},
                cancel_checker=context.is_cancelled if context.cancellation_checker is not None else None,
            )
        )
        return dict(result.json_value) if isinstance(result.json_value, dict) else None

    def _revealed_tools(
        self, task: "TaskState", state: TacticalState, subgoal: PhaseSubgoal, context: "CoreLoopContext"
    ) -> tuple["ToolDescriptor", ...]:
        if context.tools is None:
            return ()
        available = tuple(context.tools.list())
        candidates = available
        if state.phase.tool_curation_status == "used":
            relevant = set(state.phase.curated_tool_ids)
            candidates = tuple(item for item in available if item.tool_id in relevant)
        if self._reveal_policy is not None and not state.revealed_tool_ids:
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
    ) -> tuple[bool, bool]:
        deterministic = _condition_met(subgoal, action.result)
        if context.system_one is None or not hasattr(context.system_one, "evaluate_tactical_progress"):
            raise SystemOneUnavailableError("tactical_acceptance_unavailable")
        try:
            review = context.system_one.evaluate_tactical_progress(
                goal=task.goal,
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
            )
        except Exception as exc:
            raise SystemOneUnavailableError(f"tactical_acceptance:{type(exc).__name__}") from exc
        else:
            metadata = {"status": "used" if review.confident else "ambiguous_fallback", **review.to_metadata()}
            result = review.complete if review.confident else deterministic
            retry_approved = bool(getattr(review, "retry_approved", False))
            metadata["retry_approved"] = retry_approved
        _record_system_one_tactical_review(task, state, subgoal, action, metadata)
        context.emit_telemetry({
            "event": "system_one_tactical_review", "task_id": task.task_id,
            "phase_id": state.phase.phase_id, "subgoal_id": subgoal.subgoal_id,
            "action_id": action.action_id, **metadata,
        })
        return result, retry_approved

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
        if not descriptor.read_only and SideEffectClass.READ_ONLY in subgoal.allowed_side_effects and len(subgoal.allowed_side_effects) == 1:
            raise PermissionError(f"tactical_side_effect_not_authorized:{tool_id}")
        if tool_id == "native.exact_text_edit":
            path = str(arguments.get("path") or "").strip().replace("\\", "/")
            if path not in set(state.phase.mutation_scope.allowed_paths):
                raise PermissionError(f"tactical_mutation_path_not_authorized:{path or '(missing)'}")
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
        if int(state.remaining_tool_calls or 0) <= 0:
            state.transition(PhaseStatus.BUDGET_EXHAUSTED)
            return PhaseExecutor._outcome(state, "The phase tool-call budget was exhausted.")
        if state.deadline_at and datetime.now().astimezone() >= datetime.fromisoformat(state.deadline_at):
            state.transition(PhaseStatus.BUDGET_EXHAUSTED)
            return PhaseExecutor._outcome(state, "The phase deadline was reached.")
        return None

    @staticmethod
    def _active_subgoal(state: TacticalState) -> PhaseSubgoal:
        return next(item for item in state.phase.subgoals if item.subgoal_id == state.active_subgoal_id)

    @staticmethod
    def _subgoal_calls(state: TacticalState, subgoal_id: str) -> int:
        return sum(1 for item in state.actions if item.subgoal_id == subgoal_id)

    @staticmethod
    def _checkpoint(task: "TaskState", state: TacticalState, context: "CoreLoopContext") -> None:
        task.intelligence_engine = "hierarchical_v3"
        task.intelligence_schema_version = 3
        task.hierarchical_state = state.to_dict()
        if context.question_store is not None:
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
        disposition = str(metadata.get("routing_disposition") or "")
        if disposition == "steering" and message.user == task.user and message.project_id == task.project_id:
            return True
        if disposition == "correlated_response" and task.correlation_id and message.correlation_id == task.correlation_id:
            question_id = str(metadata.get("answered_question_id") or "")
            question = context.question_store.get_question(question_id) if context.question_store is not None and question_id else None
            if question is not None and question.status == "answered" and question.task_id == task.task_id and question.respondent_user_id == message.user:
                return True
    return False


def _tactical_prompt(
    state: TacticalState, subgoal: PhaseSubgoal, tools: tuple["ToolDescriptor", ...], *, goal: str = ""
) -> str:
    tool_rows = [
        {"tool_id": item.tool_id, "name": item.name, "description": item.description, "schema": item.argument_schema}
        for item in tools
    ]
    return (
        "Select exactly one concrete tactical action for the current bounded subgoal and define acceptance questions for its result. "
        "Do not change the objective, acceptance criteria, mutation scope, or budgets. "
        "Return JSON with tool_id, arguments, and a non-empty acceptance_questions array. Each item must be a Noul question "
        "shaped as {question_id, type:'noul', instructions, criteria:{true, false}}. Questions must evaluate the tool result "
        "against the stage goal and relevant acceptance criteria.\n\n"
        f"Goal: {goal}\n"
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


def _record_system_one_registry_selection(
    task: "TaskState", state: TacticalState, metadata: dict[str, Any]
) -> None:
    history = task.metadata.setdefault("system_one_tool_registry_selections", [])
    if not isinstance(history, list):
        history = []
        task.metadata["system_one_tool_registry_selections"] = history
    history.append({"phase_id": state.phase.phase_id, **metadata})
    if len(history) > 20:
        del history[:-20]


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
