"""Bounded tactical execution inside one hierarchical CAPD Do phase."""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING, Any, Callable
from uuid import uuid4

from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.intelligence.v3.contracts import FailurePolicy
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseOutcome
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseStatus
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseSubgoal
from alphonse.agent_v2.core.intelligence.v3.contracts import SideEffectClass
from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalAction
from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalState
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.core.messages.queue import MessageSelector
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor
    from alphonse.agent_v2.core.intelligence.task_state import TaskState

ActionSelector = Callable[[TacticalState, PhaseSubgoal, tuple["ToolDescriptor", ...]], dict[str, Any] | None]


class PhaseExecutor:
    """Run tactical actions until a phase reaches a bounded outcome."""

    def __init__(self, *, action_selector: ActionSelector | None = None, reveal_policy: ToolRevealPolicy | None = None) -> None:
        self._action_selector = action_selector
        self._reveal_policy = reveal_policy

    def run(self, task: "TaskState", state: TacticalState, context: "CoreLoopContext") -> PhaseOutcome:
        if state.status == PhaseStatus.PLANNED:
            state.transition(PhaseStatus.RUNNING)
            self._checkpoint(task, state)
            context.emit_activity(
                phase=ImprovementPhase.DO,
                label="phase started",
                message=state.phase.objective,
                progress={"phase_id": state.phase.phase_id, "subgoal_id": state.active_subgoal_id},
            )
        while state.status == PhaseStatus.RUNNING:
            interruption = self._interruption(task, state, context)
            if interruption is not None:
                self._checkpoint(task, state)
                return interruption
            subgoal = self._active_subgoal(state)
            tools = self._revealed_tools(task, state, subgoal, context)
            selected = self._select_action(task, state, subgoal, tools, context)
            if selected is None:
                state.transition(PhaseStatus.BLOCKED)
                self._checkpoint(task, state)
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
                self._checkpoint(task, state)
                return self._outcome(state, str(exc), blockers=(str(exc),))
            state.consume_tool_call()
            state.actions.append(action)
            self._checkpoint(task, state)
            context.emit_activity(
                phase=ImprovementPhase.DO,
                label="tactical action",
                message=subgoal.objective,
                progress={
                    "phase_id": state.phase.phase_id,
                    "subgoal_id": subgoal.subgoal_id,
                    "action_id": action.action_id,
                    "tool_id": action.tool_id,
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
            )
            state.actions[-1] = completed
            state.evidence.append(
                {
                    "evidence_ref": f"tactical-action:{completed.action_id}",
                    "phase_id": state.phase.phase_id,
                    "subgoal_id": subgoal.subgoal_id,
                    "tool_id": completed.tool_id,
                    "status": completed.status,
                    "result": completed.result,
                    "error": completed.error,
                }
            )
            self._checkpoint(task, state)
            if completed.status == "waiting":
                state.transition(PhaseStatus.WAITING_USER)
                self._checkpoint(task, state)
                return self._outcome(state, "A tactical action requires user input.")
            if completed.status != "success":
                failure = self._handle_failure(state, subgoal, completed)
                self._checkpoint(task, state)
                if failure is not None:
                    return failure
                continue
            if not _condition_met(subgoal, completed.result):
                if self._subgoal_calls(state, subgoal.subgoal_id) >= subgoal.limits.max_tool_calls:
                    state.transition(PhaseStatus.BUDGET_EXHAUSTED)
                    self._checkpoint(task, state)
                    return self._outcome(state, "The subgoal call budget was exhausted before its completion condition was met.")
                continue
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
                self._checkpoint(task, state)
                return self._outcome(state, "All phase subgoals completed with recorded evidence.")
            state.active_subgoal_id = next_subgoal.subgoal_id
            state.revealed_capabilities = []
            state.revealed_tool_ids = []
            state.transition(PhaseStatus.RUNNING)
            self._checkpoint(task, state)
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
            return self._action_selector(state, subgoal, tools)
        if context.inference is None:
            return None
        prompt = _tactical_prompt(state, subgoal, tools)
        result = context.inference.generate_json(
            InferenceRequest(
                prompt=prompt,
                purpose=InferencePurpose.TACTICAL_ACTION,
                project_id=task.project_id,
                user=task.user,
                task_id=task.task_id,
                tools=tools,
                metadata={"phase_id": state.phase.phase_id, "subgoal_id": subgoal.subgoal_id},
            )
        )
        return dict(result.json_value) if isinstance(result.json_value, dict) else None

    def _revealed_tools(
        self, task: "TaskState", state: TacticalState, subgoal: PhaseSubgoal, context: "CoreLoopContext"
    ) -> tuple["ToolDescriptor", ...]:
        if context.tools is None:
            return ()
        available = tuple(context.tools.list())
        if self._reveal_policy is not None and not state.revealed_tool_ids:
            reveal = self._reveal_policy.reveal(task, state, subgoal, available)
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
            allowed_ids = {item.tool_id for item in available if item.tool_id in state.phase.authorized_capabilities}
            state.revealed_tool_ids = sorted(allowed_ids)
        return tuple(item for item in available if item.tool_id in allowed_ids)

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
        return TacticalAction(
            action_id=str(selected.get("action_id") or f"action-{uuid4()}"),
            subgoal_id=subgoal.subgoal_id,
            tool_id=tool_id,
            arguments=dict(arguments),
        )

    def _handle_failure(
        self, state: TacticalState, subgoal: PhaseSubgoal, action: TacticalAction
    ) -> PhaseOutcome | None:
        if subgoal.failure_policy == FailurePolicy.LOCAL_FALLBACK and self._subgoal_calls(state, subgoal.subgoal_id) < subgoal.limits.max_tool_calls and int(state.remaining_tool_calls or 0) > 0:
            return None
        if subgoal.failure_policy == FailurePolicy.WAIT_USER:
            state.transition(PhaseStatus.WAITING_USER)
            return self._outcome(state, action.error or "A user decision is required.")
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
    def _checkpoint(task: "TaskState", state: TacticalState) -> None:
        task.intelligence_engine = "hierarchical_v3"
        task.intelligence_schema_version = 3
        task.hierarchical_state = state.to_dict()

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
    pending = context.messages.peek(MessageSelector(user=task.user, project_id=task.project_id))
    if pending is None:
        return False
    metadata = pending.message.metadata if isinstance(pending.message.metadata, dict) else {}
    return str(metadata.get("routing_disposition") or "") in {"steering", "correlated_response"}


def _tactical_prompt(state: TacticalState, subgoal: PhaseSubgoal, tools: tuple["ToolDescriptor", ...]) -> str:
    tool_rows = [
        {"tool_id": item.tool_id, "name": item.name, "description": item.description, "schema": item.argument_schema}
        for item in tools
    ]
    return (
        "Select exactly one concrete tactical action for the current bounded subgoal. "
        "Do not change the objective, acceptance criteria, mutation scope, or budgets. "
        "Return JSON only: {\"tool_id\":\"...\",\"arguments\":{...}}.\n\n"
        f"Phase state:\n{state.prompt_projection(max_evidence_entries=6, max_chars=9000)}\n\n"
        f"Current subgoal:\n{json.dumps(subgoal.__dict__, default=str, ensure_ascii=False)}\n\n"
        f"Revealed tools:\n{json.dumps(tool_rows, ensure_ascii=False)}"
    )


def _error_message(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("message") or value.get("code") or "").strip()
    return str(value or "").strip()
