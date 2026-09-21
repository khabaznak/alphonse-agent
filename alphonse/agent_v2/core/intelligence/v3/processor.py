"""Hierarchical V3 processor and per-task engine router."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from alphonse.agent_v2.core.core import ImprovementPhase, ProcessingResult, ProcessingStatus, StateSnapshot
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
from alphonse.agent_v2.core.intelligence.pdca.nodes.act_node import act_node
from alphonse.agent_v2.core.intelligence.pdca.nodes.check_node import check_node
from alphonse.agent_v2.core.intelligence.v3.contracts import PhaseStatus, TacticalState, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor
from alphonse.agent_v2.core.intelligence.v3.outer import V3OuterController
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase
from alphonse.agent_v2.core.intelligence.v3.planner import V3PhasePlanValidationError
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext, IntelligenceProcessor
    from alphonse.agent_v2.core.intelligence.task_state import TaskState


class HierarchicalCAPDProcessor:
    def __init__(self, *, max_phases: int = 8) -> None:
        self.max_phases = max(1, max_phases)
        self.executor = PhaseExecutor(reveal_policy=ToolRevealPolicy())
        self.outer = V3OuterController()

    def process(self, task: "TaskState", context: "CoreLoopContext") -> ProcessingResult:
        context.emit_ui_event("run_started", {"task": task.to_dict(), "engine": "hierarchical_v3"})
        task.intelligence_engine = "hierarchical_v3"
        task.intelligence_schema_version = 3
        if self._should_respond_directly(task, context):
            self._complete_direct_response(task, context)
            self._persist(task, context)
            return self._result(task, context)
        if not task.acceptance_contract:
            check_node(task, context=context)
            act_node(task, context=context)
            self._persist(task, context)
        if not task.acceptance_contract:
            task.status = "failed"
            task.outcome = {"status": "failure", "reason": "V3 could not establish acceptance criteria."}
            return self._result(task, context)

        phases = 0
        while phases < self.max_phases and task.status not in {"completed", "failed", "waiting_user", "cancelled"}:
            phases += 1
            try:
                if task.hierarchical_state:
                    state = TacticalState.from_dict(task.hierarchical_state)
                    if state.status.value in {"planned", "running", "waiting_user"}:
                        if state.status.value == "waiting_user":
                            state.status = PhaseStatus.RUNNING
                    else:
                        state = self._new_state(task, context)
                else:
                    state = self._new_state(task, context)
                    self._persist(task, context)
            except V3PhasePlanValidationError as exc:
                task.status = "failed"
                task.outcome = {"status": "failure", "reason": str(exc)}
                self._persist(task, context)
                break
            outcome = self.executor.run(task, state, context)
            self._append_history(task, state, outcome.to_dict())
            review, decision = self.outer.review_and_route(task, state, outcome, context)
            self._persist(task, context)
            if task.metadata.get("v3_route") in {"plan_next_phase", "strategic_replan"}:
                task.hierarchical_state = {}
                if state.steering_pending:
                    check_node(task, context=context)
                    act_node(task, context=context)
                continue
            _ = review, decision
            break
        if phases >= self.max_phases and task.status == "running":
            task.status = "failed"
            task.outcome = {"status": "failure", "reason": "V3 phase budget exhausted without a terminal outcome."}
        return self._result(task, context)

    @staticmethod
    def _new_state(task: "TaskState", context: "CoreLoopContext") -> TacticalState:
        phase = plan_phase(task, context)
        prior = [
            entry for item in task.metadata.get("v3_phase_history") or []
            for entry in (item.get("evidence") or {}).get("entries", []) if isinstance(entry, dict)
        ]
        return new_tactical_state(phase, cumulative_evidence=prior)

    @staticmethod
    def _append_history(task: "TaskState", state: TacticalState, outcome: dict) -> None:
        history = task.metadata.setdefault("v3_phase_history", [])
        if isinstance(history, list):
            history.append({"phase": state.phase.to_dict(), "outcome": outcome, "evidence": state.evidence.to_dict()})

    @staticmethod
    def _should_respond_directly(task: "TaskState", context: "CoreLoopContext") -> bool:
        if context.system_one is not None and hasattr(context.system_one, "classify_direct_response"):
            try:
                decision = context.system_one.classify_direct_response(goal=task.goal)
            except Exception as exc:
                metadata = {"status": "fallback", "error_type": type(exc).__name__}
            else:
                metadata = {"status": "used" if decision.confident else "ambiguous_fallback", **decision.to_metadata()}
                task.metadata["system_one_direct_response"] = metadata
                context.emit_telemetry({"event": "system_one_direct_response", "task_id": task.task_id, **metadata})
                if decision.confident and decision.direct_response:
                    return True
                return _obvious_social_message(task.goal)
            task.metadata["system_one_direct_response"] = metadata
            context.emit_telemetry({"event": "system_one_direct_response", "task_id": task.task_id, **metadata})
        return _obvious_social_message(task.goal)

    @staticmethod
    def _complete_direct_response(task: "TaskState", context: "CoreLoopContext") -> None:
        fallback = "¡Hola! Qué gusto saludarte. ¿En qué puedo ayudarte?"
        message = fallback
        if context.inference is not None:
            prompt = (
                "Reply directly to this casual conversational message. Be warm, brief, natural, and use the user's language. "
                "Do not mention planning, tools, acceptance criteria, or internal processing. Do not claim any action occurred.\n\n"
                f"User message: {task.goal}"
            )
            try:
                result = context.inference.generate_markdown(InferenceRequest(
                    prompt=prompt,
                    purpose=InferencePurpose.FINAL_RESPONSE,
                    project_id=task.project_id,
                    user=task.user,
                    task_id=task.task_id,
                    tools=(),
                    cancel_checker=context.is_cancelled if context.cancellation_checker is not None else None,
                ))
            except Exception as exc:
                if context.is_cancelled():
                    raise
                task.metadata["v3_direct_response_fallback"] = {"error_type": type(exc).__name__}
            else:
                message = str(result.content or "").strip() or fallback
        task.metadata["prepared_user_response"] = {"source": "v3_direct_response", "message": message}
        task.metadata["v3_route"] = "respond_and_end"
        task.status = "completed"
        task.outcome = {"status": "success", "reason": "The conversational request was answered directly without tool execution."}

    @staticmethod
    def _persist(task: "TaskState", context: "CoreLoopContext") -> None:
        if context.question_store is not None:
            context.question_store.save_task_checkpoint(task, status=task.status)

    @staticmethod
    def _result(task: "TaskState", context: "CoreLoopContext") -> ProcessingResult:
        status = (
            ProcessingStatus.PARKED if task.status == "waiting_user" else
            ProcessingStatus.FAILED if task.status == "failed" else
            ProcessingStatus.CANCELLED if task.status == "cancelled" else ProcessingStatus.COMPLETED
        )
        context.emit_ui_event("run_finished", {"task": task.to_dict(), "status": task.status, "engine": "hierarchical_v3"})
        return ProcessingResult(
            snapshot=StateSnapshot(
                phase=ImprovementPhase.ACT,
                task_owner=task.user,
                current_work=task.goal,
                metadata={
                    "engine": "hierarchical_v3",
                    "status": task.status,
                    "outcome": task.outcome,
                    "phase_progress": task.hierarchical_state,
                    "task_state": task.to_dict(),
                },
            ),
            status=status,
            error=(
                f"v3_task_failed:{str((task.outcome or {}).get('reason') or 'deterministic_failure')}"
                if status == ProcessingStatus.FAILED else None
            ),
        )


class EngineRoutingProcessor:
    """Delegate to the engine stamped on the queued task."""

    def __init__(self, *, v2: "IntelligenceProcessor", v3: "IntelligenceProcessor") -> None:
        self.v2 = v2
        self.v3 = v3

    def process(self, task: "TaskState", context: "CoreLoopContext") -> ProcessingResult:
        return (self.v3 if task.intelligence_engine == "hierarchical_v3" else self.v2).process(task, context)


_SOCIAL_MESSAGE = re.compile(
    r"^\s*(?:[¡!¿?.,]\s*)*(?:hola|hello|hi|hey|buen(?:os\s+d[ií]as|as\s+tardes|as\s+noches)|"
    r"gracias|thank\s+you|thanks|adi[oó]s|bye)(?:\s+(?:alphonse|alfonse))?[!¡?.\s]*$",
    re.IGNORECASE,
)


def _obvious_social_message(value: str) -> bool:
    return bool(_SOCIAL_MESSAGE.fullmatch(str(value or "")))
