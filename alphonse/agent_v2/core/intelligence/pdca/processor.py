"""PDCA-backed intelligence processor for Alphonse v2."""

from __future__ import annotations

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.core import ProcessingResult
from alphonse.agent_v2.core.core import ProcessingStatus
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.intelligence.pdca.graph import run_pdca_once
from alphonse.agent_v2.core.intelligence.task_state import TaskState


class PDCAIntelligenceProcessor:
    """Runs a TaskState through the v2 PDCA graph."""

    def process(self, task: TaskState, context: CoreLoopContext) -> ProcessingResult:
        context.emit_ui_event("run_started", {"task": task.to_dict()})
        result = run_pdca_once(task, context=context)
        status = (
            ProcessingStatus.CANCELLED
            if str(result.status or "").strip().lower() == "cancelled"
            else ProcessingStatus.PARKED
            if str(result.status or "").strip().lower() == "waiting_user"
            else ProcessingStatus.COMPLETED
        )
        payload = {"task": result.to_dict(), "status": result.status}
        if status == ProcessingStatus.PARKED:
            context.emit_ui_event("state_snapshot", {"task_state": result.to_dict()})
            question = result.metadata.get("question_interrupt")
            if isinstance(question, dict):
                context.emit_ui_event("question_interrupt_opened", {"question": question})
                payload["question"] = question
        context.emit_ui_event("run_finished", payload)
        return ProcessingResult(
            snapshot=StateSnapshot(
                phase=ImprovementPhase.ACT,
                task_owner=result.user,
                current_work=result.goal,
                metadata={
                    "check_verdict": result.check_verdict,
                    "check_reason": result.check_reason,
                    "status": result.status,
                    "outcome": dict(result.outcome) if isinstance(result.outcome, dict) else None,
                    "act_route": result.metadata.get("act_route"),
                    "plan_json": result.plan_json,
                    "planned_tool_call": result.metadata.get("planned_tool_call"),
                    "question_interrupt": result.metadata.get("question_interrupt"),
                    "task_state": result.to_dict(),
                },
            ),
            status=status,
        )
