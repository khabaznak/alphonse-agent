"""PDCA-backed intelligence processor for Alphonse v2."""

from __future__ import annotations

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.core import ProcessingResult
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.intelligence.pdca.graph import run_pdca_once
from alphonse.agent_v2.core.intelligence.task_state import TaskState


class PDCAIntelligenceProcessor:
    """Runs a TaskState through the v2 PDCA graph."""

    def process(self, task: TaskState, context: CoreLoopContext) -> ProcessingResult:
        result = run_pdca_once(task, context=context)
        return ProcessingResult(
            snapshot=StateSnapshot(
                phase=ImprovementPhase.ACT,
                task_owner=result.user,
                current_work=result.goal,
                metadata={
                    "check_verdict": result.check_verdict,
                    "check_reason": result.check_reason,
                    "plan_json": result.plan_json,
                    "planned_tool_call": result.metadata.get("planned_tool_call"),
                    "task_state": result.to_dict(),
                },
            )
        )
