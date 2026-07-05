"""Basic placeholder intelligence processor for Alphonse v2."""

from __future__ import annotations

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.core import ImprovementPhase
from alphonse.agent_v2.core.core import ProcessingResult
from alphonse.agent_v2.core.core import StateSnapshot
from alphonse.agent_v2.core.intelligence.task_state import TaskState


class BasicIntelligenceProcessor:
    """Small processor used by the first native TUI."""

    def process(self, task: TaskState, context: CoreLoopContext) -> ProcessingResult:
        _ = context
        response = _response_for_task(task)
        return ProcessingResult(
            snapshot=StateSnapshot(
                phase=ImprovementPhase.ACT,
                task_owner=task.user,
                current_work=task.goal,
                metadata={"response": response},
            )
        )


def _response_for_task(task: TaskState) -> str:
    is_command = bool(task.metadata.get("is_command"))
    if is_command:
        command = str(task.metadata.get("command") or "")
        args = str(task.metadata.get("command_args") or "")
        suffix = f" {args}" if args else ""
        return f"Command /{command}{suffix} detected. Command execution is not implemented yet."
    return f"I received your message: {task.goal}"
