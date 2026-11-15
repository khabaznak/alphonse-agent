"""Route queued tasks to the explicitly registered intelligence engine."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphonse.agent_v2.core.core import CoreLoopContext, IntelligenceProcessor, ProcessingResult
    from alphonse.agent_v2.core.intelligence.task_state import TaskState


class EngineRoutingProcessor:
    """Delegate to the engine stamped on the queued task."""

    def __init__(self, *, v3: "IntelligenceProcessor | None",
                 owner_read_candidate: "IntelligenceProcessor | None" = None) -> None:
        self.v3 = v3
        self.owner_read_candidate = owner_read_candidate

    def processor_for(self, task: "TaskState") -> "IntelligenceProcessor":
        """Validate a task's engine stamp before task-scoped context is loaded."""
        if (task.intelligence_engine == "hierarchical_v3" and task.intelligence_schema_version == 3
                and self.v3 is not None):
            return self.v3
        if (task.intelligence_engine == "single_loop_experimental" and task.intelligence_schema_version == 1
                and self.owner_read_candidate is not None):
            return self.owner_read_candidate
        # An unregistered engine must never silently execute under a different
        # processor, especially on queue retry or checkpoint resumption. The
        # core loop turns this into a failed processing result without running
        # the normal successful-task finalization path.
        raise ValueError("intelligence_engine_unregistered")

    def process(self, task: "TaskState", context: "CoreLoopContext") -> ProcessingResult:
        return self.processor_for(task).process(task, context)
