"""Plan node for the v2 PDCA graph."""

from __future__ import annotations

from alphonse.agent_v2.core.intelligence.task_state import TaskState


def plan_node(task: TaskState) -> TaskState:
    """Prepare the task plan for the current PDCA pass."""
    return task
