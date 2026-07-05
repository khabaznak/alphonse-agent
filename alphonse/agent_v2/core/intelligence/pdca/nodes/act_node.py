"""Act node for the v2 PDCA graph."""

from __future__ import annotations

from alphonse.agent_v2.core.intelligence.task_state import TaskState


def act_node(task: TaskState) -> TaskState:
    """Apply learning or adjustments from the current PDCA pass."""
    return task
