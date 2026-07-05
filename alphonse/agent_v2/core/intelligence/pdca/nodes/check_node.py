"""Check node for the v2 PDCA graph."""

from __future__ import annotations

from alphonse.agent_v2.core.intelligence.task_state import TaskState


def check_node(task: TaskState) -> TaskState:
    """Review what the TaskState is about before planning.

    Detailed check semantics will be defined when the intelligence processor
    contract is expanded.
    """
    return task
