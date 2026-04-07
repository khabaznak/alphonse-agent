from __future__ import annotations

from typing import Any

from alphonse.agent.cortex.graph import task_record_entry_node


def task_mode_entry_node(state: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper around TaskRecord-first entry hydration."""
    return task_record_entry_node(state)
