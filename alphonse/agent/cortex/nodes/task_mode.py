from __future__ import annotations

from typing import Any

from alphonse.agent.cortex.task_mode.state import build_default_task_state


def task_mode_entry_node(state: dict[str, Any]) -> dict[str, Any]:
    task_state = state.get("task_state")
    merged = dict(task_state) if isinstance(task_state, dict) else {}
    defaults = build_default_task_state()

    for key, value in defaults.items():
        if key not in merged:
            merged[key] = value

    goal = str(merged.get("goal") or "").strip()
    if not goal:
        merged["goal"] = str(state.get("last_user_message") or "").strip()
    return {"task_state": merged}
