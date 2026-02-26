from __future__ import annotations

from typing import Any

from alphonse.agent.cortex.task_mode.state import build_default_task_state
from alphonse.agent.cortex.task_mode.types import TraceEvent


def task_state_with_defaults(state: dict[str, Any]) -> dict[str, Any]:
    existing = state.get("task_state")
    task_state = dict(existing) if isinstance(existing, dict) else {}
    defaults = build_default_task_state()
    for key, value in defaults.items():
        if key not in task_state:
            task_state[key] = value
    task_plan(task_state)
    task_trace(task_state)
    task_state.setdefault("facts", {})
    task_state.setdefault("status", "running")
    task_state.setdefault("repair_attempts", 0)
    task_state.setdefault("acceptance_criteria", [])
    return task_state


def has_acceptance_criteria(task_state: dict[str, Any]) -> bool:
    return bool(normalize_acceptance_criteria_values(task_state.get("acceptance_criteria")))


def normalize_acceptance_criteria_values(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        text = str(item or "").strip()
        if not text:
            continue
        out.append(text[:180])
        if len(out) >= 8:
            break
    return out


def task_plan(task_state: dict[str, Any]) -> dict[str, Any]:
    plan = task_state.get("plan")
    if not isinstance(plan, dict):
        plan = {"version": 1, "steps": [], "current_step_id": None}
        task_state["plan"] = plan
    if not isinstance(plan.get("steps"), list):
        plan["steps"] = []
    if "version" not in plan:
        plan["version"] = 1
    if "current_step_id" not in plan:
        plan["current_step_id"] = None
    return plan


def task_trace(task_state: dict[str, Any]) -> dict[str, Any]:
    trace = task_state.get("trace")
    if not isinstance(trace, dict):
        trace = {"summary": "", "recent": []}
        task_state["trace"] = trace
    if not isinstance(trace.get("recent"), list):
        trace["recent"] = []
    if "summary" not in trace:
        trace["summary"] = ""
    return trace


def append_trace_event(task_state: dict[str, Any], event: TraceEvent) -> None:
    trace = task_trace(task_state)
    recent = trace["recent"]
    recent.append(
        {
            "type": str(event.get("type") or "event"),
            "summary": str(event.get("summary") or "").strip()[:180],
            "correlation_id": event.get("correlation_id"),
        }
    )
    trace["recent"] = recent[-25:]


def next_step_id(task_state: dict[str, Any]) -> str:
    steps = task_plan(task_state).get("steps")
    index = len(steps) + 1 if isinstance(steps, list) else 1
    return f"step_{index}"


def current_step(task_state: dict[str, Any]) -> dict[str, Any] | None:
    plan = task_plan(task_state)
    current_id = str(plan.get("current_step_id") or "").strip()
    steps = plan.get("steps")
    if not current_id or not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, dict):
            continue
        if str(step.get("step_id") or "") == current_id:
            return step
    return None


def correlation_id(state: dict[str, Any]) -> str | None:
    value = state.get("correlation_id")
    if value is None:
        return None
    return str(value)
