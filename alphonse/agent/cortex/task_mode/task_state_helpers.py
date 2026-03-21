from __future__ import annotations

from typing import Any

from alphonse.agent.cortex.task_mode.state import build_default_task_state
from alphonse.agent.cortex.task_mode.types import AcceptanceCriterion
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
    task_state.setdefault("pending_plan_raw", None)
    task_state.setdefault("current_plan_step", None)
    task_state.setdefault("pending_control_tool_call", None)
    task_state.setdefault("success_evaluation_last", None)
    task_state.setdefault("completion_decision", None)
    task_state.setdefault("zero_progress_last_signature", None)
    task_state.setdefault("zero_progress_streak", 0)
    task_state.setdefault("planner_error_streak", 0)
    task_state.setdefault("planner_error_last", None)
    task_state.setdefault("planner_error_last_fact_key", None)
    task_state.setdefault("check_decision_last", None)
    task_state.setdefault("check_provenance", "entry")
    task_state.setdefault("judge_verdict", None)
    task_state.setdefault("judge_invalid_streak", 0)
    task_state.setdefault("steering_consumed_in_check", False)
    return task_state


def has_acceptance_criteria(task_state: dict[str, Any]) -> bool:
    return bool(normalize_acceptance_criteria_records(task_state.get("acceptance_criteria")))


def normalize_acceptance_criteria_values(value: Any) -> list[str]:
    return [item["text"] for item in normalize_acceptance_criteria_records(value)]


def normalize_acceptance_criteria_records(value: Any) -> list[AcceptanceCriterion]:
    if not isinstance(value, list):
        return []
    out: list[AcceptanceCriterion] = []
    for index, item in enumerate(value, start=1):
        if isinstance(item, dict):
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            criterion_id = str(item.get("id") or "").strip() or f"ac_{index}"
            status = str(item.get("status") or "pending").strip().lower()
            evidence_refs_raw = item.get("evidence_refs")
            evidence_refs = (
                [str(ref).strip() for ref in evidence_refs_raw if str(ref).strip()][:8]
                if isinstance(evidence_refs_raw, list)
                else []
            )
            created_by_case = str(item.get("created_by_case") or "").strip().lower() or "new_request"
            if created_by_case not in {"new_request", "execution_review", "task_resumption"}:
                created_by_case = "new_request"
            out.append(
                {
                    "id": criterion_id[:40],
                    "text": text[:180],
                    "status": "satisfied" if status == "satisfied" else "pending",
                    "evidence_refs": evidence_refs,
                    "created_by_case": created_by_case,
                }
            )
            if len(out) >= 16:
                break
            continue
        text = str(item or "").strip()
        if not text:
            continue
        out.append(
            {
                "id": f"ac_{index}",
                "text": text[:180],
                "status": "pending",
                "evidence_refs": [],
                "created_by_case": "new_request",
            }
        )
        if len(out) >= 16:
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
