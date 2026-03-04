from __future__ import annotations

from typing import Any, Callable


def goal_satisfied(
    task_state: dict[str, Any],
    *,
    has_acceptance_criteria: Callable[[dict[str, Any]], bool],
) -> bool:
    outcome = task_state.get("outcome")
    if isinstance(outcome, dict):
        kind = str(outcome.get("kind") or "").strip().lower()
        if kind in {"task_completed", "reminder_created", "message_delivered"} and has_acceptance_criteria(task_state):
            return True
    evaluation = evaluate_success_from_evidence(task_state=task_state)
    if not bool(evaluation.get("is_done", False)):
        return False
    if not has_acceptance_criteria(task_state):
        return False
    return True


def derive_outcome_from_state(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any] | None:
    _ = state
    _ = current_step
    return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None


def evaluate_success_from_evidence(task_state: dict[str, Any]) -> dict[str, Any]:
    goal = str(task_state.get("goal") or "").strip().lower()
    criteria = task_state.get("acceptance_criteria")
    criteria_list = [str(item).strip().lower() for item in criteria if str(item).strip()] if isinstance(criteria, list) else []
    facts = task_state.get("facts")
    fact_entries = [
        entry
        for entry in (facts or {}).values()
        if isinstance(entry, dict) and not bool(entry.get("internal"))
    ] if isinstance(facts, dict) else []
    latest = fact_entries[-1] if fact_entries else {}
    latest_tool = str(latest.get("tool") or "").strip()
    latest_status = str(latest.get("status") or "").strip().lower()

    if not fact_entries:
        return {
            "is_done": False,
            "reason": "no_evidence",
            "outcome_kind": "",
            "confidence": 0.0,
            "missing_evidence": ["No tool results were recorded yet."],
            "supporting_facts": [],
            "final_response_hint": "",
        }

    if _looks_like_job_count_task(goal=goal, criteria=criteria_list):
        count = _extract_jobs_count(fact_entries)
        if count is not None:
            return {
                "is_done": True,
                "reason": "jobs_count_obtained",
                "outcome_kind": "task_completed",
                "confidence": 0.98,
                "missing_evidence": [],
                "supporting_facts": [f"job_list count={count}"],
                "final_response_hint": f"Tenemos {count} jobs programados actualmente.",
            }

    if _looks_like_reminder_task(goal=goal, criteria=criteria_list):
        reminder_ok = _has_reminder_evidence(fact_entries)
        if reminder_ok:
            return {
                "is_done": True,
                "reason": "reminder_created",
                "outcome_kind": "reminder_created",
                "confidence": 0.95,
                "missing_evidence": [],
                "supporting_facts": ["reminder_id and fire_at present"],
                "final_response_hint": "Listo. El recordatorio quedó programado.",
            }

    if _looks_like_message_task(goal=goal, criteria=criteria_list):
        msg_ok = _has_message_delivery_evidence(fact_entries)
        if msg_ok:
            return {
                "is_done": True,
                "reason": "message_delivered",
                "outcome_kind": "task_completed",
                "confidence": 0.93,
                "missing_evidence": [],
                "supporting_facts": ["send_message status=ok"],
                "final_response_hint": "Listo, el mensaje fue enviado correctamente.",
            }

    if _has_tool_failure(fact_entries):
        return {
            "is_done": False,
            "reason": "latest_failure",
            "outcome_kind": "",
            "confidence": 0.2,
            "missing_evidence": ["Latest mission step failed."],
            "supporting_facts": [f"{latest_tool} status={latest_status or 'unknown'}"] if latest_tool else [],
            "final_response_hint": "",
        }

    if latest_tool and latest_status == "ok" and latest_tool not in {"send_message", "sendMessage"}:
        return {
            "is_done": False,
            "reason": "evidence_present_but_not_terminal",
            "outcome_kind": "",
            "confidence": 0.6,
            "missing_evidence": ["Need explicit completion evidence against acceptance criteria."],
            "supporting_facts": [f"{latest_tool} returned status ok"],
            "final_response_hint": "",
        }

    return {
        "is_done": False,
        "reason": "criteria_not_satisfied",
        "outcome_kind": "",
        "confidence": 0.25,
        "missing_evidence": ["Acceptance criteria are not fully satisfied yet."],
        "supporting_facts": [f"{latest_tool} status={latest_status or 'unknown'}"] if latest_tool else [],
        "final_response_hint": "",
    }


def _looks_like_job_count_task(*, goal: str, criteria: list[str]) -> bool:
    if "job" in goal and any(token in goal for token in {"count", "cuánt", "cuantos", "scheduled"}):
        return True
    blob = " ".join(criteria)
    return "job" in blob and any(token in blob for token in {"count", "conteo", "total"})


def _looks_like_reminder_task(*, goal: str, criteria: list[str]) -> bool:
    if any(token in goal for token in {"recordatorio", "reminder"}):
        return True
    return any("reminder" in item or "recordatorio" in item for item in criteria)


def _looks_like_message_task(*, goal: str, criteria: list[str]) -> bool:
    if any(token in goal for token in {"send message", "enviar mensaje", "notify", "notificar"}):
        return True
    return any("message" in item or "mensaje" in item for item in criteria)


def _extract_jobs_count(fact_entries: list[dict[str, Any]]) -> int | None:
    for entry in reversed(fact_entries):
        if str(entry.get("tool") or "").strip() != "job_list":
            continue
        payload = entry.get("result_payload")
        if not isinstance(payload, dict):
            result = entry.get("result")
            payload = result.get("result") if isinstance(result, dict) and isinstance(result.get("result"), dict) else {}
        if isinstance(payload, dict):
            jobs = payload.get("jobs")
            if isinstance(jobs, list):
                return len(jobs)
            data = payload.get("data")
            if isinstance(data, list):
                return len(data)
            count = payload.get("count")
            if isinstance(count, int):
                return count
            total = payload.get("total")
            if isinstance(total, int):
                return total
    return None


def _has_reminder_evidence(fact_entries: list[dict[str, Any]]) -> bool:
    for entry in reversed(fact_entries):
        tool = str(entry.get("tool") or "").strip()
        if tool not in {"create_reminder", "createReminder"}:
            continue
        payload = entry.get("result_payload")
        if isinstance(payload, dict) and str(payload.get("reminder_id") or "").strip() and str(payload.get("fire_at") or "").strip():
            return True
    return False


def _has_message_delivery_evidence(fact_entries: list[dict[str, Any]]) -> bool:
    for entry in reversed(fact_entries):
        tool = str(entry.get("tool") or "").strip()
        if tool not in {"send_message", "sendMessage"}:
            continue
        status = str(entry.get("status") or "").strip().lower()
        if status == "ok":
            return True
    return False


def _has_tool_failure(fact_entries: list[dict[str, Any]]) -> bool:
    latest = fact_entries[-1] if fact_entries else {}
    status = str((latest or {}).get("status") or "").strip().lower()
    return status in {"failed", "error"}
