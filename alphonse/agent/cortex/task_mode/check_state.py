from __future__ import annotations

from typing import Any, Callable


def goal_satisfied(
    task_state: dict[str, Any],
    *,
    has_acceptance_criteria: Callable[[dict[str, Any]], bool],
) -> bool:
    outcome = task_state.get("outcome")
    if not isinstance(outcome, dict) or not outcome:
        return False
    kind = str(outcome.get("kind") or "").strip().lower()
    if kind == "task_completed":
        summary = str(
            outcome.get("final_text")
            or outcome.get("summary")
            or ""
        ).strip()
        if not summary or _looks_like_question(summary):
            return False
        return has_acceptance_criteria(task_state)
    return True


def derive_outcome_from_state(
    *,
    state: dict[str, Any],
    task_state: dict[str, Any],
    current_step: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> dict[str, Any] | None:
    current = current_step(task_state)
    if not isinstance(current, dict):
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None
    step_id = str(current.get("step_id") or "").strip()
    if not step_id:
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None
    facts = task_state.get("facts")
    if not isinstance(facts, dict):
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None
    entry = facts.get(step_id)
    if not isinstance(entry, dict):
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None

    tool_name = str(entry.get("tool") or "").strip()
    result = entry.get("result")
    if tool_name not in {"create_reminder", "createReminder"} or not isinstance(result, dict):
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None

    payload = result
    if str(result.get("status") or "").strip().lower() in {"ok", "executed"} and isinstance(
        result.get("result"), dict
    ):
        payload = dict(result.get("result") or {})

    reminder_id = str(payload.get("reminder_id") or "").strip()
    fire_at = str(payload.get("fire_at") or "").strip()
    message = str(payload.get("message") or "").strip()
    if not reminder_id or not fire_at:
        return task_state.get("outcome") if isinstance(task_state.get("outcome"), dict) else None

    for_whom = str(payload.get("for_whom") or state.get("channel_target") or "").strip()
    if not for_whom:
        for_whom = str(payload.get("delivery_target") or "").strip()
    return {
        "kind": "reminder_created",
        "evidence": {
            "reminder_id": reminder_id,
            "fire_at": fire_at,
            "message": message,
            "for_whom": for_whom,
        },
    }


def _looks_like_question(text: str) -> bool:
    rendered = str(text or "").strip().lower()
    if not rendered:
        return False
    if "?" in rendered:
        return True
    starters = (
        "can ",
        "could ",
        "would ",
        "should ",
        "do ",
        "did ",
        "is ",
        "are ",
        "what ",
        "when ",
        "where ",
        "why ",
        "how ",
    )
    return rendered.startswith(starters)
