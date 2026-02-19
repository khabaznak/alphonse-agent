from __future__ import annotations

from typing import Any

from alphonse.agent.cognition.pending_interaction import PendingInteraction
from alphonse.agent.cognition.pending_interaction import PendingInteractionType
from alphonse.agent.cognition.pending_interaction import try_consume
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
    pending = _parse_pending_interaction(state.get("pending_interaction"))
    if (
        isinstance(pending, PendingInteraction)
        and pending.type == PendingInteractionType.SLOT_FILL
        and str(pending.key) == "acceptance_criteria"
        and isinstance(pending.context, dict)
        and str(pending.context.get("source") or "") == "task_mode.acceptance_criteria"
    ):
        resolution = try_consume(str(state.get("last_user_message") or ""), pending)
        criteria_text = str((resolution.result or {}).get("acceptance_criteria") or "").strip()
        if resolution.consumed and criteria_text:
            merged["acceptance_criteria"] = _normalize_acceptance_criteria(criteria_text)
            merged["status"] = "running"
            merged["next_user_question"] = None
            return {"task_state": merged, "pending_interaction": None}
    return {"task_state": merged}


def _parse_pending_interaction(raw: Any) -> PendingInteraction | None:
    if not isinstance(raw, dict):
        return None
    type_raw = str(raw.get("type") or "").strip().upper()
    key = str(raw.get("key") or "").strip()
    if not type_raw or not key:
        return None
    try:
        interaction_type = PendingInteractionType(type_raw)
    except Exception:
        return None
    context = raw.get("context") if isinstance(raw.get("context"), dict) else {}
    created_at = str(raw.get("created_at") or "").strip()
    if not created_at:
        return None
    expires_at = raw.get("expires_at")
    expires = str(expires_at).strip() if isinstance(expires_at, str) else None
    return PendingInteraction(
        type=interaction_type,
        key=key,
        context=context,
        created_at=created_at,
        expires_at=expires,
    )


def _normalize_acceptance_criteria(text: str) -> list[str]:
    lines = [item.strip(" -\t") for item in str(text or "").replace(";", "\n").splitlines()]
    out = [item for item in lines if item]
    return out[:8]
