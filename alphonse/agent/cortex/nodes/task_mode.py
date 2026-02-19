from __future__ import annotations

import json
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
    if not goal or _looks_like_raw_payload_block(goal):
        merged["goal"] = _resolve_goal_text(state)
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


def _resolve_goal_text(state: dict[str, Any]) -> str:
    incoming = state.get("incoming_raw_message")
    if isinstance(incoming, dict):
        extracted = _extract_goal_from_payload(incoming)
        if extracted:
            return extracted
    return _extract_goal_from_packed_message(str(state.get("last_user_message") or ""))


def _extract_goal_from_payload(payload: dict[str, Any]) -> str:
    direct = str(payload.get("text") or "").strip()
    if direct:
        return direct
    message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
    nested = str(message.get("text") or "").strip()
    if nested:
        return nested
    provider_event = payload.get("provider_event") if isinstance(payload.get("provider_event"), dict) else {}
    provider_message = provider_event.get("message") if isinstance(provider_event.get("message"), dict) else {}
    provider_text = str(provider_message.get("text") or "").strip()
    if provider_text:
        return provider_text
    return ""


def _extract_goal_from_packed_message(last_user_message: str) -> str:
    rendered = str(last_user_message or "").strip()
    if not rendered:
        return ""
    marker = "```json"
    if marker in rendered:
        after = rendered.split(marker, 1)[1]
        json_payload = after.split("```", 1)[0].strip()
        try:
            parsed = json.loads(json_payload)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            extracted = _extract_goal_from_payload(parsed)
            if extracted:
                return extracted
    for line in rendered.splitlines():
        if line.lower().startswith("- text:"):
            candidate = line.split(":", 1)[1].strip()
            if candidate:
                return candidate
    return rendered[:240]


def _looks_like_raw_payload_block(text: str) -> bool:
    rendered = str(text or "").strip().lower()
    return rendered.startswith("## raw message")
