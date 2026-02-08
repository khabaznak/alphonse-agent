from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any


class PendingInteractionType(str, Enum):
    SLOT_FILL = "SLOT_FILL"
    CONFIRMATION = "CONFIRMATION"
    OPTION_SELECTION = "OPTION_SELECTION"


@dataclass(frozen=True)
class PendingInteraction:
    type: PendingInteractionType
    key: str
    context: dict[str, Any]
    created_at: str
    expires_at: str | None = None


@dataclass(frozen=True)
class PendingResolution:
    consumed: bool
    result: dict[str, Any] | None = None
    error: str | None = None


def build_pending_interaction(
    interaction_type: PendingInteractionType,
    key: str,
    context: dict[str, Any] | None = None,
    ttl_minutes: int | None = 5,
) -> PendingInteraction:
    now = _now()
    expires_at = None
    if ttl_minutes is not None:
        expires_at = (datetime.now(timezone.utc) + timedelta(minutes=ttl_minutes)).isoformat()
    return PendingInteraction(
        type=interaction_type,
        key=key,
        context=context or {},
        created_at=now,
        expires_at=expires_at,
    )


def serialize_pending_interaction(pending: PendingInteraction) -> dict[str, Any]:
    return {
        "type": pending.type.value,
        "key": pending.key,
        "context": pending.context,
        "created_at": pending.created_at,
        "expires_at": pending.expires_at,
    }


def is_expired(pending: PendingInteraction) -> bool:
    if not pending.expires_at:
        return False
    try:
        expires = datetime.fromisoformat(pending.expires_at)
    except ValueError:
        return False
    return datetime.now(timezone.utc) >= expires


def try_consume(
    message_text: str,
    pending: PendingInteraction,
) -> PendingResolution:
    if is_expired(pending):
        return PendingResolution(consumed=False, error="expired")
    text = str(message_text or "").strip()
    if pending.type == PendingInteractionType.SLOT_FILL:
        return _consume_slot_fill(text, pending)
    if pending.type == PendingInteractionType.CONFIRMATION:
        return _consume_confirmation(text)
    if pending.type == PendingInteractionType.OPTION_SELECTION:
        return _consume_option_selection(text, pending)
    return PendingResolution(consumed=False)


def _consume_slot_fill(text: str, pending: PendingInteraction) -> PendingResolution:
    if pending.key == "user_name":
        if not text:
            return PendingResolution(consumed=False, error="empty")
        if len(text) > 40:
            return PendingResolution(consumed=False, error="too_long")
        return PendingResolution(consumed=True, result={"user_name": text})
    if pending.key == "address_text":
        if not text:
            return PendingResolution(consumed=False, error="empty")
        if len(text) < 5:
            return PendingResolution(consumed=False, error="too_short")
        if len(text) > 200:
            return PendingResolution(consumed=False, error="too_long")
        return PendingResolution(consumed=True, result={"address_text": text})
    return PendingResolution(consumed=False)


def _consume_confirmation(text: str) -> PendingResolution:
    normalized = text.lower().strip()
    yes = {"sí", "si", "yes", "y", "ok", "dale", "va"}
    no = {"no", "n", "nel", "nope"}
    if normalized in yes:
        return PendingResolution(consumed=True, result={"confirmed": True})
    if normalized in no:
        return PendingResolution(consumed=True, result={"confirmed": False})
    return PendingResolution(consumed=False)


def _consume_option_selection(text: str, pending: PendingInteraction) -> PendingResolution:
    choices = pending.context.get("choices") if isinstance(pending.context, dict) else None
    if not choices:
        return PendingResolution(consumed=False)
    normalized = text.strip()
    if normalized.isdigit():
        idx = int(normalized) - 1
        if 0 <= idx < len(choices):
            return PendingResolution(consumed=True, result={"selected": choices[idx]})
    return PendingResolution(consumed=False)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
