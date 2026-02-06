from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

from alphonse.agent.cognition.slots.resolvers import ParseResult, ResolverRegistry


@dataclass
class SlotMachine:
    slot_name: str
    slot_type: str
    state: str
    attempts: int
    last_input: str | None
    last_error: str | None
    value: Any | None
    confidence: float | None
    created_at: str
    expires_at: str | None
    paused_at: str | None
    context: dict[str, Any]


def create_machine(
    slot_name: str,
    slot_type: str,
    context: dict[str, Any],
    *,
    expires_at: str | None = None,
) -> SlotMachine:
    return SlotMachine(
        slot_name=slot_name,
        slot_type=slot_type,
        state="ASKED",
        attempts=0,
        last_input=None,
        last_error=None,
        value=None,
        confidence=None,
        created_at=datetime.now(timezone.utc).isoformat(),
        expires_at=expires_at,
        paused_at=None,
        context=context,
    )


def apply_input(
    machine: SlotMachine, text: str, registry: ResolverRegistry
) -> tuple[SlotMachine, ParseResult]:
    resolver = registry.get(machine.slot_type)
    machine.attempts += 1
    machine.last_input = text
    if not resolver:
        machine.state = "INVALID"
        machine.last_error = "missing_resolver"
        return machine, ParseResult(ok=False, error="missing_resolver")
    result = resolver.parse(text, machine.context)
    if result.ok:
        machine.state = "PARSED"
        machine.value = result.value
        machine.confidence = result.confidence
        machine.last_error = None
        return machine, result
    machine.state = "INVALID"
    machine.last_error = result.error
    return machine, result


def serialize_machine(machine: SlotMachine) -> dict[str, Any]:
    return asdict(machine)


def deserialize_machine(raw: dict[str, Any]) -> SlotMachine | None:
    if not isinstance(raw, dict):
        return None
    try:
        slot_name = str(raw.get("slot_name") or "")
        slot_type = str(raw.get("slot_type") or "")
        if not slot_name or not slot_type:
            return None
        return SlotMachine(
            slot_name=slot_name,
            slot_type=slot_type,
            state=str(raw.get("state") or ""),
            attempts=int(raw.get("attempts") or 0),
            last_input=raw.get("last_input"),
            last_error=raw.get("last_error"),
            value=raw.get("value"),
            confidence=raw.get("confidence"),
            created_at=str(raw.get("created_at") or ""),
            expires_at=raw.get("expires_at"),
            paused_at=raw.get("paused_at"),
            context=raw.get("context") or {},
        )
    except Exception:
        return None
