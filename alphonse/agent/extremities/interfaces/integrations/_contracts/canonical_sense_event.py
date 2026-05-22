from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CanonicalSenseEvent:
    """Provider-agnostic conscious handoff for non-message sensor events."""

    source_key: str
    provider: str
    event_id: str
    event_kind: str
    occurred_at: str
    subject: dict[str, Any]
    summary: str
    data: dict[str, Any] = field(default_factory=dict)
    raw_event: dict[str, Any] = field(default_factory=dict)
    dedupe_key: str | None = None
    contract_version: str = "1.0"

    def to_payload(self) -> dict[str, Any]:
        return {
            "contract_type": "canonical_sense_event",
            "contract_version": str(self.contract_version or "1.0").strip() or "1.0",
            "source_key": str(self.source_key or "").strip(),
            "provider": str(self.provider or "").strip(),
            "event_id": str(self.event_id or "").strip(),
            "event_kind": str(self.event_kind or "").strip(),
            "occurred_at": str(self.occurred_at or "").strip(),
            "subject": dict(self.subject),
            "summary": str(self.summary or "").strip(),
            "data": dict(self.data),
            "raw_event": dict(self.raw_event),
            "dedupe_key": str(self.dedupe_key or "").strip() or None,
        }
