from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntentionMetadata:
    urgency: str | None = None
    audience: str | None = None
    semantics: str | None = None


@dataclass(frozen=True)
class Intention:
    key: str
    metadata: IntentionMetadata | None = None
