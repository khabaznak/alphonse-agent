"""Versioned source-linked memory envelopes; authorization precedes relevance."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, Callable

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class AccessScope:
    owner_user_id: str
    visibility: Literal["private", "shared"] = "private"
    shared_with: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.owner_user_id, str) or not self.owner_user_id.strip()
                or self.visibility not in {"private", "shared"}
                or not isinstance(self.shared_with, tuple)
                or any(not isinstance(user, str) or not user.strip() for user in self.shared_with)):
            raise ValueError("invalid_memory_access_scope")
        if self.visibility == "private" and self.shared_with:
            raise ValueError("private_memory_cannot_have_recipients")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "AccessScope":
        recipients = value.get("shared_with", ())
        if not isinstance(recipients, (list, tuple)):
            raise ValueError("invalid_memory_recipients")
        return cls(**{**value, "shared_with": tuple(recipients)})

    def allows(self, requester: str) -> bool:
        return bool(requester) and (requester == self.owner_user_id or
                                   (self.visibility == "shared" and requester in self.shared_with))


@dataclass(frozen=True)
class SourceReference:
    source_id: str
    kind: Literal["conversation", "task_evidence", "legacy_ledger"]
    access: AccessScope
    sequence: int | None = None
    speaker_id: str | None = None
    role: str | None = None
    channel: str | None = None
    recorded_at: str | None = None
    observed_at: str | None = None
    provenance: Literal["recorded", "legacy_import", "legacy_unverified"] = "recorded"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.source_id:
            raise ValueError("unsupported_memory_reference")
        if self.kind not in {"conversation", "task_evidence", "legacy_ledger"}:
            raise ValueError("invalid_memory_source_kind")
        if self.provenance not in {"recorded", "legacy_import", "legacy_unverified"}:
            raise ValueError("invalid_memory_provenance")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceReference":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported_memory_schema_version")
        return cls(**{**value, "access": AccessScope.from_dict(value["access"])})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ContextItem:
    source: SourceReference
    content: str
    representation: Literal["message", "audit", "working_state", "derived_knowledge"] = "message"
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.representation not in {
            "message", "audit", "working_state", "derived_knowledge"
        }:
            raise ValueError("unsupported_context_item")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ContextItem":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported_memory_schema_version")
        return cls(**{**value, "source": SourceReference.from_dict(value["source"])})

    def visible_to(self, requester: str) -> bool:
        return self.schema_version == SCHEMA_VERSION and self.source.access.allows(requester)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DurableMemoryRecord:
    """Derived knowledge, never a replacement for its underlying evidence."""
    memory_id: str
    content: str
    access: AccessScope
    sources: tuple[SourceReference, ...]
    recorded_at: str
    observed_at: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    supersedes: str | None = None
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or not self.memory_id or not self.sources:
            raise ValueError("source_linked_memory_required")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DurableMemoryRecord":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported_memory_schema_version")
        return cls(**{**value, "access": AccessScope.from_dict(value["access"]),
                      "sources": tuple(SourceReference.from_dict(source) for source in value["sources"])})

    def visible_to(self, requester: str) -> bool:
        # A derived record cannot widen the audience of any supporting source.
        return self.access.allows(requester) and all(s.access.allows(requester) for s in self.sources)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecallResult:
    query: str
    status: Literal["found", "empty", "unavailable"]
    items: tuple[ContextItem, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION or self.status not in {"found", "empty", "unavailable"}:
            raise ValueError("invalid_recall_result")
        if (self.status == "found") != bool(self.items):
            raise ValueError("inconsistent_recall_result")

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecallResult":
        if value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported_memory_schema_version")
        return cls(**{**value, "items": tuple(ContextItem.from_dict(item) for item in value.get("items", ()))})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def visible_unique(items: list[ContextItem], requester: str) -> list[ContextItem]:
    """Filter before deduplication, truncation, curation, or model submission."""
    seen: set[str] = set()
    result = []
    for item in items:
        if item.visible_to(requester) and item.source.source_id not in seen:
            seen.add(item.source.source_id)
            result.append(item)
    return result


def conversation_item(event: Any, requester: str, *, max_chars: int | None = None) -> ContextItem | None:
    """Fail closed for unversioned/malformed objects from alternate backends."""
    try:
        if event.schema_version != SCHEMA_VERSION:
            return None
        if not isinstance(event.shared_with, (list, tuple)):
            return None
        scope = AccessScope(event.owner_user_id, event.visibility, tuple(event.shared_with))
        if not scope.allows(requester):
            return None
        source = SourceReference(
            source_id=f"conversation:{event.event_id}", kind="conversation", access=scope,
            sequence=event.sequence, speaker_id=event.speaker_id or None, role=event.role,
            channel=event.source if event.provenance == "recorded" else None,
            recorded_at=event.recorded_at or None, observed_at=event.observed_at or None,
            provenance=event.provenance,
        )
        return ContextItem(source, event.content if max_chars is None else event.content[:max_chars])
    except (AttributeError, TypeError, ValueError):
        return None


def resolve_durable_memory(
    record: DurableMemoryRecord,
    requester: str,
    resolver: Callable[[SourceReference, str], ContextItem | None],
) -> DurableMemoryRecord | None:
    """Recheck live source access/existence before exposing derived knowledge.

    A cached source envelope is provenance, never an authorization credential.
    Missing, deleted and denied sources all withhold the derived record.
    """
    if not record.access.allows(requester):
        return None
    sources = []
    for source in record.sources:
        current = resolver(source, requester)
        if current is None or current.source.source_id != source.source_id or not current.visible_to(requester):
            return None
        sources.append(current.source)
    return replace(record, sources=tuple(sources))
