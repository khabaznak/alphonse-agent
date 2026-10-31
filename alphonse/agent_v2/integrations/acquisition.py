"""Read-only adapter contract. Implementations are trusted native provider code.

No implicit retries, URL-following or provider writes. Each operation must honor
its deadline and remaining limits at the transport (including decompression).
Content is untrusted evidence, never instructions or authority.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from alphonse.agent_v2.accounts import AccountSession, AccountHandle


@dataclass(frozen=True)
class ReadLimits:
    pages: int
    items: int
    bytes: int


@dataclass(frozen=True)
class ReadContext:
    session: AccountSession
    grant: AccountHandle
    deadline: datetime
    limits: ReadLimits


@dataclass(frozen=True)
class SourceRecord:
    resource: str  # Exact authorized container, e.g. folder/calendar/course ID.
    source_id: str
    version: str
    source_timestamp: str
    mime: bytes
    safe_url: str = ''
    deleted: bool = False


@dataclass(frozen=True)
class SourcePage:
    records: tuple[SourceRecord, ...] = ()
    cursor: str | None = None  # Opaque; never parsed, logged or sent to a model.
    complete: bool = False
    coverage_complete: bool = True
    bytes_read: int = 0  # All transport bytes, including unused response fields.


class CursorExpired(Exception):
    """Requires a new bounded baseline, never an unbounded full-account scan."""


class ReadAdapter(Protocol):
    version: str  # Changes invalidate opaque cursor state.
    def health(self, context: ReadContext) -> dict: ...
    def capabilities(self, context: ReadContext) -> tuple[str, ...]: ...
    def search(self, context: ReadContext, *, query: dict, since: datetime,
               cursor: str | None) -> SourcePage: ...
    def changes(self, context: ReadContext, *, query: dict, cursor: str) -> SourcePage: ...
    def read(self, context: ReadContext, *, resource: str, source_id: str) -> SourceRecord: ...
    def attachment(self, context: ReadContext, *, resource: str, source_id: str,
                   attachment_id: str, max_bytes: int) -> bytes: ...
