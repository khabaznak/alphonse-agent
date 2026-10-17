"""Attributable, project-independent knowledge. Sources and policy outrank models."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import re
import sqlite3
from threading import RLock
from typing import Any, Callable

from alphonse.agent_v2.database import connect_database
from .contracts import AccessScope, ContextItem, DurableMemoryRecord, SourceReference, conversation_item

KNOWLEDGE_VERSION = 3
KINDS = {"fact", "preference", "instruction", "decision", "commitment", "observation"}
CANDIDATE_FIELDS = {
    "content", "kind", "subjects", "claim_key", "quote", "persistence", "intent_quote",
    "valid_from", "valid_until", "validity_quote", "supersedes", "conflicts_with", "relation_quote",
    "schedule_id", "task_id", "commitment_state", "state_quote", "supporting_memory_ids",
}


def instant(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("knowledge_time_requires_timezone")
    return parsed.astimezone(timezone.utc)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class KnowledgeRecord:
    memory_id: str
    content: str
    access: AccessScope
    sources: tuple[SourceReference, ...]
    recorded_at: str
    kind: str
    subjects: tuple[str, ...]
    claim_key: str
    basis: str
    quotes: tuple[str, ...]
    persistence: str = "contextual"
    status: str = "active"
    observed_at: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    supersedes: str | None = None
    conflicts_with: tuple[str, ...] = ()
    project_id: str = ""
    task_id: str | None = None
    schedule_id: str | None = None
    commitment_state: str | None = None
    dependencies: tuple[str, ...] = ()
    schema_version: int = KNOWLEDGE_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != KNOWLEDGE_VERSION:
            raise ValueError("unsupported_knowledge_version")
        if (not self.memory_id or not self.content.strip() or not self.sources or not self.subjects
                or not self.claim_key.strip() or self.kind not in KINDS
                or self.basis not in {"user_report", "tool_observation", "legacy_unverified"}
                or self.status not in {"active", "superseded", "disputed"}
                or self.persistence not in {"contextual", "explicit"}
                or len(self.sources) != len(self.quotes) or any(not q.strip() for q in self.quotes)):
            raise ValueError("invalid_knowledge_record")
        instant(self.recorded_at)
        for value in (self.observed_at, self.valid_from, self.valid_until):
            if value is not None:
                instant(value)
        if self.valid_from and self.valid_until and instant(self.valid_from) > instant(self.valid_until):
            raise ValueError("invalid_knowledge_validity")
        if self.kind in {"preference", "instruction"} and self.persistence != "explicit":
            raise ValueError("persistent_intent_required")
        if self.kind == "commitment" and self.commitment_state not in {"unresolved", "reported_completed", "reported_cancelled"}:
            raise ValueError("invalid_commitment_state")
        if self.kind != "commitment" and (self.commitment_state is not None or self.schedule_id is not None or self.task_id is not None):
            raise ValueError("commitment_links_require_commitment")
        if (not isinstance(self.dependencies, tuple) or len(self.dependencies) > 8
                or any(not isinstance(ref, str) or not ref.strip() for ref in self.dependencies)):
            raise ValueError("invalid_knowledge_dependencies")
        # The derived audience must be a subset of every supporting audience.
        audience = {self.access.owner_user_id, *self.access.shared_with}
        if any(not all(s.access.allows(user) for user in audience) for s in self.sources):
            raise ValueError("knowledge_audience_widening")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> KnowledgeRecord:
        if value.get("schema_version") != KNOWLEDGE_VERSION:
            raise ValueError("unsupported_knowledge_version")
        if not isinstance(value.get("dependencies"), (list, tuple)):
            raise ValueError("knowledge_dependencies_required")
        return cls(**{**value, "access": AccessScope.from_dict(value["access"]),
                      "sources": tuple(SourceReference.from_dict(s) for s in value["sources"]),
                      "subjects": tuple(value["subjects"]), "quotes": tuple(value["quotes"]),
                      "dependencies": tuple(value["dependencies"]),
                      "conflicts_with": tuple(value.get("conflicts_with", ()))})

    @staticmethod
    def legacy_envelope(record: DurableMemoryRecord) -> dict[str, Any]:
        """Explicit reconciliation input, not an automatic upgrade to knowledge v2."""
        return {"legacy_envelope": record.to_dict(), "basis": "legacy_unverified",
                "requires_reconciliation": True}


class KnowledgeStore:
    """Immutable payloads plus transactional lifecycle edges and processing receipts."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db_path = str(db_path)
        self._lock = RLock()
        self._memory = sqlite3.connect(":memory:", check_same_thread=False) if self.db_path == ":memory:" else None
        if self._memory is not None:
            self._memory.row_factory = sqlite3.Row
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS memory_knowledge (
                    memory_id TEXT PRIMARY KEY, owner TEXT NOT NULL, claim_key TEXT NOT NULL,
                    payload TEXT NOT NULL, status TEXT NOT NULL, conflicts TEXT NOT NULL);
                CREATE INDEX IF NOT EXISTS memory_knowledge_owner ON memory_knowledge(owner, claim_key);
                CREATE TABLE IF NOT EXISTS memory_processing (
                    owner TEXT NOT NULL, source_id TEXT NOT NULL, version INTEGER NOT NULL,
                    PRIMARY KEY(owner, source_id, version));
                CREATE TABLE IF NOT EXISTS memory_task_sources (
                    source_id TEXT PRIMARY KEY, owner TEXT NOT NULL, payload TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memory_task_state (
                    task_id TEXT PRIMARY KEY, owner TEXT NOT NULL, status TEXT NOT NULL, recorded_at TEXT NOT NULL);
            """)

    @contextmanager
    def _connect(self):
        with self._lock:
            conn = self._memory if self._memory is not None else connect_database(self.db_path)
            try:
                if conn.in_transaction:
                    # Source resolution can re-enter this store while a write
                    # transaction is open. A read must not commit its caller.
                    yield conn
                else:
                    with conn:
                        yield conn
            finally:
                if self._memory is None:
                    conn.close()

    def processed(self, owner: str, source_id: str) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM memory_processing WHERE owner=? AND source_id=? AND version=?",
                                (owner, source_id, KNOWLEDGE_VERSION)).fetchone() is not None

    def pending_events(self, conversations: Any, owner: str, *, limit: int = 16) -> list[Any]:
        # The conversation API is bounded; no full household history enters a prompt.
        cursor = 0
        with self._connect() as conn:
            row = conn.execute("SELECT MAX(CAST(substr(source_id, 10) AS INTEGER)) FROM memory_processing "
                               "WHERE owner=? AND source_id LIKE 'sequence:%' AND version=?",
                               (owner, KNOWLEDGE_VERSION)).fetchone()
            cursor = int(row[0] or 0)
        return conversations.list_after(owner_user_id=owner, after_sequence=cursor, limit=limit)

    def mark_sequence(self, owner: str, sequence: int) -> None:
        with self._connect() as conn:
            conn.execute("INSERT OR IGNORE INTO memory_processing VALUES (?,?,?)",
                         (owner, f"sequence:{sequence}", KNOWLEDGE_VERSION))

    def _record(self, row: sqlite3.Row) -> KnowledgeRecord:
        return replace(KnowledgeRecord.from_dict(json.loads(row["payload"])),
                       status=row["status"], conflicts_with=tuple(json.loads(row["conflicts"])))

    def _readable_record(self, row: sqlite3.Row) -> KnowledgeRecord | None:
        try:
            return self._record(row)
        except (ValueError, TypeError, KeyError):
            # Preserve incompatible payloads for explicit reconciliation; never
            # invent dependencies or let one old row disable supported records.
            return None

    def records(self, requester: str, resolver: Callable, *, historical: bool = False,
                query: str = "", limit: int = 32, at: str | None = None) -> list[KnowledgeRecord]:
        """Access and live source resolution precede ranking and result budgets."""
        terms = set(re.findall(r"[\w'-]+", query.casefold()))
        current_time = instant(at or now())
        results = []
        with self._connect() as conn:
            # Shared audiences can cross owners; project is deliberately not a predicate.
            rows = conn.execute("SELECT * FROM memory_knowledge ORDER BY rowid DESC").fetchall()
        indexed = {row["memory_id"]: row for row in rows}
        cache: dict[str, KnowledgeRecord | None] = {}

        def resolve_record(memory_id: str, visiting: frozenset[str] = frozenset()) -> KnowledgeRecord | None:
            if memory_id in visiting or memory_id not in indexed:
                return None
            if memory_id in cache:
                return cache[memory_id]
            record = self._readable_record(indexed[memory_id])
            if record is None:
                return None
            if not record.access.allows(requester):
                return None
            live = [resolver(source, requester) for source in record.sources]
            if any(item is None or item.source.source_id != source.source_id or not item.visible_to(requester)
                   for source, item in zip(record.sources, live)):
                return None
            # A stale envelope cannot widen current audiences in extraction context either.
            try:
                record = replace(record, sources=tuple(item.source for item in live))
            except ValueError:
                return None
            dependencies = [resolve_record(ref, visiting | {memory_id}) for ref in record.dependencies]
            if any(dependency is None for dependency in dependencies):
                return None
            if any(dependency.status == "disputed" for dependency in dependencies) and record.status != "superseded":
                record = replace(record, status="disputed")
            if not historical:
                if record.status == "superseded" or record.kind == "observation" or record.basis == "tool_observation":
                    return None
                if record.valid_from and current_time < instant(record.valid_from):
                    return None
                if record.valid_until and current_time >= instant(record.valid_until):
                    return None
            cache[memory_id] = record
            return record

        for row in rows:
            record = resolve_record(row["memory_id"])
            if record is None:
                continue
            score = len(terms & set(re.findall(r"[\w'-]+", (record.content + " " + " ".join(record.subjects)).casefold())))
            if terms and not score and record.persistence != "explicit":
                continue
            results.append((score, record))
        return [r for _, r in sorted(results, key=lambda x: x[0], reverse=True)[:limit]]

    def comparison_pairs(self, owner: str, records: list[KnowledgeRecord], resolver: Callable) -> list[tuple[KnowledgeRecord, KnowledgeRecord]]:
        """Authorized same-key pairs; a model judges semantics outside the write lock."""
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM memory_knowledge WHERE owner=? AND status!='superseded'", (owner,)).fetchall()
        prior = [record for row in rows if (record := self._readable_record(row)) is not None]
        indexed = {r.memory_id: r for r in prior}
        pairs = []
        for record in records:
            for old in prior:
                if (old.access == record.access and old.claim_key == record.claim_key and old.subjects == record.subjects
                        and old.memory_id != record.supersedes and old.memory_id not in record.conflicts_with
                        and old.content != record.content and self._sources_available(old, owner, resolver)
                        and self._dependencies_current(old, indexed.get, {record.supersedes})):
                    pairs.append((old, record))
                    if len(pairs) > 128:
                        raise ValueError("knowledge_comparison_budget_exceeded")
            prior.append(record)
            indexed[record.memory_id] = record
        return pairs

    @staticmethod
    def _dependencies_current(record: KnowledgeRecord, lookup: Callable, blocked: set, visited: frozenset = frozenset()) -> bool:
        if record.memory_id in visited:
            return False
        for dependency_id in record.dependencies:
            dependency = lookup(dependency_id)
            if dependency is None or dependency_id in blocked or dependency.status == "superseded":
                return False
            if dependency.valid_from and instant(dependency.valid_from) > instant(now()):
                return False
            if dependency.valid_until and instant(dependency.valid_until) <= instant(now()):
                return False
            if not KnowledgeStore._dependencies_current(dependency, lookup, blocked, visited | {record.memory_id}):
                return False
        return True

    def commit(self, owner: str, source_id: str, records: list[KnowledgeRecord], *, resolver: Callable | None = None,
               comparisons: dict[tuple[str, str], str] | None = None) -> bool:
        with self._connect() as conn:
            # Serialize receipt check and all transitions across processes as well as threads.
            conn.execute("BEGIN IMMEDIATE")
            def lookup(memory_id: str) -> KnowledgeRecord | None:
                row = conn.execute("SELECT * FROM memory_knowledge WHERE memory_id=?", (memory_id,)).fetchone()
                return self._readable_record(row) if row else None
            if conn.execute("SELECT 1 FROM memory_processing WHERE owner=? AND source_id=? AND version=?",
                            (owner, source_id, KNOWLEDGE_VERSION)).fetchone():
                return False
            for record in records:
                if record.access.owner_user_id != owner or not any(s.source_id == source_id for s in record.sources):
                    raise ValueError("knowledge_write_scope")
                if resolver is not None and not self._sources_available(record, owner, resolver):
                    raise ValueError("knowledge_support_no_longer_available")
                for dependency_id in record.dependencies:
                    row = conn.execute("SELECT * FROM memory_knowledge WHERE memory_id=?", (dependency_id,)).fetchone()
                    if row is None or dependency_id == record.memory_id:
                        raise ValueError("knowledge_dependency_missing")
                    dependency = self._record(row)
                    if (not dependency.access.allows(owner) or resolver is None
                            or not self._sources_available(dependency, owner, resolver)):
                        raise ValueError("knowledge_dependency_unavailable")
                    support = dict(zip((s.source_id for s in record.sources), record.quotes))
                    if any(support.get(s.source_id) != quote for s, quote in zip(dependency.sources, dependency.quotes)):
                        raise ValueError("knowledge_dependency_sources_required")
                # Exact textual identity needs no semantic call. Every other
                # same-key relation needs a reviewed judgment, including races.
                rows = conn.execute("SELECT * FROM memory_knowledge WHERE owner=? AND claim_key=? AND status!='superseded'",
                                    (owner, record.claim_key)).fetchall()
                conflicts = set(record.conflicts_with)
                for row in rows:
                    old = self._readable_record(row)
                    if old is None:
                        continue
                    if resolver is not None and not self._sources_available(old, owner, resolver):
                        continue
                    if not self._dependencies_current(old, lookup, {record.supersedes}):
                        continue
                    if (old.access == record.access and old.subjects == record.subjects
                            and old.memory_id != record.supersedes and old.memory_id not in conflicts and old.content != record.content):
                        relation = (comparisons or {}).get((old.memory_id, record.memory_id))
                        if relation not in {"equivalent", "compatible", "conflicting"}:
                            raise ValueError("knowledge_comparison_required")
                        if relation == "conflicting":
                            conflicts.add(old.memory_id)
                if conflicts:
                    record = replace(record, conflicts_with=tuple(sorted(conflicts)), status="disputed")
                for related_id in (*record.conflicts_with, *((record.supersedes,) if record.supersedes else ())):
                    row = conn.execute("SELECT * FROM memory_knowledge WHERE memory_id=?", (related_id,)).fetchone()
                    if row is None:
                        raise ValueError("knowledge_relation_missing")
                    old = self._record(row)
                    if resolver is not None and not self._sources_available(old, owner, resolver):
                        raise ValueError("knowledge_relation_source_unavailable")
                    if old.access != record.access or old.claim_key != record.claim_key or old.subjects != record.subjects:
                        raise ValueError("knowledge_relation_scope")
                    if old.status == "superseded":
                        raise ValueError("knowledge_relation_not_current")
                    if record.supersedes == related_id:
                        # A report cannot retract another speaker's claim, including shared sources.
                        if (record.basis != "user_report"
                                or old.sources[0].speaker_id != owner or record.sources[0].speaker_id != owner):
                            raise ValueError("knowledge_correction_requires_original_reporter")
                        if (record.sources[0].sequence is None or old.sources[0].sequence is None
                                or record.sources[0].sequence <= old.sources[0].sequence):
                            raise ValueError("knowledge_correction_must_follow_original")
                        conn.execute("UPDATE memory_knowledge SET status='superseded' WHERE memory_id=?", (related_id,))
                    else:
                        links = sorted(set(old.conflicts_with) | {record.memory_id})
                        conn.execute("UPDATE memory_knowledge SET status='disputed', conflicts=? WHERE memory_id=?",
                                     (json.dumps(links), related_id))
                conn.execute("INSERT INTO memory_knowledge VALUES (?,?,?,?,?,?)",
                             (record.memory_id, owner, record.claim_key, json.dumps(record.to_dict()),
                              record.status, json.dumps(record.conflicts_with)))
            conn.execute("INSERT INTO memory_processing VALUES (?,?,?)", (owner, source_id, KNOWLEDGE_VERSION))
        return True

    @staticmethod
    def _sources_available(record: KnowledgeRecord, requester: str, resolver: Callable) -> bool:
        for source in record.sources:
            item = resolver(source, requester)
            if item is None or item.source.source_id != source.source_id or not item.visible_to(requester):
                return False
        return True

    def capture_task(self, task: Any) -> list[ContextItem]:
        owner = str(task.user or "")
        items = task.evidence_items(owner) if callable(getattr(task, "evidence_items", None)) else []
        with self._connect() as conn:
            for item in items:
                existing = conn.execute("SELECT payload FROM memory_task_sources WHERE source_id=?", (item.source.source_id,)).fetchone()
                payload = json.dumps(item.to_dict(), sort_keys=True)
                if existing and existing[0] != payload:
                    raise ValueError("task_evidence_identity_changed")
                conn.execute("INSERT OR IGNORE INTO memory_task_sources VALUES (?,?,?)", (item.source.source_id, owner, payload))
            task_id = str(getattr(task, "task_id", "") or getattr(task, "message_id", ""))
            if owner and task_id:
                conn.execute("INSERT INTO memory_task_state VALUES (?,?,?,?) ON CONFLICT(task_id) DO UPDATE SET "
                             "status=excluded.status, recorded_at=excluded.recorded_at WHERE owner=excluded.owner",
                             (task_id, owner, str(task.status), now()))
        return items

    def task_source(self, reference: SourceReference, requester: str) -> ContextItem | None:
        with self._connect() as conn:
            row = conn.execute("SELECT payload FROM memory_task_sources WHERE source_id=? AND owner=?",
                               (reference.source_id, requester)).fetchone()
        return ContextItem.from_dict(json.loads(row[0])) if row else None

    def task_status(self, task_id: str, requester: str) -> dict[str, str] | None:
        with self._connect() as conn:
            row = conn.execute("SELECT status, recorded_at FROM memory_task_state WHERE task_id=? AND owner=?",
                               (task_id, requester)).fetchone()
        return dict(row) if row else None


class KnowledgeMemory:
    """System Two extracts; Jev judges bounded candidates; deterministic policy writes."""

    def __init__(self, *, store: KnowledgeStore, conversations: Any, extractor: Callable,
                 reviewer: Callable, schedule_resolver: Callable | None = None, comparison_reviewer: Callable | None = None,
                 support_selector: Callable | None = None) -> None:
        self.store, self.conversations = store, conversations
        self.extractor, self.reviewer = extractor, reviewer
        self.schedule_resolver = schedule_resolver
        self.comparison_reviewer = comparison_reviewer
        self.support_selector = support_selector

    def resolve(self, reference: SourceReference, requester: str) -> ContextItem | None:
        if reference.kind == "conversation" and reference.source_id.startswith("conversation:"):
            event = self.conversations.read(event_id=reference.source_id.removeprefix("conversation:"), requester_user_id=requester)
            return conversation_item(event, requester) if event else None
        if reference.kind == "task_evidence":
            return self.store.task_source(reference, requester)
        return None

    def process_pending(self, owner: str, *, cancel_checker: Callable | None = None) -> None:
        # Prioritize today's new messages, then make bounded progress through the backlog.
        events = self.conversations.list_recent(owner_user_id=owner, limit=16)
        errors = []
        attempted = set()
        for event in events:
            attempted.add(event.event_id)
            try:
                self.process_event(event, owner, cancel_checker=cancel_checker)
            except Exception as exc:
                errors.append(type(exc).__name__)
        for event in self.store.pending_events(self.conversations, owner):
            if event.event_id not in attempted:
                try:
                    self.process_event(event, owner, cancel_checker=cancel_checker)
                except Exception as exc:
                    errors.append(type(exc).__name__)
                    break
            item = conversation_item(event, owner)
            if item is None or not self.store.processed(owner, item.source.source_id):
                break
            self.store.mark_sequence(owner, event.sequence)
        if errors:
            raise RuntimeError(f"knowledge_updates_pending:{len(errors)}")

    def process_event(self, event: Any, owner: str, *, cancel_checker: Callable | None = None) -> None:
        self._check_cancelled(cancel_checker)
        item = conversation_item(event, owner)
        if item is None:
            return
        if item.source.role != "user" or item.source.provenance != "recorded":
            self.store.commit(owner, item.source.source_id, [])
            return
        self.process(item, owner, project_id=event.project_id, cancel_checker=cancel_checker)

    def process_task(self, task: Any, *, cancel_checker: Callable | None = None) -> None:
        self._check_cancelled(cancel_checker)
        owner = str(task.user or "")
        errors = []
        for item in self.store.capture_task(task):
            try:
                self.process(item, owner, project_id=str(task.project_id or ""), cancel_checker=cancel_checker)
            except Exception as exc:
                errors.append(type(exc).__name__)
        try:
            self.process_pending(owner, cancel_checker=cancel_checker)
        except Exception as exc:
            errors.append(type(exc).__name__)
        if errors:
            raise RuntimeError(f"knowledge_updates_pending:{len(errors)}")

    @staticmethod
    def _check_cancelled(checker: Callable | None) -> None:
        if checker and checker():
            raise RuntimeError("knowledge_update_cancelled")

    def process(self, item: ContextItem, owner: str, *, project_id: str = "", cancel_checker: Callable | None = None) -> None:
        self._check_cancelled(cancel_checker)
        source = item.source
        if source.access.owner_user_id != owner or not item.visible_to(owner):
            raise ValueError("knowledge_source_owner_required")
        if self.store.processed(owner, source.source_id):
            return
        if not (source.kind == "task_evidence" or (source.kind == "conversation" and source.role == "user" and source.provenance == "recorded")):
            raise ValueError("knowledge_requires_user_or_tool_evidence")
        matched = self.store.records(owner, self.resolve, historical=True, query=item.content, limit=24)
        recent = self.store.records(owner, self.resolve, historical=True, limit=8)
        existing = []
        used = 0
        for record in {r.memory_id: r for r in matched + recent}.values():
            size = len(json.dumps(record.to_dict()))
            if used + size <= 16000:
                existing.append(record)
                used += size
        raw, approved = [], []
        # Long messages are processed in bounded overlapping chunks under the
        # same canonical source identity. The receipt commits only after all
        # chunks succeed; retrying never promotes a partial extraction.
        for offset in range(0, len(item.content), 11000):
            self._check_cancelled(cancel_checker)
            chunk = item.content[offset:offset + 12000]
            supporting_ids = self.support_selector(ContextItem(source, chunk, item.representation), existing) if existing and self.support_selector else []
            if existing and self.support_selector is None:
                raise ValueError("knowledge_support_selection_unavailable")
            if (not isinstance(supporting_ids, list) or len(supporting_ids) > 8
                    or any(ref not in {r.memory_id for r in existing} for ref in supporting_ids)):
                raise ValueError("knowledge_dependency_not_visible")
            payload = {"source": ContextItem(source, chunk, item.representation).to_dict(),
                       # Relation metadata is not evidence for a synthesized fact.
                       "existing": [{"memory_id": r.memory_id, "claim_key": r.claim_key,
                                     "subjects": r.subjects, "status": r.status, "kind": r.kind} for r in existing],
                       "supporting_knowledge": [r.to_dict() for r in existing if r.memory_id in supporting_ids]}
            candidates = self.extractor(payload, owner, cancel_checker=cancel_checker)
            if not isinstance(candidates, dict) or set(candidates) != {"candidates"} or not isinstance(candidates["candidates"], list) or len(candidates["candidates"]) > 8:
                raise ValueError("invalid_knowledge_extraction")
            batch = candidates["candidates"]
            if len(json.dumps(batch, allow_nan=False)) > 32000:
                raise ValueError("knowledge_candidate_budget_exceeded")
            normalized_batch = []
            for candidate in batch:
                if not isinstance(candidate, dict) or set(candidate) - CANDIDATE_FIELDS:
                    raise ValueError("unknown_knowledge_candidate_field")
                quote = candidate.get("quote")
                if not isinstance(quote, str) or not quote.strip() or len(quote) > 2000 or quote not in chunk:
                    raise ValueError("knowledge_quote_required")
                dependencies = candidate.get("supporting_memory_ids", [])
                if (not isinstance(dependencies, list) or len(dependencies) > 8
                        or any(not isinstance(ref, str) or ref not in supporting_ids for ref in dependencies)):
                    raise ValueError("knowledge_dependency_not_visible")
                # A generator cannot omit or under-declare factual context that
                # was exposed to it. Carry all selected support, transitively.
                normalized_batch.append({**candidate, "supporting_memory_ids": list(dict.fromkeys(supporting_ids))})
            batch = normalized_batch
            # The semantic review can reject extraction; it cannot bypass write policy.
            decisions = self.reviewer(payload, batch) if batch else []
            if len(decisions) != len(batch) or any(type(flag) is not bool for flag in decisions):
                raise ValueError("invalid_knowledge_review")
            raw.extend(batch)
            approved.extend(decisions)
        related = {r.memory_id: r for r in existing}
        records = []
        for candidate, accept in zip(raw, approved):
            if not accept:
                continue
            if not isinstance(candidate, dict):
                raise ValueError("invalid_knowledge_candidate")
            if set(candidate) - CANDIDATE_FIELDS:
                raise ValueError("unknown_knowledge_candidate_field")
            quote = candidate.get("quote")
            if not isinstance(quote, str) or not quote.strip() or len(quote) > 2000 or quote not in item.content:
                raise ValueError("knowledge_quote_required")
            kind, persistence = candidate.get("kind"), candidate.get("persistence", "contextual")
            if kind in {"preference", "instruction"}:
                intent = candidate.get("intent_quote")
                if persistence != "explicit" or not isinstance(intent, str) or not intent.strip() or intent not in item.content:
                    raise ValueError("knowledge_persistent_intent_required")
            basis = "user_report" if source.kind == "conversation" and source.role == "user" and source.provenance == "recorded" else "tool_observation"
            if basis == "tool_observation":
                # Audit results are observations of an attempt, never verified durable external truth.
                kind, persistence = "observation", "contextual"
            if candidate.get("valid_from") or candidate.get("valid_until"):
                validity = candidate.get("validity_quote")
                if not isinstance(validity, str) or not validity.strip() or validity not in item.content:
                    raise ValueError("knowledge_validity_quote_required")
            supersedes = candidate.get("supersedes")
            conflicts = tuple(candidate.get("conflicts_with", ()))
            if supersedes or conflicts:
                relation = candidate.get("relation_quote")
                if not isinstance(relation, str) or not relation.strip() or relation not in item.content:
                    raise ValueError("knowledge_relation_quote_required")
                if any(ref not in related for ref in (*conflicts, *((supersedes,) if supersedes else ()))):
                    raise ValueError("knowledge_relation_not_visible")
            subjects = candidate.get("subjects")
            if not isinstance(subjects, list) or not subjects or len(subjects) > 8 or any(not isinstance(s, str) or not s.strip() for s in subjects):
                raise ValueError("knowledge_subjects_required")
            content, key = candidate.get("content"), candidate.get("claim_key")
            if not isinstance(content, str) or len(content) > 2000 or not isinstance(key, str) or len(key) > 200:
                raise ValueError("invalid_knowledge_content")
            # New writes default private. A model never decides to share household information.
            access = AccessScope(owner)
            schedule_id = candidate.get("schedule_id")
            if schedule_id and (kind != "commitment" or not isinstance(schedule_id, str) or schedule_id not in item.content
                                or self.schedule_resolver is None or self.schedule_resolver(schedule_id, owner) is None):
                raise ValueError("knowledge_schedule_not_authoritative")
            linked_task = candidate.get("task_id")
            if linked_task and (kind != "commitment" or not isinstance(linked_task, str) or linked_task not in item.content
                                or self.store.task_status(linked_task, owner) is None):
                raise ValueError("knowledge_task_not_authoritative")
            commitment_state = candidate.get("commitment_state", "unresolved") if kind == "commitment" else None
            if commitment_state in {"reported_completed", "reported_cancelled"}:
                state_quote = candidate.get("state_quote")
                if not isinstance(state_quote, str) or not state_quote.strip() or state_quote not in item.content:
                    raise ValueError("knowledge_commitment_state_quote_required")
            identity = json.dumps([KNOWLEDGE_VERSION, owner, source.source_id, candidate], sort_keys=True, ensure_ascii=False)
            dependencies = tuple(dict.fromkeys(candidate.get("supporting_memory_ids", [])))
            support = {source.source_id: (source, quote)}
            observation_time = source.observed_at
            for dependency_id in dependencies:
                dependency = related[dependency_id]
                for reference, supporting_quote in zip(dependency.sources, dependency.quotes):
                    support.setdefault(reference.source_id, (reference, supporting_quote))
                if dependency.kind == "observation" or dependency.basis == "tool_observation":
                    kind, persistence = "observation", "contextual"
                    if dependency.basis == "tool_observation":
                        basis = "tool_observation"
                    observation_time = dependency.observed_at
            if len(support) > 16:
                raise ValueError("knowledge_support_budget_exceeded")
            records.append(KnowledgeRecord(
                memory_id="memory:" + hashlib.sha256(identity.encode()).hexdigest(), content=content,
                access=access, sources=tuple(ref for ref, _ in support.values()),
                quotes=tuple(text for _, text in support.values()), recorded_at=now(), kind=kind,
                subjects=tuple(sorted({s.strip().casefold() for s in subjects})), claim_key=key.strip().casefold(), basis=basis, persistence=persistence,
                status="disputed" if conflicts else "active", observed_at=observation_time,
                valid_from=candidate.get("valid_from"), valid_until=candidate.get("valid_until"),
                supersedes=supersedes, conflicts_with=conflicts, project_id=project_id,
                task_id=linked_task, schedule_id=schedule_id, commitment_state=commitment_state, dependencies=dependencies))
        # Re-read sources before persistence, rather than treating extraction input as authority.
        current = self.resolve(source, owner)
        if current is None or current.source != source or current.content != item.content:
            raise ValueError("knowledge_source_no_longer_available")
        self._check_cancelled(cancel_checker)
        records = list({r.memory_id: r for r in records}.values())
        pairs = self.store.comparison_pairs(owner, records, self.resolve)
        comparisons = {}
        for offset in range(0, len(pairs), 8):
            self._check_cancelled(cancel_checker)
            if self.comparison_reviewer is None:
                raise ValueError("knowledge_comparison_unavailable")
            batch = pairs[offset:offset + 8]
            judgments = self.comparison_reviewer(batch)
            if len(judgments) != len(batch) or any(j not in {"equivalent", "compatible", "conflicting"} for j in judgments):
                raise ValueError("knowledge_comparison_ambiguous")
            comparisons.update({(old.memory_id, new.memory_id): judgment for (old, new), judgment in zip(batch, judgments)})
        # Models ran outside the transaction: recheck *all* supporting sources,
        # then commit rejects any concurrently introduced, unreviewed relation.
        for record in records:
            for reference, supporting_quote in zip(record.sources, record.quotes):
                live = self.resolve(reference, owner)
                if live is None or live.source != reference or supporting_quote not in live.content:
                    raise ValueError("knowledge_support_no_longer_available")
        self._check_cancelled(cancel_checker)
        self.store.commit(owner, source.source_id, records, resolver=self.resolve, comparisons=comparisons)

    def recall(self, requester: str, *, query: str = "", historical: bool = False, limit: int = 16) -> list[KnowledgeRecord]:
        return self.store.records(requester, self.resolve, query=query, historical=historical, limit=limit)

    def render(self, record: KnowledgeRecord, requester: str) -> str:
        links = ", ".join(s.source_id for s in record.sources)
        status = ""
        if record.kind == "commitment":
            schedule = self.schedule_resolver(record.schedule_id, requester) if record.schedule_id and self.schedule_resolver else None
            task = self.store.task_status(record.task_id, requester) if record.task_id else None
            status = f"; commitment={record.commitment_state}; schedule={record.schedule_id or 'none'} state={getattr(schedule, 'status', 'unavailable')}; task={record.task_id or 'none'} last_recorded_state={task or 'unavailable'}; memory does not schedule actions"
        return (f"- [{record.memory_id}; {record.kind}; {record.basis}; {record.status}; persistence={record.persistence}; "
                f"observed={record.observed_at or 'unknown'}; valid_from={record.valid_from or 'unknown'}; "
                f"valid_until={record.valid_until or 'unknown'}; sources={links}{status}] " + " ".join(record.content.split()))
