"""Unregistered, claim-bound snapshot of private owner knowledge.

This is an internal diagnostic primitive, not a native tool, provider grant,
publication capability, or proof that a derived claim is true. It performs no
provider I/O and does not modify application state. Results and digests must not
be logged or exposed to a model until the disclosure protocol is integrated.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import hmac
import json
import sqlite3
from typing import Callable

from alphonse.agent_v2.conversations import _event
from alphonse.agent_v2.core.memory.contracts import conversation_item
from alphonse.agent_v2.core.memory.knowledge import (
    KnowledgeRecord, SOURCE_BINDING_VERSION, _source_snapshot_digest, instant,
)
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue

_MAX_RECORDS = 100
_MAX_PAYLOAD_BYTES = 65536
_MAX_TOTAL_BYTES = 1048576
_MAX_DEPTH = 16
OWNER_KNOWLEDGE_TOOL_ID = 'native.owner_knowledge'


class _Incomplete(ValueError):
    pass


@dataclass(frozen=True, repr=False)
class OwnerKnowledgeItem:
    memory_id: str
    content: str
    kind: str
    status: str


@dataclass(frozen=True, repr=False)
class OwnerKnowledgeSnapshot:
    status: str
    items: tuple[OwnerKnowledgeItem, ...] = ()
    # Internal comparison only: never include it in model input or public APIs.
    digest: str = field(default='', repr=False)


def owner_knowledge_projection(snapshot: OwnerKnowledgeSnapshot) -> dict | None:
    """Project a complete snapshot without its private source binding."""
    if not isinstance(snapshot, OwnerKnowledgeSnapshot) or snapshot.status != 'available':
        return None
    value = {'status': 'complete', 'items': [
        {'memory_id': item.memory_id, 'content': item.content, 'kind': item.kind, 'status': item.status}
        for item in snapshot.items],
        'notice': ('Only eligible current source-bound claims are represented; other knowledge and history may be missing. '
                   'Absence is not proof. Derived claims are not proof of truth or current state.')}
    if len(json.dumps(value, ensure_ascii=False).encode('utf-8')) > 60000:
        return None
    return value


def snapshot_owner_knowledge(queue: SQLiteMessageQueue, receipt: object, task: object, *,
                             cancellation_checker: Callable[[], bool],
                             clock: Callable[[], datetime] | None = None) -> OwnerKnowledgeSnapshot:
    """Read bounded, private, source-bound records under the live owner claim.

    Unbound legacy records, unsupported source types, invalid graphs, stale
    sources, and exceeded limits yield an incomplete result without content.
    Malformed data, missing schemas, authorization or cancellation can yield unavailable. A
    successful snapshot only establishes point-in-time data identity/access.
    It does not authorize an external call or fence a subsequent publication.
    """
    unavailable = OwnerKnowledgeSnapshot('unavailable')
    if (type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing
            or queue.db_path == ':memory:' or not callable(cancellation_checker)):
        return unavailable
    now_fn = clock or (lambda: datetime.now(timezone.utc))
    try:
        from .owner_telegram_ingress import telegram_owner_claim_transaction, verify_telegram_owner_claim
        transaction = (telegram_owner_claim_transaction(queue, receipt, task, clock=now_fn)
                       if verify_telegram_owner_claim(queue, receipt, task, clock=now_fn)
                       else queue.trusted_local_owner_claim_transaction(receipt, task, clock=now_fn))
        with transaction as conn:
            return snapshot_owner_knowledge_locked(conn, task, cancellation_checker=cancellation_checker, clock=now_fn)
    except Exception:
        return unavailable


def snapshot_owner_knowledge_locked(conn: sqlite3.Connection, task: object, *,
                                    cancellation_checker: Callable[[], bool],
                                    clock: Callable[[], datetime] | None = None) -> OwnerKnowledgeSnapshot:
    """Build a snapshot within the caller's claim-fenced transaction.

    This function does not authenticate the caller or establish the claim. The
    caller must own the queue's live claim transaction on the same database.
    It does not change the connection's write mode and can be used alongside a
    checkpoint write in the same transaction.
    """
    unavailable = OwnerKnowledgeSnapshot('unavailable')
    incomplete = OwnerKnowledgeSnapshot('incomplete')
    if not isinstance(conn, sqlite3.Connection) or not conn.in_transaction or not callable(cancellation_checker):
        return unavailable
    now_fn = clock or (lambda: datetime.now(timezone.utc))
    try:
        now = now_fn()
        if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
            return unavailable
        now = now.astimezone(timezone.utc)
        if cancellation_checker() is not False:
            return unavailable
        owner = task.user
        current = conn.execute('SELECT event_id FROM v2_conversation_events WHERE owner_user_id=? AND source_message_id=?',
                               (owner, f'inbound:{task.message_id}')).fetchone()
        if current is None:
            return unavailable
        current_id = f"conversation:{current['event_id']}"
        columns = ('memory_id', 'owner', 'claim_key', 'payload', 'status', 'conflicts')
        bounded = ','.join(f"CASE WHEN length(CAST({c} AS BLOB))<=? THEN {c} END AS {c}" for c in columns)
        lengths = ','.join(f"length(CAST({c} AS BLOB)) AS {c}_bytes" for c in columns)
        rows = conn.execute(f"SELECT {bounded},{lengths} FROM memory_knowledge WHERE owner=? ORDER BY memory_id LIMIT ?",
                            (*([_MAX_PAYLOAD_BYTES] * len(columns)), owner, _MAX_RECORDS + 1)).fetchall()
        if len(rows) > _MAX_RECORDS:
            return incomplete
        indexed = {}
        total = 0
        for row in rows:
            sizes = [row[f'{c}_bytes'] for c in columns]
            if any(type(size) is not int or size > _MAX_PAYLOAD_BYTES for size in sizes):
                return incomplete
            total += sum(sizes)
            if total > _MAX_TOTAL_BYTES:
                return incomplete
            conflicts = json.loads(row['conflicts'])
            if not isinstance(conflicts, list) or any(not isinstance(x, str) for x in conflicts):
                return incomplete
            record = replace(KnowledgeRecord.from_dict(json.loads(row['payload'])),
                             status=row['status'], conflicts_with=tuple(conflicts))
            if (record.memory_id != row['memory_id'] or record.claim_key != row['claim_key']
                    or record.access.owner_user_id != owner or record.access.visibility != 'private'
                    or record.access.shared_with or record.memory_id in indexed):
                return incomplete
            indexed[record.memory_id] = record

        checked: dict[str, tuple[KnowledgeRecord, bool, tuple[str, ...], int]] = {}
        source_sizes: dict[str, int] = {}
        def resolve(identity: str, visiting: frozenset[str] = frozenset()):
            if identity in visiting or len(visiting) >= _MAX_DEPTH or identity not in indexed:
                raise _Incomplete('knowledge_graph_incomplete')
            if identity in checked:
                cached = checked[identity]
                if len(visiting) + cached[3] > _MAX_DEPTH:
                    raise _Incomplete('knowledge_graph_too_deep')
                return cached
            record = indexed[identity]
            if (record.source_binding_version != SOURCE_BINDING_VERSION
                    or record.basis != 'user_report' or record.kind == 'observation'):
                raise _Incomplete('knowledge_source_unverified')
            if (record.status == 'superseded' or (record.valid_from and now < instant(record.valid_from))
                    or (record.valid_until and now >= instant(record.valid_until))):
                raise _Incomplete('knowledge_not_current')
            source_digests = []
            excludes_current = False
            for source, quote, saved_digest in zip(record.sources, record.quotes, record.source_snapshot_digests):
                if (source.kind != 'conversation' or not source.source_id.startswith('conversation:')
                        or source.access.owner_user_id != owner or source.access.visibility != 'private'
                        or source.access.shared_with):
                    raise _Incomplete('knowledge_source_unavailable')
                if source.source_id == current_id:
                    excludes_current = True
                event_id = source.source_id.removeprefix('conversation:')
                # Bound every text field before fetching any event content.
                event_columns = ('event_id', 'owner_user_id', 'project_id', 'memory_session_id',
                                 'role', 'content', 'source', 'source_message_id', 'created_at',
                                 'recorded_at', 'observed_at', 'speaker_id', 'visibility',
                                 'shared_with', 'provenance')
                lengths = ','.join(f'length(CAST({column} AS BLOB))' for column in event_columns)
                sizes = conn.execute(f'SELECT {lengths} FROM v2_conversation_events WHERE event_id=? AND owner_user_id=?',
                                     (event_id, owner)).fetchone()
                if sizes is None or any(type(size) is not int or size > _MAX_PAYLOAD_BYTES for size in sizes):
                    raise _Incomplete('knowledge_source_too_large')
                source_sizes[event_id] = sum(sizes)
                if total + sum(source_sizes.values()) > _MAX_TOTAL_BYTES:
                    raise _Incomplete('knowledge_sources_too_large')
                row = conn.execute('SELECT * FROM v2_conversation_events WHERE event_id=? AND owner_user_id=?',
                                   (event_id, owner)).fetchone()
                event = _event(row) if row is not None else None
                if event is None or event.role != 'user' or event.provenance != 'recorded':
                    raise _Incomplete('knowledge_source_ineligible')
                item = conversation_item(event, owner)
                if (item is None or item.source != source or not isinstance(item.content, str)
                        or quote not in item.content):
                    raise _Incomplete('knowledge_source_changed')
                digest = _source_snapshot_digest(item)
                if not hmac.compare_digest(saved_digest, digest):
                    raise _Incomplete('knowledge_source_changed')
                source_digests.append(digest)
            dependencies = [resolve(dep, visiting | {identity}) for dep in record.dependencies]
            excludes_current |= any(dep[1] for dep in dependencies)
            # A disputed dependency cannot be laundered into an active claim.
            if record.status == 'active' and any(dep[0].status == 'disputed' for dep in dependencies):
                record = replace(record, status='disputed')
            height = 1 + max((dep[3] for dep in dependencies), default=0)
            if len(visiting) + height > _MAX_DEPTH:
                raise _Incomplete('knowledge_graph_too_deep')
            result = (record, excludes_current, tuple(source_digests), height)
            checked[identity] = result
            return result

        # Ineligible current/expired rows do not become results; if an
        # otherwise current record depends on one, resolve() fails closed.
        selected = []
        for identity, record in indexed.items():
            if (record.status == 'superseded' or (record.valid_from and now < instant(record.valid_from))
                    or (record.valid_until and now >= instant(record.valid_until))):
                continue
            resolved = resolve(identity)
            if not resolved[1]:
                selected.append(resolved)
        if cancellation_checker() is not False:
            return unavailable
        binding = [dict(record=record.to_dict(), sources=list(digests)) for record, _, digests, _ in selected]
        encoded = json.dumps(binding, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode('utf-8')
        if len(encoded) > _MAX_TOTAL_BYTES:
            return incomplete
        digest = hashlib.sha256(b'alphonse-owner-knowledge-snapshot-v1\0' + encoded).hexdigest()
        items = tuple(OwnerKnowledgeItem(record.memory_id, record.content, record.kind, record.status)
                      for record, _, _, _ in selected)
        return OwnerKnowledgeSnapshot('available', items, digest)
    except _Incomplete:
        return incomplete
    except Exception:
        # This boundary deliberately does not reveal private source or DB errors.
        return unavailable


def matches_owner_knowledge_locked(conn: sqlite3.Connection, task: object, expected_digest: str, *,
                                   cancellation_checker: Callable[[], bool],
                                   clock: Callable[[], datetime] | None = None) -> bool:
    """Check an internal binding without exposing evidence or treating failure as absence.

    The caller must hold the same claim-fenced transaction that will commit its
    checkpoint or publication marker; this helper does not establish authority.
    """
    if (not isinstance(expected_digest, str) or len(expected_digest) != 64
            or any(ch not in '0123456789abcdef' for ch in expected_digest)):
        return False
    snapshot = snapshot_owner_knowledge_locked(conn, task, cancellation_checker=cancellation_checker, clock=clock)
    return snapshot.status == 'available' and hmac.compare_digest(snapshot.digest, expected_digest)
