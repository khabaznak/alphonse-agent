"""Authorized structured recall and bounded source expansion."""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from alphonse.agent_v2.core.memory.contracts import ContextItem, SourceReference, AccessScope, RecallResult, conversation_item
from alphonse.agent_v2.core.memory.recall import (
    SOURCE_TYPES, SearchHit, bounded_int, date_bound, fingerprint, normalized, rank,
)
from alphonse.agent_v2.core.core import ToolDescriptor, ToolExecutionContext, ToolKind
from alphonse.agent_v2.core.tools.registry import ToolDefinition

SEARCH_MEMORY_TOOL_ID = "native.search_memory"
SEARCH_MEMORY_TOOL_NAME = "search_memory"
READ_MEMORY_SOURCE_TOOL_ID = "native.read_memory_source"
OWNER_CONVERSATION_MAX_CHARS = 36000


def complete_conversation_search(actor: str, tool_id: str, arguments: dict[str, Any], payload: dict[str, Any], *,
                                 allow_excerpt_truncation: bool = False,
                                 allow_search_overflow: bool = False,
                                 allow_search_pages: bool = False) -> bool:
    """Recognize an owner search page under the caller's coverage policy.

    The native tool must have run under the authenticated actor. This checks
    the response envelope only; it does not fence changes or establish that
    historical search results are current, exhaustive in meaning, or authorized
    for later publication. The opt-in overflow flag is only for a host that
    withholds incomplete result content and requests a narrower search.
    """
    query = arguments.get('query')
    if (not isinstance(actor, str) or not actor.strip() or tool_id != SEARCH_MEMORY_TOOL_ID
            or arguments.get('source_types') != ['conversation']
            or not isinstance(query, str) or not query.strip()
            or type(arguments.get('offset', 0)) is not int or arguments.get('offset', 0) < 0
            or (arguments.get('offset', 0) != 0 and not allow_search_pages)
            or type(payload.get('schema_version')) is not int or payload.get('schema_version') != 2
            or payload.get('query') != query.strip() or payload.get('status') not in ('found', 'empty')
            or payload.get('coverage') != {'conversation': 'complete'} or payload.get('backend') != 'sqlite_fts5'
            or type(payload.get('truncated')) is not bool
            or (payload['truncated'] and not allow_excerpt_truncation)):
        return False
    results = payload.get('results')
    if (not isinstance(results, list) or type(payload.get('total_matches')) is not int
            or (payload['status'] == 'found') != bool(results)):
        return False
    offset = arguments.get('offset', 0)
    if allow_search_pages:
        limit = arguments.get('limit', 8)
        if (type(limit) is not int or not 1 <= limit <= 30
                or payload.get('offset') != offset or not isinstance(payload.get('snapshot'), str)
                or not payload['snapshot'] or offset > payload['total_matches']
                or len(results) != min(limit, payload['total_matches'] - offset)
                or any(type(item.get('rank')) is not int or item['rank'] != offset + i + 1
                       for i, item in enumerate(results) if isinstance(item, dict))):
            return False
    next_offset = payload.get('next_offset')
    if next_offset is None:
        if payload['total_matches'] != offset + len(results):
            return False
    elif (not (allow_search_overflow or allow_search_pages) or type(next_offset) is not int
          or next_offset != offset + len(results) or next_offset <= offset
          or payload['total_matches'] <= next_offset or payload['truncated'] is not True):
        return False
    for item in results:
        if not isinstance(item, dict):
            return False
        source_id = item.get('source_id')
        source = item.get('source')
        access = source.get('access') if isinstance(source, dict) else None
        if (not isinstance(source_id, str) or not source_id.startswith('conversation:')
                or item.get('source_type') != 'conversation' or not isinstance(source, dict)
                or source.get('source_id') != source_id or source.get('kind') != 'conversation'
                or not isinstance(access, dict) or access.get('owner_user_id') != actor
                or type(item.get('truncated')) is not bool
                or (item['truncated'] and not allow_excerpt_truncation)):
            return False
    # A complete result page can still contain deliberately shortened source
    # excerpts. Never confuse those with a missing/paginated result page.
    return payload['truncated'] == (next_offset is not None or any(item['truncated'] for item in results))


def complete_conversation_read(actor: str, tool_id: str, arguments: dict[str, Any], payload: dict[str, Any], *,
                               allow_chunks: bool = False) -> bool:
    """Recognize a complete direct-owner conversation read, not current authority.

    Invoke the native tool under the authenticated actor first. This checks the
    returned envelope only; it cannot fence revocation, publication or prove
    that historical text is current or supports a particular answer.
    """
    source_id = arguments.get('source_id')
    if (not isinstance(actor, str) or not actor.strip() or tool_id != READ_MEMORY_SOURCE_TOOL_ID
            or not isinstance(source_id, str) or not source_id.startswith('conversation:')
            or not source_id.removeprefix('conversation:').strip()
            or type(arguments.get('offset', 0)) is not int or arguments.get('offset', 0) < 0
            or (arguments.get('offset', 0) != 0 and not allow_chunks)
            or type(payload.get('schema_version')) is not int or payload.get('schema_version') != 2
            or payload.get('status') != 'found' or type(payload.get('truncated')) is not bool
            or payload.get('historical_only') is not True):
        return False
    items = payload.get('items')
    if not isinstance(items, list) or len(items) != 1 or not isinstance(items[0], dict):
        return False
    item = items[0]
    source = item.get('source')
    access = source.get('access') if isinstance(source, dict) else None
    content = item.get('content')
    offset = arguments.get('offset', 0)
    if not (item.get('source_id') == source_id and isinstance(source, dict)
        and source.get('source_id') == source_id and source.get('kind') == 'conversation'
        and isinstance(access, dict) and access.get('owner_user_id') == actor
        and isinstance(content, str) and type(item.get('start_char')) is int and item['start_char'] == offset
        and type(item.get('end_char')) is int and item['end_char'] == offset + len(content)
        and type(item.get('total_chars')) is int and 0 <= offset <= item['end_char'] <= item['total_chars']):
        return False
    total, end = item['total_chars'], item['end_char']
    # A partial page is useful only as a bounded, contiguous sequence. It is
    # not complete evidence on its own, and an empty intermediate page cannot
    # advance a sequence.
    if (not allow_chunks and (offset != 0 or end != total)
            or allow_chunks and (total > OWNER_CONVERSATION_MAX_CHARS or (offset < total and end == offset))
            or payload.get('next_offset') != (end if end < total else None)
            or payload['truncated'] != (offset > 0 or end < total)):
        return False
    # A read may also return adjacent messages. The anchor's owner does not
    # authorize promoting arbitrary neighbor text into the candidate's evidence.
    neighbors = payload.get('neighbors')
    count = arguments.get('neighbors', 2)
    if (type(count) is not int or not 0 <= count <= 5 or not isinstance(neighbors, list)
            or len(neighbors) > 2 * count or payload.get('supporting_sources') != []):
        return False
    seen = {source_id}
    for neighbor in neighbors:
        if not isinstance(neighbor, dict):
            return False
        neighbor_source = neighbor.get('source')
        neighbor_access = neighbor_source.get('access') if isinstance(neighbor_source, dict) else None
        neighbor_id = neighbor_source.get('source_id') if isinstance(neighbor_source, dict) else None
        if (not isinstance(neighbor_id, str) or not neighbor_id.startswith('conversation:')
                or neighbor_id in seen or neighbor_source.get('kind') != 'conversation'
                or not isinstance(neighbor_access, dict) or neighbor_access.get('owner_user_id') != actor
                or not isinstance(neighbor.get('content'), str) or neighbor.get('truncated') is not False):
            return False
        seen.add(neighbor_id)
    return True


def revalidate_conversation_observation(
    actor: str, tool_id: str, arguments: dict[str, Any], payload: dict[str, Any], *, context: ToolExecutionContext,
    allow_excerpt_truncation: bool = False, allow_read_chunks: bool = False,
    allow_search_overflow: bool = False,
    allow_search_pages: bool = False,
) -> bool:
    """Re-read an owner-scoped observation before promoting it to evidence.

    This is a point-in-time check, not authentication, an atomic revocation
    fence, semantic verification, or authorization to publish. Re-running the
    native operation directly avoids repeating invocation UI/memory events.
    """
    try:
        if (not isinstance(context, ToolExecutionContext)
                or not isinstance(actor, str) or not actor.strip()
                or getattr(context.task, 'user', None) != actor
                or not isinstance(arguments, dict) or not isinstance(payload, dict)):
            return False
        cancelled = context.cancellation_checker
        if cancelled is not None and (not callable(cancelled) or cancelled() is not False):
            return False
        if tool_id == SEARCH_MEMORY_TOOL_ID:
            if not complete_conversation_search(actor, tool_id, arguments, payload,
                                                allow_excerpt_truncation=allow_excerpt_truncation,
                                                allow_search_overflow=allow_search_overflow,
                                                allow_search_pages=allow_search_pages):
                return False
            current = execute_search_memory(arguments, context=context)
            valid = complete_conversation_search(actor, tool_id, arguments, current,
                                                  allow_excerpt_truncation=allow_excerpt_truncation,
                                                  allow_search_overflow=allow_search_overflow,
                                                  allow_search_pages=allow_search_pages)
        elif tool_id == READ_MEMORY_SOURCE_TOOL_ID:
            if not complete_conversation_read(actor, tool_id, arguments, payload, allow_chunks=allow_read_chunks):
                return False
            current = execute_read_memory_source(arguments, context=context)
            valid = complete_conversation_read(actor, tool_id, arguments, current, allow_chunks=allow_read_chunks)
        else:
            return False
        if cancelled is not None and cancelled() is not False:
            return False
        # JSON comparison is type-sensitive (unlike True == 1), rejects NaNs,
        # and covers all returned content, including neighboring messages.
        return bool(valid and json.dumps(payload, sort_keys=True, allow_nan=False)
                    == json.dumps(current, sort_keys=True, allow_nan=False))
    except Exception:
        return False


def conversation_search_sequences(actor: str, observations: dict[str, tuple[str, dict, dict]]) -> list[list[tuple[str, dict, dict]]]:
    """Group sequential searches, rejecting gaps, changed snapshots and mixed filters."""
    groups: list[list[tuple[str, dict, dict]]] = []
    for ref, (tool, args, payload) in observations.items():
        if tool != SEARCH_MEMORY_TOOL_ID:
            continue
        if not complete_conversation_search(actor, tool, args, payload, allow_excerpt_truncation=True,
                                            allow_search_pages=True):
            raise ValueError('owner_search_sequence_invalid')
        offset = args.get('offset', 0)
        if offset == 0:
            groups.append([(ref, args, payload)])
            continue
        if not groups:
            raise ValueError('owner_search_sequence_invalid')
        pages = groups[-1]
        first_args, first = pages[0][1:]
        previous = pages[-1][2]
        filters = lambda a: {k: v for k, v in a.items() if k != 'offset'}
        if (len(pages) >= 3 or first['total_matches'] > 90
                or previous['next_offset'] != offset or filters(args) != filters(first_args)
                or payload['snapshot'] != first['snapshot'] or payload['total_matches'] != first['total_matches']):
            raise ValueError('owner_search_sequence_invalid')
        pages.append((ref, args, payload))
    return groups


def conversation_read_sequences(actor: str, observations: dict[str, tuple[str, dict, dict]]) -> dict[str, list[tuple[str, dict, dict]]]:
    """Validate contiguous owner-read prefixes and group them by source.

    Prefixes may be incomplete, but gaps, overlaps, changed source metadata or
    inconsistent lengths are rejected. Callers must independently revalidate
    every page against the currently authorized source before exposing content.
    """
    groups: dict[str, list[tuple[str, dict, dict]]] = {}
    for ref, (tool, args, payload) in observations.items():
        if tool != READ_MEMORY_SOURCE_TOOL_ID:
            continue
        if not complete_conversation_read(actor, tool, args, payload, allow_chunks=True):
            raise ValueError('owner_read_sequence_invalid')
        groups.setdefault(args['source_id'], []).append((ref, args, payload))
    for pages in groups.values():
        pages.sort(key=lambda entry: entry[1].get('offset', 0))
        offset = 0
        source = None
        total = None
        for _, args, payload in pages:
            item = payload['items'][0]
            if (args.get('offset', 0) != offset or (total is not None and total != item['total_chars'])
                    or (source is not None and json.dumps(source, sort_keys=True, allow_nan=False)
                        != json.dumps(item['source'], sort_keys=True, allow_nan=False))):
                raise ValueError('owner_read_sequence_invalid')
            offset = item['end_char']
            total, source = item['total_chars'], item['source']
    return groups


def _identity(context):
    if context is None or not str(context.task.user or "").strip():
        raise ValueError("memory_requester_required")
    return str(context.task.user)


def _current_owner_conversation_id(context: ToolExecutionContext, requester: str) -> str | None:
    """Resolve the active owner input by its server-bound queue identity.

    A submitted request is task input, not historical evidence. Do not derive
    its identity from model arguments, text, or a caller-provided event ID.
    Older in-flight requests may not yet have a conversation record.
    """
    task = context.task
    if getattr(task, 'intelligence_engine', None) != 'single_loop_experimental':
        return None
    from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    queue = context.messages
    conversations = context.conversation_store
    receipt = context.claim_receipt
    # Ephemeral injected processors do not use the trusted owner ingress and
    # do not create its durable conversation event. A fenced SQLite host must
    # instead prove the live claim and shared database below.
    from alphonse.agent_v2.core.messages.queue import InMemoryMessageQueue
    if type(queue) is InMemoryMessageQueue and receipt is None:
        return None
    try:
        if (type(queue) is not SQLiteMessageQueue or type(conversations) is not SQLiteConversationStore
                or receipt is None or queue.experimental_claim_fencing is not True
                or not os.path.samefile(queue.db_path, conversations.db_path)
                or requester != task.user or task.message_id != receipt.message_id
                or task.task_id != receipt.message_id or receipt.user_id != requester
                or not queue.verify_trusted_local_owner_claim(receipt, task)):
            raise ValueError('owner_memory_claim_unavailable')
        event = conversations.event_for_source_message_id(
            owner_user_id=requester, source_message_id=f'inbound:{receipt.message_id}')
        if event is None:
            return None
        if (event.owner_user_id != requester or event.role != 'user'
                or event.content != str(task.goal or '').strip()
                or event.source_message_id != f'inbound:{receipt.message_id}'):
            raise ValueError('owner_memory_claim_unavailable')
        return f'conversation:{event.event_id}'
    except (AttributeError, OSError, TypeError):
        raise ValueError('owner_memory_claim_unavailable') from None


def _snapshot(context, requester: str, query: str, historical: bool, selected: set[str]):
    """All source adapters run independently. Access precedes ranking and excerpts."""
    hits, coverage = [], {}
    memory = getattr(context, "memory", None)
    knowledge = getattr(memory, "knowledge", None)
    conversations = getattr(context, "conversation_store", None)
    excluded = _current_owner_conversation_id(context, requester) if 'conversation' in selected else None

    def collect(kind, reader):
        if kind not in selected:
            return
        if reader is None:
            coverage[kind] = "unavailable"
            return
        try:
            candidates, complete = reader()
            # Even alternate/misbehaving adapters must pass the common access gate.
            hits.extend(hit for hit in candidates if hit.item.visible_to(requester))
            coverage[kind] = "complete" if complete else "partial"
        except OSError:
            coverage[kind] = "unavailable"
        except Exception:
            # Do not echo backend exception text: it may contain private data/paths.
            coverage[kind] = "failed"

    def conversation_hits():
        snapshot = getattr(conversations, "recall_items", None)
        if callable(snapshot):
            items = snapshot(requester)
        else:
            events = conversations.search(owner_user_id=requester, query=query, limit=8)
            items = [item for event in events if (item := conversation_item(event, requester)) is not None]
        return [SearchHit(item.source.source_id, "conversation", item) for item in items
                if item.source.source_id != excluded], callable(snapshot)

    def knowledge_hits():
        records = knowledge.recall(requester, query="", historical=historical, limit=None)
        candidates = []
        for record in records:
            if not record.access.allows(requester) or not all(s.access.allows(requester) for s in record.sources):
                continue
            # The knowledge store resolves transitive sources/dependencies under current access.
            # Recheck the public sources here too, before rendering alternate adapters.
            if not callable(getattr(knowledge, "resolve", None)):
                continue
            live = [knowledge.resolve(source, requester) for source in record.sources]
            if any(item is None or not item.visible_to(requester) or item.source.source_id != source.source_id
                   for item, source in zip(live, record.sources)):
                continue
            item = ContextItem(live[0].source, record.content, "derived_knowledge")
            candidates.append(SearchHit(record.memory_id, "knowledge", item, record.subjects, record.public_dict()))
        return candidates, True

    def task_hits():
        items = knowledge.store.recall_task_items(requester)
        return [SearchHit(item.source.source_id, "task_evidence", item) for item in items], True

    def ledger_hits():
        snapshot = getattr(memory, "recall_items", None)
        if callable(snapshot):
            items = snapshot(user_id=requester, project_id=str(context.task.project_id or ""))
            return [SearchHit(item.source.source_id, "legacy_ledger", item) for item in items], True
        text = str(memory.search(user_id=requester, project_id=str(context.task.project_id or ""), query=query) or "")
        if not text or text in {"- (no matches)", "- (no searchable terms)"}:
            return [], False
        identity = "legacy-ledger:" + hashlib.sha256((requester + "\0" + str(context.task.project_id) + "\0" + text).encode()).hexdigest()
        item = ContextItem(SourceReference(identity, "legacy_ledger", AccessScope(requester), provenance="legacy_unverified"), text, "audit")
        return [SearchHit(identity, "legacy_ledger", item)], False

    collect("conversation", conversation_hits if callable(getattr(conversations, "recall_items", None)) or callable(getattr(conversations, "search", None)) else None)
    collect("knowledge", knowledge_hits if callable(getattr(knowledge, "recall", None)) else None)
    collect("task_evidence", task_hits if callable(getattr(getattr(knowledge, "store", None), "recall_task_items", None)) else None)
    collect("legacy_ledger", ledger_hits if callable(getattr(memory, "recall_items", None)) or callable(getattr(memory, "search", None)) else None)
    unique = {hit.source_id: hit for hit in hits}
    return list(unique.values()), coverage


def execute_search_memory(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    requester = _identity(context)
    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip() or len(query) > 512:
        raise ValueError("memory_search_query_required_or_too_long")
    query = query.strip()
    historical = arguments.get("historical", False)
    if type(historical) is not bool:
        raise ValueError("memory_historical_must_be_boolean")
    limit = bounded_int(arguments, "limit", 8, 1, 30)
    offset = bounded_int(arguments, "offset", 0, 0, 1000000)
    types = arguments.get("source_types", sorted(SOURCE_TYPES))
    if not isinstance(types, list) or not types or len(types) > 4 or any(not isinstance(t, str) or t not in SOURCE_TYPES for t in types):
        raise ValueError("memory_invalid_source_types")
    entities = arguments.get("entities", [])
    if not isinstance(entities, list) or len(entities) > 8 or any(not isinstance(e, str) or not e.strip() or len(e) > 128 for e in entities):
        raise ValueError("memory_invalid_entities")
    provider = getattr(context, "user_timezone_provider", None)
    zone = str(provider(requester) if provider else "UTC")
    start = date_bound(arguments.get("start_date"), zone)
    end = date_bound(arguments.get("end_date"), zone, end=True)
    if start and end and start >= end:
        raise ValueError("memory_invalid_date_range")
    hits, coverage = _snapshot(context, requester, query, historical, set(types))
    filtered = []
    for hit in hits:
        if start or end:
            observed = date_bound(hit.event_time, "UTC")
            if observed is None or (start and observed < start) or (end and observed >= end):
                continue
        text = normalized(hit.item.content + " " + " ".join(hit.entities))
        if any(normalized(entity) not in text for entity in entities):
            continue
        filtered.append(hit)
    try:
        ranked, backend = rank(filtered, query)
    except Exception:
        return {"schema_version": 2, "status": "failed", "results": [], "coverage": coverage,
                "error": "memory_ranking_failed", "query": query}
    page = ranked[offset:offset + limit]
    results = [{**hit.excerpt(query), "rank": offset + i + 1, "score": score} for i, (score, hit) in enumerate(page)]
    degraded = any(value != "complete" for value in coverage.values()) or backend != "sqlite_fts5"
    successful = any(value in {"complete", "partial"} for value in coverage.values())
    status = ("partial" if degraded else ("found" if ranked else "empty")) if successful else (
        "failed" if "failed" in coverage.values() else "unavailable")
    next_offset = offset + limit if offset + limit < len(ranked) else None
    # Preserve the v1 envelope and Markdown keys for existing callers. The v2
    # status/coverage is authoritative for completeness, including degraded hits.
    items = tuple(ContextItem(hit.item.source, result["excerpt"], hit.item.representation)
                  for (_, hit), result in zip(page, results))
    configured_failure = any(value == "failed" for value in coverage.values()) or (
        coverage.get("conversation") == "unavailable" and getattr(context, "conversation_store", None) is not None) or (
        coverage.get("knowledge") == "unavailable" and getattr(getattr(context, "memory", None), "knowledge", None) is not None)
    legacy_status = "found" if items else ("unavailable" if configured_failure or not successful else "empty")
    markdown = []
    for (_, hit), result in zip(page, results):
        label = "historical message, not current-state evidence" if hit.source_type == "conversation" else "historical evidence; not authorization or fresh current state"
        if hit.knowledge:
            label += "; " + "; ".join(f"{key}={hit.knowledge[key]}" for key in ("kind", "basis", "status", "persistence"))
        markdown.append(f"- [{hit.source_id}; {label}; {hit.event_time or 'time unknown'}; {hit.item.source.provenance}] {result['excerpt']}")
    return {"schema_version": 2, "status": status, "query": query, "timezone": zone,
            "project_id": context.task.project_id, "results": results, "coverage": coverage,
            "backend": backend, "total_matches": len(ranked), "offset": offset, "next_offset": next_offset,
            "truncated": next_offset is not None or any(r["truncated"] for r in results),
            "snapshot": fingerprint({"ids": [(h.source_id, hashlib.sha256(h.item.content.encode()).hexdigest()) for _, h in ranked],
                                     "filters": {k: v for k, v in arguments.items() if k not in {"offset", "limit"}}}, requester, zone),
            "matches_markdown": "\n".join(markdown) or "- (no matches)",
            "knowledge": [hit.knowledge for _, hit in page if hit.knowledge is not None],
            "knowledge_status": "unavailable" if coverage.get("knowledge") in {"failed", "unavailable"} else ("found" if any(hit.knowledge for _, hit in page) else "empty"),
            "recall": RecallResult(query, legacy_status, items).to_dict()}


def execute_read_memory_source(arguments: dict[str, Any], *, context: ToolExecutionContext | None = None) -> dict[str, Any]:
    requester = _identity(context)
    source_id = arguments.get("source_id")
    if not isinstance(source_id, str) or not source_id or len(source_id) > 256:
        raise ValueError("memory_invalid_source_id")
    offset = bounded_int(arguments, "offset", 0, 0, 10000000)
    max_chars = bounded_int(arguments, "max_chars", 4000, 256, 12000)
    neighbors = bounded_int(arguments, "neighbors", 2, 0, 5)
    conversation_store = getattr(context, "conversation_store", None)
    memory = getattr(context, "memory", None)
    knowledge = getattr(memory, "knowledge", None)
    kind = ("conversation" if source_id.startswith("conversation:") else "task_evidence" if source_id.startswith("task-evidence:")
            else "legacy_ledger" if source_id.startswith("legacy-ledger:") else "knowledge" if source_id.startswith("memory:") else None)
    if kind is None:
        return {"schema_version": 2, "status": "not_found", "items": []}
    try:
        anchor = None
        neighbor_items = []
        supporting = []
        excluded = None
        if kind == "conversation":
            excluded = _current_owner_conversation_id(context, requester)
            if source_id == excluded:
                return {"schema_version": 2, "status": "not_found", "items": []}
            if not callable(getattr(conversation_store, "read", None)):
                return {"schema_version": 2, "status": "unavailable", "items": []}
            event = conversation_store.read(event_id=source_id.removeprefix("conversation:"), requester_user_id=requester)
            item = conversation_item(event, requester)
            if item and item.source.source_id == source_id:
                anchor = SearchHit(source_id, kind, item)
                reader = getattr(conversation_store, "read_neighbors", None)
                if callable(reader):
                    neighbor_items = reader(event_id=event.event_id, requester_user_id=requester, neighbors=neighbors)
        elif kind == "task_evidence":
            store = getattr(knowledge, "store", None)
            if not callable(getattr(store, "task_source", None)):
                return {"schema_version": 2, "status": "unavailable", "items": []}
            item = store.task_source(SourceReference(source_id, kind, AccessScope(requester)), requester)
            if item and item.source.source_id == source_id and item.visible_to(requester):
                anchor = SearchHit(source_id, kind, item)
        else:
            hits, coverage = _snapshot(context, requester, "", True, {kind})
            if coverage.get(kind) != "complete":
                return {"schema_version": 2, "status": coverage.get(kind, "unavailable"), "items": []}
            anchor = next((hit for hit in hits if hit.source_id == source_id), None)
            if anchor and anchor.knowledge:
                supporting = anchor.knowledge["sources"]
        if anchor is None or not anchor.item.visible_to(requester):
            return {"schema_version": 2, "status": "not_found", "items": []}
        text = anchor.item.content
        end = min(len(text), offset + max_chars)
        item = {"source_id": source_id, "source": anchor.item.source.to_dict(), "content": text[offset:end],
                "start_char": offset, "end_char": end, "total_chars": len(text), "knowledge": anchor.knowledge}
        adjacent = []
        for candidate in neighbor_items:
            if len(adjacent) >= 2 * neighbors:
                break
            if (candidate.source.source_id == source_id or candidate.source.source_id == excluded or not candidate.visible_to(requester)
                    or candidate.source.access.owner_user_id != anchor.item.source.access.owner_user_id):
                continue
            adjacent.append({"source": candidate.source.to_dict(), "content": candidate.content[:1000],
                             "truncated": len(candidate.content) > 1000})
        return {"schema_version": 2, "status": "found", "items": [item], "neighbors": adjacent,
                "supporting_sources": supporting, "next_offset": end if end < len(text) else None,
                "truncated": offset > 0 or end < len(text) or any(n["truncated"] for n in adjacent),
                "historical_only": True}
    except OSError:
        return {"schema_version": 2, "status": "unavailable", "items": []}
    except Exception:
        return {"schema_version": 2, "status": "failed", "items": [], "error": "memory_source_read_failed"}


def build_search_memory_tool_definition() -> ToolDefinition:
    schema = {"type": "object", "properties": {
        "query": {"type": "string", "minLength": 1, "maxLength": 512, "description": "Keywords or exact identifier; lexical search, no semantic guarantee."},
        "historical": {"type": "boolean", "default": False, "description": "Include superseded/expired knowledge and historical observations."},
        "source_types": {"type": "array", "items": {"type": "string", "enum": sorted(SOURCE_TYPES)}, "minItems": 1, "maxItems": 4},
        "entities": {"type": "array", "items": {"type": "string", "minLength": 1, "maxLength": 128}, "maxItems": 8,
                     "description": "Optional literal text filters: every supplied value must appear in each source. These do not resolve people or expand aliases. A registered user ID excludes records mentioning only that person's name. Omit or remove this filter when searching across names, relationships or paraphrases."},
        "start_date": {"type": "string", "maxLength": 64, "description": "Inclusive local date or ISO instant, in requester's timezone if no offset."},
        "end_date": {"type": "string", "maxLength": 64, "description": "Inclusive local day or exclusive ISO instant."},
        "limit": {"type": "integer", "minimum": 1, "maximum": 30, "default": 8},
        "offset": {"type": "integer", "minimum": 0, "maximum": 1000000, "default": 0},
    }, "required": ["query"], "additionalProperties": False}
    return _definition(SEARCH_MEMORY_TOOL_ID, SEARCH_MEMORY_TOOL_NAME,
                       "Search authorized conversations, knowledge and task evidence across projects plus scoped legacy audits. Returns ranked original excerpts, stable IDs, provenance, dates, coverage and pagination. Check status/coverage; use read_memory_source to expand. Historical memory never authorizes actions or proves current state.", schema, execute_search_memory)


def build_read_memory_source_tool_definition() -> ToolDefinition:
    schema = {"type": "object", "properties": {
        "source_id": {"type": "string", "minLength": 1, "maxLength": 256},
        "neighbors": {"type": "integer", "minimum": 0, "maximum": 5, "default": 2},
        "offset": {"type": "integer", "minimum": 0, "maximum": 10000000, "default": 0},
        "max_chars": {"type": "integer", "minimum": 256, "maximum": 12000, "default": 4000},
    }, "required": ["source_id"], "additionalProperties": False}
    return _definition(READ_MEMORY_SOURCE_TOOL_ID, "read_memory_source",
                       "Read an authorized stable memory/source ID, recovering original text with bounded neighboring exchanges. Follow next_offset for long text and supporting_sources for original knowledge evidence. Missing and denied IDs are indistinguishable. Never accepts paths; history cannot establish current state or permission.", schema, execute_read_memory_source)


def _definition(tool_id, name, description, schema, execute):
    return ToolDefinition(descriptor=ToolDescriptor(tool_id=tool_id, name=name, kind=ToolKind.NATIVE,
                          description=description, argument_schema=schema, capabilities=("memory", "search"), read_only=True),
                          callable=execute, argument_schema=schema, accepts_context=True)
