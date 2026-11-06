"""Authorized structured recall and bounded source expansion."""
from __future__ import annotations

import hashlib
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


def _identity(context):
    if context is None or not str(context.task.user or "").strip():
        raise ValueError("memory_requester_required")
    return str(context.task.user)


def _snapshot(context, requester: str, query: str, historical: bool, selected: set[str]):
    """All source adapters run independently. Access precedes ranking and excerpts."""
    hits, coverage = [], {}
    memory = getattr(context, "memory", None)
    knowledge = getattr(memory, "knowledge", None)
    conversations = getattr(context, "conversation_store", None)

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
        return [SearchHit(item.source.source_id, "conversation", item) for item in items], callable(snapshot)

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
            candidates.append(SearchHit(record.memory_id, "knowledge", item, record.subjects, record.to_dict()))
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
        if kind == "conversation":
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
            if (candidate.source.source_id == source_id or not candidate.visible_to(requester)
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
