"""Deterministic lexical recall over authorized source snapshots (no model calls)."""
from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
import hashlib
import json
import re
import sqlite3
import unicodedata
from zoneinfo import ZoneInfo

from .contracts import ContextItem

SOURCE_TYPES = {"conversation", "knowledge", "task_evidence", "legacy_ledger"}


def normalized(value: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", value.casefold()) if not unicodedata.combining(c))


def keywords(query: str) -> list[str]:
    # Ignore query syntax; exact punctuation-bearing identifiers remain in the text.
    stop = {"the", "a", "an", "is", "was", "what", "which", "did", "of", "to", "and", "el", "la", "los", "las", "de", "que", "y"}
    tokens = list(dict.fromkeys(re.findall(r"\w+", normalized(query))))
    return [word for word in tokens if word not in stop][:24] or tokens[:24]


def bounded_int(arguments: dict, name: str, default: int, low: int, high: int) -> int:
    value = arguments.get(name, default)
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"memory_{name}_out_of_bounds")
    return value


def date_bound(value: str | None, zone: str, *, end: bool = False) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("memory_invalid_date")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        day = datetime.fromisoformat(value).date() + (timedelta(days=1) if end else timedelta())
        parsed = datetime.combine(day, time(), ZoneInfo(zone))
    else:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo(zone))
    return parsed.astimezone(timezone.utc)


@dataclass(frozen=True)
class SearchHit:
    source_id: str
    source_type: str
    item: ContextItem
    entities: tuple[str, ...] = ()
    knowledge: dict | None = None

    @property
    def event_time(self) -> str | None:
        return self.knowledge.get("observed_at") if self.knowledge is not None else self.item.source.observed_at

    def excerpt(self, query: str, *, max_chars: int = 900) -> dict:
        text = self.item.content
        positions = [normalized(text).find(term) for term in keywords(query)]
        first = min((p for p in positions if p >= 0), default=0)
        start = max(0, first - 200)
        end = min(len(text), start + max_chars)
        # Normalization is for ranking only; recovered passages are original text.
        return {"source_id": self.source_id, "source_type": self.source_type,
                "source": self.item.source.to_dict(), "event_time": self.event_time,
                "entities": list(self.entities), "excerpt": text[start:end],
                "start_char": start, "end_char": end, "total_chars": len(text),
                "truncated": start > 0 or end < len(text), "knowledge": self.knowledge}


def rank(hits: list[SearchHit], query: str) -> tuple[list[tuple[float, SearchHit]], str]:
    """Build an isolated FTS corpus *after* visibility and filter checks.

    BM25 never sees another audience's documents or term frequencies. This first
    implementation rebuilds the in-memory index per call; no stale ACL cache exists.
    """
    terms = keywords(query)
    if not terms:
        return [], "sqlite_fts5"
    with closing(sqlite3.connect(":memory:")) as conn:
        try:
            conn.execute("CREATE VIRTUAL TABLE recall USING fts5(content, tokenize='unicode61')")
            conn.executemany("INSERT INTO recall(rowid,content) VALUES (?,?)",
                             [(i + 1, h.item.content + " " + h.source_id + " " + " ".join(h.entities)) for i, h in enumerate(hits)])
            match = " OR ".join('"' + term + '"' for term in terms)
            scores = {row[0] - 1: -row[1] for row in conn.execute("SELECT rowid, bm25(recall) FROM recall WHERE recall MATCH ?", (match,))}
            backend = "sqlite_fts5"
        except sqlite3.OperationalError as exc:
            if "no such module" not in str(exc):
                raise
            scores = {i: float(sum(t in normalized(h.item.content + " " + " ".join(h.entities)) for t in terms))
                      for i, h in enumerate(hits)}
            scores = {i: score for i, score in scores.items() if score}
            backend = "keyword_fallback"
    for i, hit in enumerate(hits):
        if normalized(query) in normalized(hit.item.content) or query == hit.source_id:
            scores[i] = scores.get(i, 0) + 10
    ranked = [(score, hits[i]) for i, score in scores.items()]
    return sorted(ranked, key=lambda pair: (-pair[0], pair[1].source_id)), backend


def fingerprint(arguments: dict, requester: str, zone: str) -> str:
    return hashlib.sha256(json.dumps([requester, zone, arguments], sort_keys=True).encode()).hexdigest()[:24]
