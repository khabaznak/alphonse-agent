# Structured recall and source expansion

Issue #14, following [source contracts](08-memory-contracts.md) and
[durable knowledge](09-durable-knowledge.md). Both native operations are read-only
`memory_recall` capabilities, available without selecting a project. Jev receives
reviewed Noul relevance questions for search and source expansion; System Two
chooses queries, filters and source IDs. Neither model grants access or creates
fresh evidence from historical statements. Restart the runtime after registry changes.

## Search contract

`native.search_memory` returns a version 2 response with `results`, `status`,
per-source `coverage`, `backend`, `total_matches`, `offset`, `next_offset`,
`truncated`, `snapshot`, `timezone`, and the original query. Each ranked result
has its own stable `source_id`, source type, original excerpt and character offsets,
original source reference (including provenance, role and actual event time), score,
entities when known, and knowledge lifecycle metadata when applicable. Knowledge IDs
identify claims independently of their supporting conversation or task-source IDs.

The existing `matches_markdown`, `knowledge`, `knowledge_status`, and version 1
`recall` keys remain available. The old three-state recall envelope cannot represent
partial coverage: consumers must inspect version 2 `status`/`coverage` before
claiming an exhaustive result. Markdown labels historical evidence explicitly.

Status meanings:

- `found`: matching sources, with complete coverage of requested source types.
- `empty`: no matching sources, with complete coverage of requested source types.
- `partial`: some adapters worked, but another was absent, failed, returned only
  legacy bounded results, or FTS5 was unavailable and keyword fallback was used.
  Partial results can contain zero hits and never establish absence of a fact.
- `unavailable`: no selected adapter could run because backends were absent/offline.
- `failed`: no selected adapter worked because of execution errors, or ranking
  failed. Errors are sanitized; backend exception text and paths are not exposed.

Queries are bounded to 512 characters; result pages to 30 hits. Query syntax is
literal text, not raw FTS syntax. Unicode keywords use OR matching with BM25;
exact phrases and punctuation-bearing identifiers get priority. Accents and case
are normalized for matching, never in recovered original text. Stop words are
limited to a small English/Spanish list. This is lexical retrieval: it does not
translate or guarantee synonym/paraphrase matches.

Optional source types are `conversation`, `knowledge`, `task_evidence`, and
`legacy_ledger`. Up to eight entity filters use accent/case-insensitive literal
matching against text and declared knowledge subjects; these are filters, not
inferred entity relationships. Historical knowledge requires `historical=true` to
include superseded, expired or time-sensitive observations. Ordinary historical
messages and task audit attempts remain searchable regardless of this flag.

Date filters use observed/event time, never recording/import time as a substitute.
Unknown event times cannot match a dated query. A date-only start includes local
midnight; a date-only end includes that entire local day. Each boundary is converted
separately using the requester's configured IANA timezone, including DST. ISO
instants respect their offsets; offset-free datetimes use that same requester zone.
The response reports UTC when no timezone provider is installed. Source timestamps
remain original UTC instants. Task records with only a recording time remain
undated event evidence.

Every source adapter runs independently. Authorized conversation events, source-
resolved durable records, owner-scoped captured task evidence and authorized legacy
task audits compete in one ranking. A conversation match cannot hide other source
types. Conversation sharing is checked against current persisted grants; knowledge
also resolves its sources and transitive dependencies. Visibility and filters apply
before the per-request SQLite FTS corpus is built, so private documents do not even
contribute to BM25 term frequencies. No model call is involved.

This first implementation loads authorized source snapshots and rebuilds an
in-memory SQLite FTS5 index per request. It avoids stale ACL indexes but has linear
scan/index cost; large-history latency and memory use need measurement before a
persistent index or semantic index is chosen. Offset pages are sorted by score and
stable ID. Concurrent source changes can shift pages; the snapshot fingerprint
covers authorized content and filters so callers can detect changes and restart.
It is metadata, not an access credential or a persistent snapshot guarantee.

## Source expansion contract

`native.read_memory_source` accepts only a stable `source_id`, `neighbors` (0–5),
`offset`, and `max_chars` (256–12000). It returns original source content, source
metadata, character offsets, total length, `next_offset`, and truncation metadata.
Follow offsets to recover a whole long original passage. Neighbor excerpts have
individual source IDs so they too can be read in full. At most five authorized
messages on each side of a conversation anchor are returned, bounded to 1000
characters apiece. Owner timeline continuity crosses project and midnight boundaries;
private exchanges never become neighbors for another requester.

Knowledge reads resolve current support and return the claim's lifecycle metadata
and `supporting_sources`; read those source IDs for original quotations and context.
Task reads preserve attempts, errors and reported statuses as audit evidence. Failed
task evidence remains failed evidence. Memory does not prove external success.

An absent/deleted/denied ID returns the same `not_found` response. Offline adapters
return `unavailable`; execution failures return `failed`. No operation accepts a
filesystem path, and ID syntax does not encode a path to open. Legacy ledger IDs
are hashes of owner, project and relative location in the existing authorized audit
scope. Source reads rediscover that location from the scope and verify containment,
including symlinks. Legacy text and unknown times retain weaker provenance; a legacy
location is mutable and its stable ID is not an immutable content hash. No legacy
files are rewritten or migrated by search or reading. Old adapters that expose only
clipped search text report partial coverage and cannot promise source expansion.

## Evaluation and limits

Run the synthetic retrieval probe with:

```sh
python -m alphonse.agent_v2.evaluation.memory_recall \
  --corpus tests/fixtures/memory_recall/v1.json \
  --output docs/v3-hierarchical-capd/evidence/memory-recall-v1/results.json
```

The [retained report](evidence/memory-recall-v1/results.json) measures exact source
sets, including an excluded-day negative: identifiers 2/2, temporal 3/3, multilingual
2/3, paraphrases 1/2. No private source leaked. Cross-language translation and a
synonym-only paraphrase fail as expected. These ten small synthetic cases characterize
gaps; they do not select an embedding model or establish answer quality, production
latency, or rollout approval. No provider/model calls are made.

The frozen #11 baseline and #13 characterization remain unchanged. The current
context replay has a separate [search-v3 characterization](evidence/memory-search-v3/results.json).
Focused tests cover cross-source recall, visibility before ranking, revoked/deleted
IDs, adversarial adapters, containment, long quotation recovery, date boundaries,
source types/entities, pagination, explicit backend states, bounded schemas, and
project-independent deterministic tool reveal.

Verification: 202 focused checks pass across recall, contracts, knowledge, formatting,
conversation/ledger storage, replay/scoring, runtime registration and tool reveal.
The full Python suite reports 707 passed and 36 failed. An independent archive of
unchanged HEAD reports 685 passed and the identical 36 failure names; no new full-
suite failures remain. Those existing failures are not waived rollout gates.
Compilation and whitespace checks pass.
