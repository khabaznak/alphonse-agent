# Durable person/audience knowledge

Implements issue #13 after the #12 source contracts. Knowledge is independent of
project selection. Project IDs are optional associations, never retrieval or
visibility predicates. Conversation events remain canonical messages; task audits
remain execution evidence; skill instructions retain artifact operating expertise.

## Records and ownership

`core/memory/knowledge.py` introduces `KnowledgeRecord` schema version 3. Nested
`SourceReference`/`AccessScope` envelopes retain their independent version 1. A
record includes its memory ID, kind (fact, preference, instruction, decision,
commitment, observation), subjects, claim key, content, exact supporting quotes,
source references, audience, recording and observation times, validity interval,
persistence intent, lifecycle, reported commitment state, correction/conflict links,
transitive knowledge dependencies and optional project,
task and schedule associations.

Version 1 `DurableMemoryRecord` payloads remain readable under their original
contract. `legacy_envelope()` produces an explicit reconciliation input, not a
fabricated upgrade: missing attribution, intent and claim identity are unknown.
Unknown versions fail closed. The earlier, uncommitted v2 knowledge draft did
not encode complete support dependencies. Those payloads, if encountered, are
preserved but withheld, not upgraded by inventing missing provenance; they do not
prevent supported v3 rows from being read or written. `project-summary.md` is retired: there is no writer,
reader, importer, or planned reconciliation workflow for it. Projects describe
bounded efforts through their objectives, goals and related project metadata;
they do not own a separate narrative memory. Existing obsolete files are ignored
and are not needed for operation or recovery. This change does not scan household
storage to delete them. Session closure and legacy-ledger archival do not produce
project summaries. Session summaries/raw task audits do not feed durable prompt
context either.

`KnowledgeStore` uses additive SQLite tables in the conversation database:
`memory_knowledge`, `memory_processing`, `memory_task_sources`, and
`memory_task_state`. Immutable record payloads preserve original content and
sources; separate lifecycle/relationship columns project current status. A
correction creates a new record, marks the previous record superseded, and
preserves the previous source and claim. A correction requires the original
reporter's own newer conversation evidence, the same audience, subjects and claim
key, and an explicit quoted correction or commitment-state update reviewed by Jev. It cannot retract another
speaker's claim or reactivate a superseded record. Different wording alone never
creates a dispute. Jev compares authorized same-key pairs with bounded Choice
answers: equivalent, compatible, conflicting, or uncertain. Only a reviewed
conflict creates dispute edges; equivalent paraphrases and compatible historical
observations remain independent attributable reports. Uncertain, missing or
malformed comparisons defer the event. All comparisons run outside the write
transaction; if another writer inserts an unreviewed same-key claim, the transaction
rolls back for retry rather than guessing. There is no newest-report-wins rule.

All new automatic writes are private to the authenticated source owner. A model
cannot choose an audience or write another person's knowledge. The typed store
can represent explicitly shared records from a trusted writer; this issue adds no
sharing-management UI or automatic household-wide sharing. Every supporting source
must permit every member of the derived audience. Reads re-resolve live sources
before ranking, truncation, or model submission. Deleted, denied, unknown-version,
or audience-restricted sources withhold dependent records. Memory never changes
native tool authorization or mutation policy.

## Processing and model responsibilities

`LedgerMemory.start_task()` processes new conversation evidence before ordinary
conversation/task reasoning, without a project or session prerequisite.
`finish_task()` also processes steering/new conversation events and captures the
task's immutable `evidence_items()` projection. Persisted journal entries retain
`task-evidence:<task-id>:<index>` identities and reject identity/content reuse.
Captured task state is a last-recorded checkpoint, not independently verified
external state. Only the real task lifecycle updates that checkpoint.

Each pass prioritizes the latest 16 events and advances a chronological backlog
by at most 16 more, using persisted receipts/cursors. This is bounded catch-up on
conversation/task activity, not an independent background sweep of inactive users.
Recent events are chronological within the selected window. Backfilled older
claims may create disputes; they cannot supersede newer evidence. Receipts are
versioned by owner/source. Extraction/review failures retain pending work; failure
on one recent event does not prevent other recent events from progressing. A
failed chronological event retains the backlog cursor for retry.

System Two performs bounded extraction/synthesis using `MEMORY_EXTRACTION`.
Existing candidates are access-filtered and limited to a 16,000-character record
budget; each source chunk is at most 12,000 characters, with overlap for long
messages. All chunks must finish before the event's records/receipt commit.
Candidate IDs are deterministic and duplicate overlap candidates are collapsed.
No assistant prose, imported legacy prose, summaries or task-success labels are
eligible as verified external truth. User messages remain user reports. Journal
results, including failed attempts, can only become historical tool observations.

Before extraction, Jev selects which prior claims are necessary factual support
for a contextual source such as "remember that number". System Two receives only
selected full claims plus relation metadata for other candidates. Relation IDs,
keys, subjects and status are not evidence for a new fact. Code automatically
attaches every selected supporting record and all its transitive source references
and exact quotes, even if the extractor omits a dependency. Unrelated prior claim
content is withheld from the extractor. All supporting sources are rechecked after
model calls and before persistence. Dependency access/lifecycle is rechecked at
read time; deleted/restricted sources, absent records and superseded/expired
support withhold current derived claims. Disputed support propagates dispute.
These dependencies never authorize an action or turn a report into verified truth.

Jev receives parallel Noul questions over explicit candidates: support, persistent
intent, temporal classification, correction/conflict relations, and exclusion of
artifact expertise or permission grants. It does not generate prose or call tools.
Missing/malformed/ambiguous judgments defer the event. Negative judgments discard
that candidate. With System One disabled or unavailable, no unchecked automatic
writes occur; sources remain available and pending. There is no extra LLM
classification fallback. System Two extraction is skipped when review is disabled.

Code validates source ownership, exact quotes, explicit preference/instruction
intent evidence, valid timezone-aware intervals, source-backed relation targets,
audience containment, schedule existence/ownership, budgets and current source
availability. Cancellation is forwarded to extraction and checked before commits.
Receipt insertion, all new records and all lifecycle transitions share one
`BEGIN IMMEDIATE` transaction. Replay/restart/concurrent writers cannot duplicate
an event's records or partially apply a correction. Knowledge failures are logged
and leave task processing operational; task metadata reports pending updates or
unavailable reads rather than falling back to prose promotion.

## Retrieval, freshness and commitments

Task context retrieves authorized knowledge across projects using bounded lexical
matching; explicit persistent preferences/instructions remain eligible even when
request words differ. This is an initial retrieval policy, not a claim of complete
semantic recall. Subjects/key normalization and synthesis are model judgments;
code cannot prove natural-language equivalence. Retrieval/ranking improvements
and rollout remain subject to #11's evaluation gates.

Memory formatting is shared by task assembly, planner tool curation, aggregate
prompt budgets and Desktop progress. Oversized knowledge is omitted at complete
record boundaries; no downstream character slice can leave a claim body without
its status, qualifications or source labels. Multiline claim text renders as one
record line to prevent embedded text from forging another record boundary.
Oversized unstructured legacy memory is omitted rather than sliced.

Default knowledge recall excludes superseded records, all observations/tool
observations, expired intervals and future intervals. Disputed current claims stay
visible as disputed. Unknown observation times remain unknown. A record is never
fresh current-state evidence merely because it was retrieved or its validity
interval is open. Current-state requests still require native reads. Native
`search_memory` adds an explicit `historical` boolean for old records/observations,
returns knowledge metadata plus source-linked recall items, and labels canonical
message excerpts as historical evidence. Empty and unavailable knowledge reads
are distinguishable. Raw legacy task-audit fallback retains its existing
project authorization boundary; it is not silently widened to other projects.

Commitments never create or mutate schedules. A schedule association must resolve
under its owner's current access, and rendering reads the current scheduler state
(active/cancelled/etc.). Missing/deleted schedules are unavailable, never assumed
complete. Unscheduled commitments retain no schedule ID. Task links, where
present, expose only the last actual task checkpoint with its recording time;
that is distinct from live scheduling state. Completion of an extraction or task
cannot mark the remembered commitment complete.

## Retention and validation

Source deletion immediately withholds derived records through live resolution,
including in extraction context. Persisted knowledge text and the separately
retained task-evidence copies require their own explicit retention/erasure policy;
this issue does not claim physical erasure of all copies or backups. No legacy
summary is automatically reimported or rewritten. Retired project-summary files
are not part of the recovery contract. Existing source IDs/history survive
migration and rollback.

Synthetic tests cover correction history and restart, disputed claims, versioned
round trips, private/shared source access and revocation, source deletion during
extraction, stale/unknown observations, explicit intent, authoritative schedule
state, invalid model outputs, retry/rollback/idempotency/concurrent writers,
long-message bounds, cancellation, runtime settings changes, and no-project /
cross-project context. Jev tests cover bounded parallel questions and malformed or
ambiguous probabilities. Provider tests are stubbed; no household records or live
provider calls are used.

The frozen #11 corpus and baseline stay unchanged. The context/native-search
adapter advances to v2 because prompt labels now explicitly distinguish household
knowledge and historical messages. Its separate
[evidence snapshot](evidence/memory-knowledge-v2/README.md) measures those runtime
changes only, not end-to-end extraction quality or rollout readiness.

Implementation verification (2026-10-17): 123 focused knowledge, formatting,
source-contract, replay and scoring tests pass, plus 19 Jev memory decision tests.
The full Python suite passes 685 tests and fails 36. All 36 failure names match an
independent archive of unchanged HEAD (607 passed, 36 failed); there are no new
full-suite failures. Existing routing, project, prompt and curation failures are
not waived rollout gates. Whitespace and Python compilation checks pass.

The three pre-commit review reproductions now have regression coverage: hidden
support dependencies survive synthesis and cause withholding on deletion or
revocation; all prompt budgets preserve complete labels/claims/citations; semantic
equivalence prevents spurious disputes while uncertain/racing comparisons defer
writes. Source dependency schema v3 also preserves unsupported earlier draft rows
without exposing them or inventing missing provenance. No live provider quality,
latency or production rollout approval is claimed by these synthetic tests.
