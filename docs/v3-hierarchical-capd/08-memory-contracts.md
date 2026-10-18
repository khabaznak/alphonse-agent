# ADR: Canonical history and source-linked memory contracts

Status: accepted for the contract layer, 2026-10-17. Implements #12 under #10;
prerequisite #11 is complete. Retrieval/assembly policy changes and rollout remain
subject to the [evaluation protocol](07-memory-evaluation.md).

Follow-up #13: [durable person/audience knowledge](09-durable-knowledge.md) adds
version 3 knowledge records and persistence while retaining these version 1 source
contracts. It replaces project-summary promotion and task-context use of prose
summaries. The scope descriptions below describe the original #12 delivery.

Follow-up #14: [structured recall](10-memory-recall.md) adds version 2 search/read
responses while preserving the version 1 source references and compatibility
recall envelope below. It replaces the first-hit-source short circuit with combined
ranking and explicit coverage states.

## Decision and ownership

SQLite `v2_conversation_events` owns canonical user/assistant messages. Its
`event_id` is immutable; `conversation:<event_id>` identifies that source in reads,
recall and context assembly. Provider `source_message_id` is a delivery deduplication
key, not a canonical identity. Sequence is the insertion order within this store;
it is never a timestamp or an ordering guarantee across independent databases.
Project and memory-session associations are optional metadata. Neither selecting
a project, ending a task, nor midnight starts a new canonical timeline.

Daily Markdown and its summary files are disposable projections. They must never
be reimported as new messages. The legacy importer accepts explicitly supplied
pre-timeline task-ledger input only. Projection and rebuild paths only read events;
repeated rebuilding cannot insert messages or change IDs. Display dates use the
member's current timezone. Unknown observation times are excluded from dated views
and remain available through owner export, recent sequence history and ID reads.
An unknown date is never replaced by the import date.

Task audit ledgers and `TaskState.evidence_journal` own execution evidence, including
failures, attempts and verification results. They are not conversation messages.
`TaskState.evidence_items()` exposes owner-scoped, versioned audit items with stable
`task-evidence:<task-or-message-id>:<journal-index>` identities; each attempt remains
distinct even when the old `tool-call:` label repeats. Original journal entries and
checkpoint serialization are unchanged. The source task ID must travel with an
export; a journal index alone is not globally unique.

Task checkpoints and their plans, acceptance contracts, bindings, facts and pending
questions own active working state. They may refer to history and audit evidence;
restoring a checkpoint does not turn it into either a new message or durable truth.
Task facts and legacy Markdown are not silently upgraded into typed knowledge.

`DurableMemoryRecord` defines derived knowledge with its own stable memory ID,
content, owner/audience, supporting source references, recorded/observed times,
optional validity interval and supersession link. This issue supplies its envelope
and access/resolution contract; it does not introduce an automatic extractor or a
new durable-knowledge database. Original sources continue to own the evidence.
Corrections create new knowledge/source records and reference what they supersede;
older statements remain historical evidence, never proof of current conditions.

## Version 1 envelopes

`core/memory/contracts.py` owns the typed contracts and JSON round trips:

- `AccessScope`: a nonempty owner and either private visibility or explicit shared
  recipients. There is no implicit household-wide or administrator model access.
- `SourceReference`: namespaced identity, source kind, access, source sequence,
  speaker account when actually known, role, channel, recorded time, observed time
  and provenance. Role is not a verified human identity.
- `ContextItem`: a source reference plus bounded content and a representation
  (`message`, `audit`, `working_state`, or `derived_knowledge`).
- `DurableMemoryRecord`: source-linked knowledge and temporal validity as above.
- `RecallResult`: a query, `found`/`empty`/`unavailable` status and context items.
  A found result has items; the other states do not. All timestamps are ISO UTC
  instants when known; optional unknown values are null in references.

Persisted conversation rows carry schema version, visibility/recipients, actual
recording time, observed time, speaker ID and provenance. `created_at` remains a
compatibility field so migration can retain the original value. New channel
records retain the existing delivery source as channel and identify the owner as
speaker only for directly recorded user turns. Assistant turns identify Alphonse.
The additive migration cannot verify old speaker/recording metadata, so leaves
those unknown and marks old rows `legacy_unverified`. Recognizable ledger imports
are `legacy_import`; their previously assigned dates remain recoverable in
`created_at` but are not trusted as observation times. The original channel string
is retained for recovery, not presented as verified imported channel provenance.

New legacy imports record the actual import time separately, preserve unknown
observed dates and speaker IDs, and use ledger provenance for both user and
assistant text. Explicit role labels recover roles only. Malformed or timezone-free
dates remain in compatibility data, not in known observation instants. Import
markers and provider IDs continue to prevent repeated canonical insertion. Original
Markdown and task evidence files are not modified or removed by migration.

## Reads, assembly and deterministic boundaries

Owner list/search APIs remain owner scoped. `read(event_id, requester_user_id)`
also permits an explicit recipient of a shared event and otherwise returns no
content. Sharing is representable and persisted; there is no automatic cross-owner
search or new sharing-management UI in this change.

`conversation_item()` validates the version and access metadata before extracting
text. Missing/unsupported metadata fails closed, including alternate backends.
Transcript assembly and native recall both use this gate. `visible_unique()`
filters access before deduplicating canonical source IDs. Assembly records selected
references in `task.metadata.conversation_sources`; native recall adds the same
references in the structured `recall` envelope while preserving the existing
`matches_markdown` compatibility field. Under #15's
[continuity projection](11-conversation-continuity.md), the assembly manifest
describes complete records actually included. Separate projection metadata records
omissions; message text is never sliced to fit the budget.

Legacy ledger search remains a compatibility adapter in the task's existing
native authorized scope. Its opaque result gets a scoped content-snapshot hash
and `legacy_unverified` audit provenance, not invented canonical event IDs, speakers
or observation dates. This ID identifies that snapshot, not a citation to an
individual message or a stable file offset; it cannot deduplicate distinct
legacy excerpts as if they were canonical messages. The structured result separates
empty and unavailable conversation search when the fallback has no evidence.
The old Markdown/error behavior is retained; downstream routing changes belong
to the search-state follow-up. Sentinel strings such as “no matches” are not evidence.

Jev and System Two receive only content that has passed deterministic visibility
checks. Jev may assess relevance; it cannot widen access, turn an old permission
into current authorization, or turn failed task evidence into success. No model
calls are required to enforce these contracts. Existing task/project authorization
continues to govern raw legacy task context and native tool operations.

A durable source envelope is not an access credential. Before exposing a stored
derived record, use `resolve_durable_memory()` with a resolver that reads each
source under the current requester. A missing/deleted/denied source withholds the
whole derived record; matching an old cached audience is insufficient. Materialized
knowledge stores added later must implement this boundary before model submission.

## Schema evolution and migration recovery

The envelope schema version is independent of the SQLite layout and task engine
version. JSON readers require an explicit supported version, reject unknown fields
through typed construction and reject unknown versions rather than interpreting
them as current records. Adding a semantic field, changing identity meanings or
changing temporal/access interpretation requires a new version and explicit adapter.
Preserve old payloads and the frozen synthetic evaluation evidence. Do not fill
missing provenance by model inference.

The SQLite migration adds metadata without replacing IDs, content, delivery source,
project/session associations, task ledgers or task evidence. Reopening is idempotent.
Older pre-sequence stores retain the existing deterministic created-time/row-order
migration; canonical IDs survive that conversion too. As before, offset timestamps
may normalize to the equivalent UTC instant. Back up a production database and
user-owned files before migration: rolling back binaries does not undo schema
changes. Export full owner history with `list_all()` for recovery/rebuild, never as
an unbounded model prompt.

## Retention, erasure and historical references

Retention policy is explicit owner/admin product policy, not a Jev judgment. This
change adds no automatic TTL or scheduled purge. Deleting a canonical event requires
its exact owner ID. It removes the event and FTS text, retains an ID/owner/deletion
instant tombstone plus a hash of the delivery key, and blocks redelivery from
resurrecting that source. A historical ID read returns no content after deletion;
callers must not guess a replacement based on similar text. Denied and missing
reads are intentionally indistinguishable to an unauthorized caller.

With a daily projector attached, deletion invalidates that owner's generated views
before committing erasure, then rebuilds from remaining events. A rebuild failure
is reported and leaves no old generated text. Owner rebuild is also available
independently, discards stale summaries and retains canonical IDs. Summary cache
fingerprints cover the exact source rows, so changed source sets invalidate cached
summaries before reuse. Rebuilds can regenerate narrative wording; only canonical
messages have stable content/identity.

A future retention coordinator must also invalidate dependent knowledge, task
checkpoint context, externally cached prompts, exports and backups under their
separate policies. The resolver boundary already withholds derived records whose
sources no longer resolve. Canonical deletion does not silently erase independently
retained task audit/evidence or user-owned legacy input; those require their own
explicit retention/deletion scope. Tombstones retain no message content. Physical
SQLite free pages/WAL and backup erasure require operational secure-deletion and
backup retention procedures; logical source deletion is not a claim of forensic
media erasure. No success claim should imply those independent copies were purged.

## Validation and remaining rollout work

Synthetic tests cover JSON/version boundaries, source identity across read/search/
assembly, private/shared visibility with adversarial backends, midnight/project
continuity, weak imports, migration/restart preservation, repeatable rebuilds,
erasure/FTS/redelivery, failed projection recovery and distinct recall status.
The frozen #11 replay remains a characterization check. This contract change does
not claim to fix implicit recall, remove irrelevant context, meet token-reduction
thresholds, or establish end-to-end task completion. Those require subsequent
assembly/retrieval work and the separate rollout evidence defined in #11.

Implementation verification (2026-10-17): 116 tests pass across
`test_memory_contracts`, conversation storage, daily/legacy ledgers, task state,
V3 context, replay and scoring. The frozen #11 report still matches byte-for-byte
JSON semantics, with no provider/network calls. A broader V3 integration run passed
80 tests and exposed three pre-existing failures; the memory-session suite exposed
one more. Each was reproduced independently from an archive of unchanged HEAD:

- `test_router_stamps_selected_session_before_queueing_and_supports_commands`
- `test_direct_reply_receives_cross_project_history_global_and_selected_project_context`
- `test_context_curation_separates_invariants_from_optional_candidates`
- `test_plan_intake_batches_eligible_messages_and_consumes_only_jev_selected_ids`

Those failures are not waived rollout gates and were not changed in this issue.
