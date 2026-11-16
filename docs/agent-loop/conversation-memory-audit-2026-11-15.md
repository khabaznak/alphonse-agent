# Conversation and memory audit

Review of the local-first conversation and memory design against an
authoritative conversation record, source-linked derived memory, truthful
action history, rebuildable views, privacy and recoverability. This describes
source and disposable tests, not a qualified deployment or a full inspection
of household data. Existing household records are not migrated by this audit.

## Current shape

- `SQLiteConversationStore` keeps conversation events and their FTS search
  projection in the unified SQLite database. Messages have stable IDs, owner,
  role, timestamps, visibility, source and provenance. A daily Markdown
  projector and older task/project Markdown ledgers are separate files.
- `KnowledgeStore` holds derived claims, source references, access scopes,
  validity and dependency state in the same database in the normal daemon.
  Reads resolve source access and dependencies again. This does not mean all
  knowledge is exposed or refreshed by the owner-only simple loop: its memory
  tools currently select owner conversation sources, not general knowledge.
- Owner conversation pickup and exact-token acknowledgement atomically commit
  the displayed reply and checkpoint. An acknowledgement is a client receipt,
  not proof that a person read it. Queue, checkpoint, outbox and effect state
  remain distinct; there is no complete unified action audit for all old and new
  capabilities.

## Findings and work

1. **Conversation completeness:** The owner path previously recorded both the
   user request and reply only on acknowledged pickup. The new path records
   accepted requests with their trusted queue entry, independently of later
   failure or cancellation; the assistant response remains acknowledgement
   gated. Current-input self-retrieval is excluded and deletion is blocked
   while associated queue or checkpoint copies remain. Pre-change pending
   requests retain compatibility behavior, and complete erasure remains open.
   See the [ingress record and qualification contract](owner-ingress-history-gap-2026-11.md).
2. **Derived views:** Owner acknowledgements use a shared transaction and bypass
   the daily Markdown projector. The SQLite history is authoritative; daily
   files may be stale. Automatic daily Markdown projection is suspended in
   this working tree because existing files can be edited and other processes
   are not fenced. Existing files are preserved, and deletion is blocked when
   a daily directory requires reconciliation. A reliable repair needs durable dirty tracking or a
   restart-safe rebuild, access revalidation and a decision about when an
   external summarizer is permitted. Old binaries or external writers remain
   outside this protection. Do not add inference to the fenced write.
3. **Privacy and erasure:** Source authorization and deletion are rechecked on
   memory retrieval, but source-linked payloads, task evidence, checkpoints,
   historical files, logs, SQLite WAL and backups have different lifecycles.
   Removing a conversation does not by itself prove complete erasure across
   them. Derived records cannot widen source access; search must also verify
   the authoritative event owner rather than relying on the FTS projection.
   This change verifies the owner, removes eligible same-database single-source
   knowledge transactionally, and rejects stale reintroduction under the
   database write lock. Mixed-source claims, dependencies, conflict/supersession
   links and unknown knowledge payloads block deletion for reconciliation rather
   than discarding independent facts. Existing index deletion errors roll the
   transaction back. These
   protections do not erase every copy, and surviving conflict/supersession
   references can require reconciliation.
4. **Memory lifecycle:** Schema supports source-linked claims, timestamps,
   validity and supersession. Current owner conversations do not establish
   reliable automatic extraction, correction UX, lifecycle controls or general
   multilingual recall. Recorded or generated memories are not authority for
   a tool effect. Missing or revoked source evidence must withhold a claim.
   Conversation-derived knowledge writes require the same database as their
   source; independently allocated in-memory stores fail closed.
5. **Recovery:** The offline memory backup uses SQLite's backup API and
   caller-supplied file roots; restore targets a new directory. It is not a
   complete household backup: root completeness, writer quiescence, encryption,
   authenticated manifests, and reconciliation of newer deletions or grants
   are not guaranteed. A previous backup can contain content later deleted.
   Existing migration and rollback limits are recorded in
   [the transition inventory](retirement-transition-2026-11-15.md).

No blanket append-only promise is made: privacy-preserving deletion and
correction require lifecycle controls across authoritative and derived copies.
Historical Markdown and independent backups must not be deleted or rewritten
by guessing which copy is authoritative.

The combined disposable conversation, memory, migration, owner daemon,
restart and claim-recovery suite passed 261 tests. Three historical daily-view
tests are skipped while projection is suspended. These results do not qualify
production erasure, household backup restoration or older-writer exclusion.
