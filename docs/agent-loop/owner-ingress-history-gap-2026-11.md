# Accepted owner requests and conversation history

Source review of the current owner-only simple loop. This is a design gap and
proposed contract, not an implemented retention or erasure guarantee.

## Current behavior

`SQLiteMessageQueue.enqueue_trusted_local_owner` authenticates the local peer,
rechecks the active canonical owner, and atomically saves the request prompt in
the inbound queue with a trusted-origin record. It does **not** write a
conversation event. `OwnerAtomicCheckpoint.acknowledge_desktop_draft` writes
both the user request and assistant answer/question in one claim-fenced
transaction only after Desktop returns the exact pickup acknowledgement token.
This acknowledgement confirms a client action, not that a person read it.

Consequently a cancelled, failed, abandoned, pending, or unacknowledged request
is not necessarily present in `v2_conversation_events` or searchable history.
It can remain in queue and checkpoint recovery state. Explicit abandonment may
delete the queue row; a tombstone-like abandonment marker and possibly a
checkpoint remain, but they are not a complete conversation-history record.
Desktop's pending state is in memory and does not by itself reconstruct all
unresolved requests after a restart. `tests/test_owner_daemon_end_to_end.py`
currently verifies that cancelled and pre-acknowledgement requests do not add
conversation events, while a successful acknowledgement adds both turns.

The two acknowledged conversation events are written through a shared SQLite
connection. This bypasses the post-commit daily Markdown projection in
`SQLiteConversationStore.record`, so the daily file is not guaranteed to
reflect current owner conversations. SQLite remains the authoritative source.

## Requirements before changing the boundary

1. In the same immediate transaction as trusted owner ingress, persist exactly
   one owner-scoped user event with stable source identity and the server's
   ingress timestamp. Check the peer and active canonical owner; a queue claim
   alone is not authority. A failed write must roll back both queue and event.
   An uncertain submission response must not lead to silent automatic replay.
2. Keep the assistant turn gated by the existing authenticated exact-token
   acknowledgement, verified claim, cancellation and evidence checks. An
   accepted request is not evidence of an answer, delivery or task completion.
   Define a versioned, auditable distinction for pre-change pending requests so
   acknowledgement retries cannot duplicate, replace, or resurrect events.
3. Exclude the current request from its own memory search **by durable event or
   source identity**, including neighbor expansion and revalidation. It is the
   task input, not independent historical evidence. Do not introduce phrase or
   language-specific rules. Preserve access checks for all other sources.
4. Define deletion across all copies before making an in-flight event deletable.
   Queue payloads, task checkpoints, selected observations, derived memory,
   generated ledgers, and backups can retain the content after a conversation
   event is deleted. In particular, checkpoints can remain after a successful
   acknowledgement, cancellation, failure, or abandonment. A queue-only guard
   is therefore insufficient. Reconcile retention and effect evidence without
   silently retrying, deleting uncertain action state, or claiming complete
   erasure. If deletion cannot be completed safely, expose a truthful pending or
   blocked state through an authorized interface.
5. Specify concurrency and recovery across owner cancellation, revocation,
   prompt deletion, claim loss, crashes, and possible external I/O. The queue,
   event, checkpoint and effect boundaries need deterministic checks in the
   transaction that changes them; model interpretation cannot grant access or
   establish completion. Older binaries and independent writers require
   explicit exclusion and cannot be fenced by new-code checks alone.
6. Make daily Markdown a recoverable projection with a durable dirty/rebuild
   strategy after committed events. A filesystem write is not atomic with the
   SQLite transaction; never acknowledge or retry the task merely because a
   projection failed. Any summarization must have its own authorization,
   budget, privacy and failure behavior.

Use disposable fixtures to exercise accepted, cancelled, failed, abandoned,
restarted, acknowledged and uncertain submissions; authorization changes;
deletion before/during/after pickup; duplicate acknowledgements; current-request
self-search and neighbors; stale claims; and projection failure/rebuild. Count
both queue and history records and assert no fabricated response, duplicate,
resurrection, unauthorized access or lost evidence. Check old in-flight records
separately; do not rewrite their stamps or replay effects.
