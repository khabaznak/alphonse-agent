# Owner knowledge retrieval boundary

Review for #229 and #230. This is a design and qualification note, not an
enabled capability. The owner-only simple loop currently reads authorized
conversation sources; it does not expose the general `KnowledgeStore` or run
automatic knowledge extraction. No household data or model provider is used
by this review.

## Existing source freshness

`KnowledgeStore.records` now requires each live source to retain the supporting
quote before returning a record. The check applies recursively to dependencies;
the same check is used for source availability during related-record review.
Malformed source envelopes and content fail closed. It complements the current
access checks; it does not make the derived claim authoritative or current.
Keeping an identical quote is not proof that surrounding context, meaning or
full source content is unchanged. The check is a literal evidence-presence test,
not a semantic truth judgment or a source-version guarantee.

## Requirements for a future owner knowledge path

1. Expose a narrowly scoped read-only service, not generic `LedgerMemory` or
   its task-event and extraction methods. Recheck canonical active owner,
   same file-backed database, live claim and cancellation. Each returned claim,
   its sources and every transitive dependency must remain owner-owned and
   currently accessible. Exclude the current request by claim-bound identity.
   Do not promote a stored preference into authorization for another tool.
2. Validate status, validity interval, supersession, conflicts, quotes,
   provenance, schema and dependency graphs. A disputed claim must stay visibly
   disputed; a missing or revoked source and unknown schema fail closed. Bound
   traversal and output; exceeding a bound is an explicit incomplete result,
   not evidence that nothing else exists. Do not expose another member's
   record or let shared-source labels silently widen this owner's access.
3. A persisted observation must bind the exact selected record and its source
   versions. Reconstruct and reauthorize it for recovery, for every projection
   into inference, and at publication. Pickup must verify the same binding
   using the claim/checkpoint transaction and same authoritative database;
   a check performed only before acquiring that transaction leaves a race.
   Stale or missing evidence is not silently refreshed into a different answer.
4. The external inference call is a separate disclosure boundary. SQLite cannot
   atomically revoke access to data after it has entered an external provider
   call. Holding a database write lock across an unbounded model call would
   harm cancellation and availability and still would not retract sent data.
   Define the user's provider and data-disclosure authorization, its timing,
   cancellation and in-flight semantics explicitly before enabling this path.
   A point-in-time access check is not a guarantee against later revocation.
5. Test current, deleted, edited, revoked, shared, expired, disputed, superseded,
   malformed and oversized records; missing and cyclic dependencies; current
   request exclusion; actor/claim changes; source changes between validation,
   inference, pickup and recovery; and lost responses. Use disposable data
   and independently count accesses. No test should imply real delivery or
   model semantic correctness merely because a claim was retrieved.

## Full source-version binding remains open

An optional digest for **new** derived knowledge could detect changes outside a
quoted span, but it needs an explicit versioned contract before implementation:
identify the authoritative full source bytes and canonical encoding; bind every
source and dependency under a consistent snapshot; decide handling of source
edits, revocation and re-extraction; and prevent a digest from widening access or
leaking private information through prompts, logs or exports. A digest proves
identity of bytes, not truth, consent or semantic entailment.

Older knowledge has no such binding. It must be marked as unverified with respect
to full-source version and either withheld or reconciled by a deliberate policy;
it must not be retroactively labeled verified by hashing today's source. Schema
evolution, historical records, backups, deletion and in-flight checkpoints need
an upgrade/recovery strategy. No digest field or automatic migration is added
in this slice.
