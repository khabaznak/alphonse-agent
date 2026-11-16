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

## Source snapshot binding foundation

Newly extracted records use version 1 source bindings: a SHA-256 digest of a
canonical, complete `ContextItem` snapshot, including its source metadata. The
input snapshot is captured before extraction; extraction and Jev review operate
on bounded chunks, and this does not mean every byte was reviewed. Supporting
records retain their previously captured bindings, and unverified dependencies
cannot be used to create a bound record. Before committing a record or an empty
extraction receipt, the source snapshot is checked in the same database write
transaction. Reads withhold bound records when a source, access metadata, or
binding changes. Processing receipts still use source ID and extraction version;
an edit does not trigger automatic re-extraction.

Older records remain readable under the existing quote checks and are explicitly
marked `unverified`. They are not retroactively bound to today's source. Unknown
or malformed bindings fail closed. Binding data stays in the private database
storage envelope; public projections and model inputs omit digests, while exposing
only the snapshot-bound or unverified state. Hashes are sensitive derived data,
not encryption or authentication against someone able to rewrite the database.
Older binaries may not understand new fields; mixed-version operation and
backup/rollback recovery need a separate compatibility plan.

This is an identity and staleness check, not proof of truth, consent, semantic
entailment, or that a trusted internal caller actually invoked review. It does
not enable owner LLM retrieval or resolve provider disclosure, revocation after
dispatch, or cancellation races between the final check and transaction. Tests
use disposable sources and fake model transports; no household data or provider
is involved.

## Claim-bound snapshot and checkpoint groundwork

The unregistered snapshot now also has a transaction-scoped entry point for a
caller already holding the trusted owner claim. Its private digest covers the
selected records and their source bindings; a public projection does not contain
the digest. Experimental checkpoint settlement requires the digest and checks
the snapshot and public projection in the same transaction. Recovery reconstructs
and checks both, and desktop pickup compares the private binding again while
holding the pickup transaction. These checks do not establish that a claim is
true, and a different source binding is rejected even if displayed claim text is
unchanged. The helper itself does not establish caller authorization.

An opt-in test host can now use a dedicated private-result path; the default host
leaves it off and no runtime registers the tool. The ordinary invocation path
redacts the tool's result; the private path calls the pinned implementation
without emitting the result to Desktop events, memory or telemetry. Synthetic
provider tests exercise projection, recovery and source changes during inference.
This is not authorization to disclose household knowledge to a configured
provider, and it does not establish production readiness. More qualification is
needed for recovery, cancellation and uncertain calls across the full runtime.
Checks are point-in-time: revocation cannot retract data already dispatched to a
provider.

## Disclosure grant storage foundation

An unregistered storage primitive can hold one explicitly issued owner grant
for an exact provider, endpoint, account and model identity. No grant exists by
default. Each replacement gets a new opaque grant ID, so revoke and regrant do
not restore previously recorded authority. A locked check can compare a saved
binding while the caller holds the same database transaction as pickup. It also
requires the canonical owner to remain active. Changing a provider identity or
revoking the grant fails the check.

This primitive is not an authentication or consent interface: no runtime or
Desktop entry point uses it, and its management caller must authenticate the
owner and obtain explicit authorization. Future integration must derive a
non-secret destination identity from trusted provider configuration, verify that
it captures account and routing changes, and check the binding at dispatch,
recovery and pickup. The storage check alone cannot validate a provider's actual
network destination or retract data after a provider call starts. These gaps and
in-flight semantics must be resolved before enabling the feature.
The constructor flag is a test/rollout gate, not a revocation control: turning it
off does not invalidate an already selected draft. A production switch needs an
independent trusted revocation check at recovery and pickup before activation.
