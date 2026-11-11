# Browser workflow state: integration prerequisites

Status: disabled-by-default browser workflow foundation. Native workflow tools
are registered disabled, and no household runtime supplies their service or a
browser backend. Chrome connection and user-facing enablement belong to #185.

The experimental browser implementation for #185 is not yet on main. Its current
session and protected-data handles are process-local, and its worker clears private
data when it ends. A checkpoint must therefore never treat a saved handle as proof
that the corresponding session, value, observation, or grant is still available.
Integrate and test this contract against the actual browser implementation once
that implementation's lifecycle and checkpoint interfaces are stable.

The foundation includes the experimental branch's actor grant store, task-scoped
session fence, and bounded volatile store, a versioned record, and an opt-in coordinator. It binds
the live session to the actor's grant revision and a transport-supplied session
identity. Private Desktop questions retain only a protected answer reference in
the question store and queued resumption. Disposable fixtures run a fill through the V3
executor and real SQLite checkpoint store using a synthetic transport, including
process loss after dispatch and recovery without a repeated fill. The browser
checkpoint writer fences terminal status, owner mismatch, stale checkpoint
versions and stale workflow records within one SQLite transaction. This does not establish safety of every
other checkpoint writer or a usable browser integration. Fill and submission
require a live protected observation and a transport that can fence the action
against that observation. Later observations can support independent verification
only through a trusted backend that implements the relevant verification
contract; only disposable backends implement these contracts today. A session
can be reacquired through an explicit task-owned attachment, but unresolved
effects from an earlier session remain blocked. Do not expose this foundation
to the model or connect it to a real browser until a backend and its effect
verification are qualified against the real integration.

The main-branch executor checkpoints tactical action arguments before dispatch
and the invocation service emits arguments and exceptions to memory, UI events
and telemetry. The prototype redacts sensitive-tool action and invocation
records, including rejected actions, and tests a fill through those surfaces.
Further review is needed for model prompts, arbitrary follow-on tools, other
checkpoint writers and handoffs before enabling browser access. The experimental
#185 branch has relevant changes to review; their presence alone is not evidence
that all leakage paths or cancellation races are covered.

## Durable facts and live authority

Use a versioned, bounded, strictly validated record bound to the task and actor.
Record only enumerated field categories, opaque runtime-generated identifiers,
timestamps, phase/action provenance, and typed evidence states. Do not put URLs,
selectors, page text, form values, screenshots, or model-generated descriptions
in this record. Do not use hashes of low-entropy private values as identifiers.
Preserve the distinction between a requested value and a supplied value, and
between a call attempted, a call returned, and an effect independently observed.
In particular, a successful click or fill does not establish submission or a
verified result. An opaque reference is evidence of a past allocation, not an
authorization or a guarantee that the referenced object remains available.

On every use, resolve references through the live store under its current actor,
task, grant, expiry, cancellation, and parent-observation checks. Re-observe the
page after any operation that may change it, including an ambiguous failure.
Bind observations and verification to the active session and page generation;
stale observations cannot justify a new mutation or result. Keep historical
facts separate from currently usable handles in planning projections.

## Recovery and checkpoint ordering

Before an external effect, durably record an attempt under the existing phase,
mutation, and cancellation gates. If execution or checkpointing is interrupted,
the effect is unknown. Reconcile using a fresh authorized observation or request
human help; never retry an uncertain external effect solely because its result
checkpoint is missing. Persist completion only with its evidence and provenance.
Browser-specific state must not change task or phase terminal states, resurrect
cancelled work, bypass prerequisites, or grant access across actors.

After restart, worker/session loss, expiration, or revocation, retain the safe
historical facts but mark corresponding handles unavailable until the live service
revalidates them. A new session requires explicit reacquisition and cannot inherit
the old page generation. Unknown schema versions, malformed or scope-mismatched
records, and legacy browser checkpoints need a safe reconciliation path rather
than assuming an empty workflow or a successful prior effect. Any migration must
preserve existing terminal and cancellation state.

## Integration and verification

Wire recording to the real browser service and private handoff lifecycle, with
checkpoint writes ordered around actual side effects and question creation. The
experimental implementation currently exposes process-local `BrowserSession`,
`BrowserPrivateData`, `BrowserService`, and `browser_handoff` operations; their
interfaces may change before #185 lands. Ensure planning and handoff projections
consume the safe state without serializing private tool results or bypassing the
existing authorization and effect gates.

Use disposable synthetic pages and inputs to test partial fills, changed pages,
expired values, grant revocation, process restart, session loss, cancellation,
unknown effects, legacy records, and actor mismatch. Inspect actual persisted
checkpoint bytes and captured logs for sentinel private values and page content.
Tests must exercise the production checkpoint, recovery, and browser execution
paths; a standalone state serializer is not sufficient qualification.
