# Uncertain delivery reconciliation: proposed first operation

Status: design and diagnostics for #224 and #225, not an implementation or
permission to send. The candidate has no production delivery action, authorized
recovery endpoint, claim-fenced publication or operation-bound provider receipt.
It remains off.
This document narrows the general evidence requirements in
[24](../v3-hierarchical-capd/24-single-loop-evidence-boundary.md) to one possible
operation: a message delivery to one authenticated, authorized recipient.

## Existing boundary

`ExperimentalCheckpointStore.reserve` persists a reservation ID, kind and budget,
but no operation binding. The separate unwired `reserve_delivery` experiment
adds a claim-bound delivery action, recipient, channel, provider, message digest,
idempotency key, grant ID and connection/grant revisions. These caller-supplied
identifiers are not an authorization check, the key is not globally enforced,
and the message must be retained by a future host. The v2 binding does not infer
missing authority identifiers from older v1 records. `settle` deliberately rejects all tool settlement,
including one that cites an actually delivered outbox ID. The evaluation-only
synthetic settlement uses forgeable fixture tables, not product authority or
provider evidence. Neither path establishes a real delivery. The read-only
recovery inspection does not authenticate callers, resolve effects or release
the marker.

The legacy outbox has another independent problem. A timed-out ordinary send
can enter `retry_wait`, and the next claimant may repeat a send that the provider
accepted before its response was lost. Scheduled sends instead become
`unconfirmed` on failed/expired dispatch. The disposable
`tests/test_effect_reconciliation_diagnostics.py` demonstrates these differences;
neither behavior proves safe recovery for the candidate. Current settlement by
outbox message ID can also accept a stale worker's acknowledgement after reclaim
(`tests/test_claim_fenced_outbox_diagnostics.py`).

## Proposed contract

1. **Before dispatch**, in the same durable, actor/checkpoint/claim-bound
   transition as the budget reservation, persist a unique action attempt and
   immutable binding: authenticated task owner and relevant queue respondent,
   task/engine/schema, operation and canonical arguments or digest with durable
   inputs, exact message/recipient/channel/integration, intended effect,
   idempotency key and applicable scope and grant revisions. A repeated key with
   different binding fails. A queue claim is ownership, not permission.
2. **At dispatch**, check a live distinct outbox claim generation, current
   authorization, cancellation and recipient scope. Record a durable dispatch
   attempt before provider I/O. A check followed by an external call is not atomic
   with revocation or lease loss; retain uncertainty if the call may have begun.
   Reclamation must not automatically resend a possibly dispatched attempt.
3. **Evidence** must come from a trusted, operation-specific provider adapter or
   independently authorized lookup, and bind the same attempt, integration,
   recipient and provider correlation. A generic outbox `delivered` status,
   arbitrary ID, unverified model claim, provider acceptance without the
   required semantics, or absent lookup result cannot by itself prove delivery
   or non-delivery. Some providers cannot establish a negative result; unknown
   must remain unknown. Recipient read is a separate observation.
4. **At settlement**, reauthenticate the recovery actor, verify current task,
   claim generation, cancellation and authority, and atomically compare the
   pending action binding and evidence to the checkpoint version. Persist the
   verified outcome and evidence before clearing the marker. If authorization is
   revoked, the result is ambiguous, bindings differ, or a required transactional
   check is unavailable, retain the marker and expose only an authorized blocker.
   Do not use reconciliation to silently change a cancelled task to success.
5. **Continuation** requires a separate reviewed policy. A verified non-effect
   does not itself authorize a retry, and a verified effect does not establish
   that the recipient read it. Unknown is never converted to failed merely to
   make retry convenient. Record resource consumption without inventing provider
   usage or refunding an uncertain reservation.

## Gates and executable evidence

Implement the required claim propagation and mixed-writer exclusion first; bind
the attempt and checkpoint in one store; then supply per-provider evidence and
authorized settlement. Cross-store/provider data requires an explicit immutable
receipt and revocation protocol. Until those conditions hold, only inspect and
preserve the marker, and keep unsupported work blocked.

Use disposable actors and a synthetic provider with barriers to test: crash
before dispatch; acceptance followed by timeout; delayed confirmation; no
authoritative lookup; duplicate and conflicting idempotency keys; wrong actor,
recipient, integration, action or arguments; changed authority or cancellation
at every boundary; expired/reclaimed same-owner claims; stale publishers and
old writers; and restart after effect before settlement. Independently count
provider attempts and delivered effects. Assert no duplicate send, stale
publication, false completion, marker clearance, or disclosure to another actor.
