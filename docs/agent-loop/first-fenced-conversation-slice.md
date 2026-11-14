# First fenced conversation slice

Working acceptance target for #216, #221, #224, #225, #226 and #228. This is
not an implementation or approval to enable the experimental runtime. Keep V3
as the production bridge while the candidate is default-off.

## Observable result

Using only disposable data and a synthetic provider, submit an authenticated
direct-owner conversation message through the ordinary daemon entry point. The
candidate acquires one generation-bound claim, makes a bounded model decision,
stages one authorized response, records a dispatch attempt, and settles the
observed effect after a simulated lost response and restart. A separate counter
must show exactly one provider attempt and one effect. The same execution must
not publish, acknowledge, or clear an uncertain marker when a claim is lost,
authorization is revoked, or the effect cannot be independently established.

This first slice deliberately excludes cross-user question resumption, real
accounts, external providers, mixed-version deployment and household traffic.
Excluding them from the first demonstration does not waive them as integration
and release prerequisites. A simulated response and fixture authorization do
not establish real delivery, production authority, or model quality.

## Current evidence and missing connections

| Boundary | Existing evidence | Work before this slice passes |
| --- | --- | --- |
| Acquisition to processor | The fenced queue can issue receipts; `V2Daemon` and `AlphonseCore` currently refuse fenced work. | Acquire before per-message side effects, carry the opaque receipt internally to the selected candidate and every relevant sink, and dispose of each outcome safely. Preserve the refusal until the whole path is ready. |
| Decision and checkpoint | `run_atomic_step` reserves and settles a synthetic model call under the supplied claim. | Connect the acquired receipt to the actual candidate execution path, including renewal or an explicit safe time bound; do not serialize it to model or user surfaces. |
| Publication and dispatch | Experimental staging checks claim and checkpoint; synthetic dispatch persists a started attempt. | Connect the candidate's response to a claim-bound publication and distinct dispatch lifecycle. Stop stale writers and make an attempt durable before the synthetic effect. |
| Recovery and settlement | Fixture settlement can match a synthetic effect against a pending action. | After restart, authenticate the recovery actor and establish current authority and exact action/effect binding; only then settle atomically. Unknown or revoked remains pending, with no replay. |

The existing experiments are intentionally unwired. In particular,
`run_atomic_step` stops at communication, and the fixture lifecycle starts with
an independently constructed reservation. Running those tests together is not
an end-to-end result. The experimental staged schema and fixture evidence must
not be installed in a household database or used as production authority.

## Acceptance checks

One integrated disposable test should drive the normal entry point and observe
durable state and the synthetic provider, rather than calling individual
reservation and settlement helpers as a substitute for the path. Exercise:

1. The successful claimed conversation through one effect and settlement.
2. A provider effect followed by a lost response and restart: no second attempt;
   independently verified evidence permits settlement by the authorized actor.
3. No authoritative provider answer: the marker remains pending, no automatic
   retry or success is reported, and an authorized user receives a truthful
   blocker without leaking another actor's data.
4. Lease expiry or generation change before publication and after dispatch:
   the stale worker neither publishes nor settles; an effect that may have
   happened remains uncertain. Do not assume a replacement claim can take
   over an old action without an explicit authorized handoff.
5. Cancellation, revocation, wrong actor and mismatched action/evidence at the
   relevant boundaries: no unauthorized effect, clearance, or false completion.

Tests must verify sink writes and provider attempts independently, use isolated
storage, and include a restart of the component under test. They should replace
overlapping lower-level checks only after those checks' distinct failure modes
are accounted for. Keep production integration blocked until mixed-writer
exclusion, real authority and provider evidence, all affected sinks, and the
additional #221–#228 requirements are reviewed and tested.

## Parallel work

The gym corpus and independent scoring (#219–#220), synthetic candidate work
(#223), host-contract development (#221), and legacy inventory (#222) can proceed
without waiting for this slice. Communication work can use disposable hosts;
production integration and qualification remain gated on the relevant contracts
and safety guarantees. Record measured outcomes and unresolved gaps in #230.
