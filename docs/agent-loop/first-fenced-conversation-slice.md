# First fenced conversation slice

## Disposable CoreLoop and inference-router progress — 14 November

The default router can now accept an explicitly injected synthetic processor
for a fixture-only engine stamp. A test queues a disposable message through
the actual CoreLoop and router, executes the candidate with an in-memory native
read tool and returns an undelivered draft. The adapter rejects persistent
queues, real memory, conversation storage and publication sinks; it uses a
synthetic budget, and its same-process duplicate check does not survive restart.
The normal runtime does not register or select this engine. CoreLoop's WAITING
state is not a safety fence: core prework occurs before processor checks, and
memory finalization and consumed-message acknowledgement can still occur. Never
attach the fixture processor to a household core or treat its draft as delivered.

An evaluation-only chooser now exercises the existing InferenceRouter with an
explicit read-only tool catalog, rather than the fixture-specific source enum.
The queued CoreLoop test uses this chooser and a scripted provider in the same
path: queue, engine router, candidate, inference router, authorized fixture
read, and undelivered draft.
It checks action shape and preserves unknown provider usage; tool authorization
and argument enforcement remain host responsibilities. It has only been tested
with a synthetic provider, not enabled in the real host. The configured Codex
provider now requests CLI JSON events for `simple_loop_action` and captures
only unambiguous reported input/output counters. An offline test drives two
decisions through that provider and the normal durable budget, without the
synthetic unknown-usage allowance. Full configured-provider usage reporting and
accounting semantics are not yet verified; absent, malformed or ambiguous counters remain
unknown, and the budget still refuses to proceed. This does not implement
pre-dispatch provider token limits, authenticated authority or delivery.

One subsequent synthetic CLI contract probe used an API-credential-scrubbed
environment, checked that the CLI reported ChatGPT login, and invoked the
installed CLI with `--json`, `--output-last-message`, an output schema and a
read-only sandbox. The expected last-message JSON was returned, and the parser
extracted one `turn.completed` usage record: 14,588 input and 15 output tokens
reported by the CLI. The receipt is
`evidence/gym-cycle-2026-11-13/codex-usage-contract-v1.jsonl`; it contains no
raw CLI output. The unexpectedly large reported input count for a tiny prompt
also shows that visible prompt size is not a reliable substitute for the
provider's count. This establishes that the parser works for one actual CLI
event stream, not the complete normal-provider path: the probe deliberately
scrubbed credentials and ignored user config, the normal provider does not do
the same auth preflight, counter semantics and costs remain unverified, and no
pre-dispatch hard cap or real user delivery was tested.

The next integration step is a disposable daemon/SQLite-queue exercise of the
configured inference router, one actor-authorized read operation, and the real
response/outbox projection. That needs durable task identity, claim-bound
publication, and cancellation/restart behavior at the actual boundaries. A
draft in a processor snapshot is not a real response, and simply relabeling it
as WAITING does not solve delivery or stale-worker handling. Until those pieces
are proven for the path, this work is not an owner-use opt-in.

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
the opt-in synthetic `run_atomic_step` can now retain a fixture-verified answer
under its claim, and a fixture test carries those exact bytes through reservation,
staging, dispatch and settlement. An answer rejected by the final freshness
check remains unaccepted and cannot be reserved. The disposable component test
now begins with acquisition and creates the initial task and budget under that
claim before the synthetic model and delivery steps. This is not the daemon
entry point: fixture authority and evidence are forgeable. A separate fresh
interpreter can now reconcile a matching fixture effect against the expired
original queue generation and current fixture grants without a receipt, replay,
queue acknowledgement or task continuation. This tests reconstruction after
simulated process loss, not authenticated recovery or replacement-claim takeover.
The fixture checker and source reads are not atomically bound to later dispatch
or revocation; no real current-authority or provider-evidence check exists. The
experimental staged schema and fixture evidence must not be installed in a
household database or used as production authority.

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
