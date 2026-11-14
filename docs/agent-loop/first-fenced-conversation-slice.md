# First fenced conversation slice

## Owner-only integration work — 14 November

An explicitly injected, default-off owner-read processor now exercises the
candidate and shared chooser with the actual conversation search/read tools and
a file-backed budget. It requires an active canonical administrator plus a
trusted claim-authorization callback, binds each observation to its action and
revalidates before reuse. Its search scope is restricted to owner conversation
records; its response is only an undelivered draft. The integrated disposable
queue test supplies a fake claim and an in-memory checkpoint collector. These
are not production authentication or durable recovery.

An explicitly registered candidate may use the queue message ID as its task ID
and load only a matching owner/engine/schema checkpoint. Existing budgets block
blind replay; observation reconstruction is not implemented. The existing
outbox projector refuses candidate snapshots rather than publishing through an
unfenced legacy path. The normal runtime still registers only V3; generic IPC
can accept caller-provided identity/channel metadata, so it must not be used as
an owner credential. Budget and checkpoint writes are not atomic, and there is
no claim-fenced response delivery. Consequently, this work does not yet enable
owner-only conversations on restart.

A separate, constructor-opt-in experimental socket route now checks the local
peer's OS UID and accepts only a prompt. In the same SQLite transaction it
selects the active canonical admin and creates a server-ID queue message plus
an origin record containing the UID and a digest of the stored payload. It
bypasses normal inbound routing and conversation-history recording. A verifier
checks this record, the exact task and payload, the active admin, and the current
unexpired queue claim in a fenced transaction. A disposable test passes that
verifier as the real owner-read processor's authorization callback and reaches
an undelivered synthetic draft. Other tests reject spoofed request fields,
different actors, stale claims, changed payloads, missing origins and inactive
admins, and check transaction rollback.

This is a component integration, not the normal conversation path: the daemon
does not enable the route and CoreLoop refuses fenced receipts before processing.
CoreLoop can now carry the acquired receipt internally into its processor context,
and an owner authorization callback verifies it against the queue. The existing
fenced-worker refusal remains in force, so this transport is not active in the
normal daemon. The UID check assumes
the local OS principal is allowed to act as the configured admin; it does not
identify a household member independently. The origin record is ordinary
same-database state, not protection against a process able to rewrite that
database. Its retention and deletion lifecycle has not been integrated. A
successful check is point-in-time only: it does not fence later inference,
reads, checkpoint or budget writes, publication, delivery or revocation.
The queue also exposes a trusted-owner claim transaction that checks the owner
and origin before and after writes on its own SQLite connection, rolling them
back if authority or the claim changes. Candidate budget, checkpoint and
evidence transitions do not yet use it; it cannot protect other databases or
external calls.

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

The candidate's router chooser is now in the shared runtime code and uses the
existing InferenceRouter with an explicit read-only tool catalog, rather than
the fixture-specific source enum. Its caller supplies a task with actor, task
ID and project ID; these are carried into each inference request, and changes
to that identity before or during a call block the returned decision. This is
an identity binding, not authentication, a claim fence or authorization. The
evaluation import remains as a compatibility alias. The normal daemon does not
register the candidate or instantiate this chooser.
The runtime and gym adapters now call a shared next-action request builder,
schema builder and action-shape normalizer, including the source-selection and
uncertainty guidance. The gym retains fixture-only schema, telemetry and
subscription checks. Optional runtime observation-progress labels must be
supplied by a host that independently enforces current actor access; no
production host currently supplies them. Synthetic self-review and the
synthetic unknown-usage allowance remain evaluation-only.
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
event stream, not the complete normal-provider path. The configured provider's
simple-loop transport now also scrubs ambient API credentials, checks CLI login
status for ChatGPT, forces the OpenAI provider, and uses an ephemeral read-only
invocation with user configuration ignored. The login command cannot apply the
same ignore-user-config option as execution, and an authentication change
between preflight and dispatch remains possible. The new configured-provider
path also has the synthetic live probe described below. Counter semantics and costs remain unverified, and
no pre-dispatch hard cap or real user delivery was tested. V3 transport behavior
is unchanged.

The configured provider's `simple_loop_action` path now also performs a
ChatGPT-login preflight with a scrubbed environment before a read-only,
ephemeral CLI invocation; V3 and the evaluation subclass retain their existing
transports. A synthetic configured-provider probe returned the expected JSON
and CLI-reported usage. The first receipt,
`configured-provider-subscription-v1.jsonl`, predates a final environment
allowlist change; a second receipt, `configured-provider-subscription-v2.jsonl`,
matches the final source hash and reports 14,744 input / 15 output tokens. This
is one synthetic run, not a guarantee that authentication stays unchanged after
preflight, that login and execution configuration are perfectly equivalent,
or that the provider enforces a pre-dispatch token cap. No real household data
or delivery was used.

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
| Acquisition to processor | The fenced queue issues receipts; an opt-in local-ingress component binds a queue row to an active admin and verifies its origin against a current claim. CoreLoop has an internal receipt field and a disposable owner-processor test supplies a real receipt. `V2Daemon` and `AlphonseCore` currently refuse fenced work. | Acquire before per-message side effects, propagate the receipt to every relevant sink, and dispose of each outcome safely. Preserve the refusal until the whole path is ready. |
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
