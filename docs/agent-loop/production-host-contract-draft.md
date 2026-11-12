# Production host contract for the dependable loop

Status: proposed contract for #221, not an implemented host or an approval to
route household traffic. The candidate in `single_loop_candidate.py` accepts
synthetic callbacks and cannot satisfy this contract by trusting those callbacks.
The first production slice is read-only and remains default-off. Existing V3
continues to serve conversations until integration and qualification gates pass.

## Boundary and authority

The conversation host creates a task from an authenticated principal, not from
model-supplied identity. Its durable identity comprises task ID, owner, engine
and schema version, and the initiating conversation/audience. A distinct queue
actor or authenticated respondent is recorded separately; neither becomes the
task owner by assertion. Persist the engine stamp at creation and preserve it on
retry and resume. Unsupported stamps, inconsistent owner and schema, or missing
authority fail closed without silently converting an existing task. Legacy
unstamped records require the explicit compatibility treatment described in
[39](../v3-hierarchical-capd/39-candidate-conversation-integration.md).

The host exposes only tools discoverable for the current actor, task and phase,
with their argument schemas, effects and prerequisites. Discovery and relevance
are not permission. At every actual use, deterministic code checks current
actor, package access, skill/account/data binding, grant and resource scope,
including revocation; it rejects malformed arguments and unavailable bindings.
Standard skills remain packages, scripts remain resources, and host execution
must obey the existing skill contract. A read-only descriptor alone does not
establish that a tool is safe: each admitted operation needs reviewed behavior,
resource-specific authorization and observation semantics. The first slice
admits an explicit allowlist of reviewed read operations; no generic Bash,
effectful operation or synthetic fixture verifier is implied by that allowlist.

The loop may choose an action or compose an answer, but it receives a bounded,
authorized projection, not private oracle answers or credentials. Jev can make
bounded relevance judgments; neither Jev nor generated prose authorizes access,
settles effects or decides that a failed check succeeded. An observation is
host-created and identifies actor, task, action attempt, tool, resource,
coverage, freshness and status, with authorized provenance and content. Partial,
failed, denied, stale, conflicting and unknown observations remain distinct;
arbitrary IDs and exit status are not verification. Recheck access and relevant
source versions before reuse or publication; projection must not widen audience.

## State, ownership and I/O

The host maintains durable task state and independently tracks communication:

| Task condition | Required behavior |
| --- | --- |
| running | Only the current fenced owner may make a transition or initiate permitted work. |
| waiting for user | Persist the question and intended respondent, plus its delivery state; generated question text alone is not delivery. |
| blocked | Preserve useful progress and a specific, actor-visible blocker; do not claim completion. |
| cancelled | Stop new dispatch/publication and retain any uncertain in-flight effect for reconciliation. |
| completed | Require the task's verified result; track response delivery separately and accurately. |
| superseded | A sanitized internal result: stale worker does not publish, finish memory, acknowledge another claim, or report completion. |

These are behavioral requirements, not a prescribed database enum. Delivery
has separate pending, confirmed, failed and unconfirmed outcomes; provider
acceptance is not recipient read. Enqueue, generation or timeout is not proof of
delivery. When delivery is necessary to satisfy the user's request, completion
must reflect its actual verified outcome; otherwise report task completion and
delivery separately. A question can be resumed only by an authenticated,
authorized respondent bound to that open question and task, with replay-safe
answer receipt and a fresh ownership claim. Preserve task owner, engine,
budgets and prior evidence on resume. An answer to a different, cancelled,
already consumed or superseded question cannot restart work.

Before provider or tool dispatch the host checks a live claim, current task
version, authorization, cancellation, deadline and durable budget; it records
the action and reservation before I/O. Bound input and output as well as action
counts, conservatively retain unknown usage and stop when limits cannot be
enforced. A deadline or exhausted budget yields a visible blocker, not an
invented result. Recheck applicable controls after I/O and at publication.
Blocking calls need lease renewal/fencing; a pre-dispatch check alone cannot
guarantee ownership throughout an external call.

Checkpoint, reservation, action marker and claim checks must have an atomic
same-store transition or an explicitly equivalent protocol. A claim binds queue
message, queue actor, owner, fresh claim generation and live lease to the
actor-bound task version. A new acquisition always gets a new generation even
when owner or message ID is reused. No old worker may update task, memory, UI,
outbox, queue acknowledgement or handoff after losing ownership. A same-store
outbox insert alone does not fence a later external dispatch. Different stores,
old binaries and direct writers require an explicit fencing/migration protocol;
where unavailable, fail closed. Issue #216 owns these facilities rather than a
second competing queue implementation.

Crash before dispatch may be retried only if durable evidence proves it was
not dispatched. Crash, timeout, cancellation or lost response after dispatch
is uncertain; preserve the marker and reconcile without automatic replay. Even
read-only operations can expose data or have observable provider side effects,
so their retry policy must be operation-specific. A stable idempotency key must
bind the same actor, task, operation, arguments, resource and audience; a
conflicting reuse fails. Idempotency does not prove an external effect occurred.
Reconciliation is actor-authorized, fenced, auditable and never manufactures a
verified result from missing evidence. See [24](../v3-hierarchical-capd/24-single-loop-evidence-boundary.md)
and [25](../v3-hierarchical-capd/25-queue-claim-fencing.md).

Mutation support requires, per operation, durable pre-dispatch action and scope
binding, current authority at dispatch and settlement, operation-specific
independent verification or provider receipt, and recovery for uncertain
effects. External I/O cannot be made atomic with a SQLite checkpoint by
assertion. Cross-store receipts need a defined integrity, access, revocation and
fencing protocol. Unsupported effects remain unavailable rather than simulated.

## Current interfaces and unresolved boundaries

This audit is a starting point for implementing #221, not an assertion that the
existing interfaces satisfy the contract. In particular, no real operation is
yet approved for the first read-only allowlist.

| Existing interface | What it supplies today | What the production host still needs |
| --- | --- | --- |
| `runtime.build_runtime_host` and `core/messages/channel.py` | V3 is constructed and new messages are stamped; the candidate is not registered. | A separately qualified, default-off candidate route with actor, engine and schema preserved across the normal conversation and resume paths. Legacy unstamped tasks need explicit recovery, not a new stamp. |
| `ToolInvocationService` | Argument validation when a descriptor exists, execution context for compatible registries, telemetry and tool results. | A reviewed operation allowlist and deterministic resource-specific authorization at each boundary. `execute_or_raise` also supports legacy registries without descriptors; it must not be treated as a complete authorization or read-only policy by itself. |
| `CandidateHost` | Injected callbacks for synthetic tests and prototype read-only behavior. | Authenticated authority, trustworthy source observations and durable ownership; injected callback assertions and the synthetic response verifier are not production evidence. |
| `ExperimentalCheckpointStore` and `TaskBudget` | Experimental versioned candidate state and model-decision reservations. | A transaction tied to the queue claim and action/effect marker, with current actor and authorization checks. A task version alone cannot fence a re-leased worker. |
| `claim_fence.py`, `supersession_inspection.py` and `supersession_hold.py` | Experimental claim transactions, status-only inspection and an opt-in hold. Inspection supports direct ownership only; holds have no release or effect-reconciliation API. | End-to-end receipt propagation and lease renewal; an authorized, atomic handoff/recovery protocol; compatible old-writer exclusion and fenced sinks. These facilities are not wired into production. |
| `SQLiteQuestionStore`, `SQLiteOutboundStore` and daemon delivery | Persistent question, checkpoint and outbox operations with existing delivery handling. | A claim- and audience-bound publication boundary and authenticated replay-safe resume, including provider-specific evidence and crash recovery. An outbox insert does not fence subsequent dispatch. |

The core and daemon deliberately refuse an experimental fenced queue before
claiming or beginning work; `tests/test_experimental_core_fence.py` and
`tests/test_fenced_queue_daemon_preflight.py` exercise that refusal. This is a
containment guard, not stale-worker fencing for the normal queue. In particular,
a socket-scoped daemon lock is not a database-wide barrier against older binaries
or direct SQLite writers. Do not remove the guard as a shortcut to integration.

For #216, classification must not be confused with permission to transition:
cancelled, waiting-user, newer-active, terminal and unavailable/mismatched
checkpoints require distinct handling, but none authorizes the stale worker to
publish, acknowledge a re-leased claim, or replay a possibly dispatched action.
Inspection and durable hold alone do not define who may release a hold, how a
single continuation is created, or how uncertain effects are reconciled. Those
decisions and executable crash/reclaim tests must precede activation.

## Dependency and acceptance map

1. **#221 contract:** review these requirements against real interfaces and
   decide the first allowlisted operations and their evidence semantics. This
   draft is not evidence that those decisions or implementation are complete.
2. **#216 / #224 ownership and recovery:** claim propagation, mixed-version
   exclusion, durable state/budget/action transactions, fenced publication and
   authorized uncertain-effect recovery. #216 remains open; existing primitives
   are default-off and do not establish the full protocol.
3. **#225 communication and #226 authorized host:** use the contract and #216
   facilities for authenticated questions, delivery and read-only observations.
   Neither generated text nor an outbox row is a verified delivery receipt.
4. **#220 / #219 / #223 gym:** independent synthetic scenarios and scoring can
   run now. They cannot substitute for production host or delivery tests.
5. **#228 integration:** only after relevant ownership, host and communication
   prerequisites are implemented; retain default-off routing and stable stamps.
6. **#227 release, then #229 retirement:** freeze held-out scenarios and prove
   deterministic safeguards and real conversation behavior, including failure
   and recovery. Preserve historical data; retain V3 until qualification passes.

### Executable evidence and missing tests

Current focused regression command (disposable fixtures, no provider calls):

```sh
pytest -q tests/test_single_loop_candidate.py tests/test_single_loop_checkpoint.py tests/test_single_loop_atomic_step.py tests/test_claim_fence_foundation.py tests/test_claim_fenced_outbox_diagnostics.py
```

These tests exercise prototype blocking, checkpoint and same-database claim
primitives. They are not acceptance tests for an integrated production host.
Before #228, add executable integration tests through the normal conversation
entry point, with synthetic actors and stores, that independently inspect both
persisted state and actual dispatch/publication counts:

- Cross-actor and revoked package, account, source and resource access is denied
  at discovery, dispatch, observation reuse and publication; partial or stale
  observations cannot support success. Concurrent revocation is barrier-tested.
- Cancellation, deadlines and budget exhaustion at each I/O boundary prevent
  further dispatch and false success; unknown provider usage stays unknown.
- Concurrent claim expiry/reclaim, cancellation, steering and answer races cannot
  cause stale checkpoint, memory, UI, queue acknowledgement, notification or
  provider dispatch. Include same-owner generation reuse, process restart and
  old-writer/different-store behavior.
- Crash before dispatch, after dispatch and before durable observation, after
  outbox insertion and during delivery preserves uncertainty; no duplicate
  external call or fabricated confirmation results. Verify conflicting
  idempotency keys are rejected and the authorized reconciliation path is fenced.
- Question generation, enqueue, failed/unconfirmed delivery and authenticated
  resume remain distinguishable; wrong actor/question, duplicate or stale answer
  cannot resume. Confirm actual response delivery separately from task outcome.
- Default routing remains V3, candidate is explicit and default-off, engine and
  schema survive retries/settings changes, and unsupported stamps fail closed.
- For each mutation admitted later, independently inspect the resource and test
  mismatched actor/action/arguments/scope/revision, fake success, partial effects,
  timeout and crash reconciliation before enabling that operation.

Each test must identify the concrete production interface under test; a skipped
placeholder, a mock-only answer, or the synthetic matched-provenance verifier is
not a passing integration gate. Qualification also needs frozen held-out and
subscription-backed runs with independent scoring; never infer real account
access, clinical correctness or user delivery from a synthetic pass.
