# Integrating the candidate with normal conversations

Status: implementation plan, not an enabled feature. The comparison currently
selects a processor in a synthetic harness with synthetic sources. It does not
route ordinary Alphonse conversations, prove equivalent capabilities or deliver
messages to a user. Keep it separate from the production selector until the
contracts below are implemented and tested.

The desired end state is a default-off selection between V3 and a candidate
processor in the same conversation pipeline. Both use the configured Alphonse
provider and authorized capabilities. Selection alone is not an authorization
or verification boundary. A task retains its selected engine and schema through
queue retries, questions and restart, even if settings change later.

## Dependency order

1. **Define the host and task contracts.** Specify the candidate processor's
   inputs, outputs, per-task limits, observation provenance and terminal states.
   Integrate actor-scoped source access and current tool authorization; preserve
   skill/package and account bindings. Define how incomplete observations,
   unavailable services, cancellation and deadlines become visible blockers.
   The synthetic verifier and its hidden source requirements are not a
   production grounding service.
2. **Make task ownership and recovery durable.** Complete the queue claim and
   supersession work in [25](25-queue-claim-fencing.md), and bind checkpoint,
   budget, in-flight action and ownership transitions. Define reconciliation
   before resuming an uncertain action; do not replay an effect or delivery
   merely because a process restarted. The current experimental checkpoint
   transaction supports model decisions only and legacy writers are not fenced.
3. **Integrate tools and evidence.** Use the normal invocation service with
   actor-scoped checks before dispatch and after I/O. Implement durable,
   action-bound evidence and operation-specific effect verification as described
   in [24](24-single-loop-evidence-boundary.md). Preserve mutation scope,
   prerequisites, revocation and cancellation. An arbitrary source ID or
   successful process exit cannot prove an effect. Start with read-only tasks;
   do not represent unsupported mutation capabilities as completed.
4. **Integrate communication and terminal handling.** Use the existing
   authenticated question/resume and message delivery lifecycle, including
   failed or uncertain delivery. Persist useful progress and raise a clear
   question or blocker when sensible options are exhausted. Mark success only
   after the relevant verified result and communication contract are met.
5. **Wire the default-off selector.** Add an explicit candidate engine and
   schema to settings, message stamping, queue/recovery and processor routing.
   Stamp once at task creation and preserve that stamp on resume; reject
   unknown or inconsistent engine/schema values instead of falling through to
   V2. Register the candidate only after the preceding contracts exist. Keep
   V3 as the existing default and make switching back apply to new tasks rather
   than silently changing a running task's engine.
6. **Qualify before enabling.** Exercise both processors through the same
   conversation entry point, configured subscription-backed model, tools and
   synthetic data. Keep model/provider instructions, source access and response
   validation differences visible; an evaluator's private answer key must not
   supply one arm with an unshared advantage. Preserve every outcome and assess
   user-visible completion, elapsed time, calls and available usage separately.
   Freeze held-out cases before running them; report unsupported capabilities
   and failures rather than treating them as passes.

## Required verification

- Selector tests: default remains V3; explicit selection reaches the intended
  processor; unsupported stamps and schema mismatches fail closed; queued and
  resumed tasks keep their original engine after a settings change.
- Real conversation pipeline tests with disposable actors and sources: access
  denial and revocation, partial/missing evidence, cancellation and deadlines,
  grounded answer, question and authenticated resume, visible failure, and
  successful versus pending/failed/unconfirmed delivery.
- Crash and concurrency tests at reservation, dispatch, evidence recording and
  delivery boundaries: stale claims and writers cannot publish; uncertain
  actions are reconciled without duplicate execution or fabricated success.
- Effect tests for each supported mutation: actor, task, arguments, resource,
  scope and authority revisions match the verified receipt; unrelated, partial,
  stale and forged evidence cannot settle an action.
- Matched live comparisons after the deterministic integration checks pass:
  clock, inbox, prescription and other frozen tasks with independent outcomes.
  Synthetic success does not establish real Gmail access, memory retrieval,
  clinical correctness or actual user delivery; test those boundaries separately.

Until these checks pass, the existing candidate and its subscription-backed
fixture runs are development evidence, not a production-ready alternative.

## Engine-stamp compatibility boundary

The production router no longer constructs or dispatches the V2 processor.
An existing V2 default preference is interpreted as V3 for new tasks; a new
request cannot select V2. Previously queued or restored V2 tasks are not
silently converted to V3 and are rejected at dispatch. Their records remain
intact, but they need a separate, explicit recovery strategy.

The router rejects unregistered stamps before calling the processor.
This includes a restored `single_loop_experimental` stamp; it does not register
the candidate. A legacy compatibility exception remains earlier in task
construction: `TaskState.from_message` and `TaskState.from_dict` convert a
missing, empty or null engine value to `tactical_v2`. A whitespace-only value is
stripped to an empty stamp and rejected by the router; a nonempty unknown value
survives restoration and is rejected. Thus the router's rejection does not
establish that all malformed incoming stamps are rejected. Existing unstamped
checkpoints can rely on this fallback, so changing it requires a versioned
migration or another explicit way to distinguish legacy records from invalid
new records. Focused tests cover restored missing/empty/null compatibility and
unknown stamps, but do not qualify the full queue/resume path. The old PDCA
implementation and historical evaluation adapters still exist in the repository;
removing their code requires separating shared components and retiring those
evaluation paths. The `agent_v2` package and database-table prefix do not by
themselves select the old V2 intelligence processor.
