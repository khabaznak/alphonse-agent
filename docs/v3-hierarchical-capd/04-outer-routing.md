# Stage 4: Outer completion routing and response

Status: complete
Depends on: Stages 1 through 3

## Objective

Make outer CAPD evaluate one complete tactical phase, distinguish tactical failure
from strategic failure, and terminate immediately after verified task completion.

This stage removes the current pattern of returning to global Plan after every tool
call or after already-proven success.

Every request enters CAPD. Plan may create a single-stage `user_response` phase when
replying is the complete strategic plan. Jev receives the full registered native and
artifact tool catalog, identifies the relevant tools, and System Two composes the
concrete call. There is no pre-CAPD conversational classifier or special intent route.

## Check responsibilities

Outer Check evaluates:

- The immutable acceptance contract.
- The phase objective and subgoal outcomes.
- Cumulative evidence from every phase and action.
- Failed dependencies, contradictions, and partial effects.
- Authorized versus actual affected paths and external side effects.
- New user steering accumulated during execution.

Check returns structured state, not a new strategy:

```text
phase_verified_task_complete
phase_verified_task_incomplete
phase_blocked
waiting_user
verification_failed
cancelled
```

Check cannot redefine acceptance criteria or ignore earlier contradictory evidence.

## Act responsibilities

Act chooses among:

- Complete and prepare a user response.
- Continue with another phase under the same strategy.
- Revise the strategic approach.
- Ask for a necessary user decision.
- Fail visibly with a concrete blocker.

Routine successful continuation should be deterministic. Strategic inference is used
only when evidence invalidates an assumption, repeated failures show no progress, or
scope/tradeoffs require a decision.

## Completion routing

```text
Check = phase_verified_task_complete
        |
        v
Generate one user-facing response from verified outcome facts
        |
        v
Deliver response and end task
```

The controller must not call Plan again for confidence gathering after this state.
Response generation may be a dedicated inference purpose; it must not execute tools
or change task status.

## Noninterference as an invariant

“No unrelated changes” is enforced by the phase mutation scope and actual affected
paths, not repeatedly interpreted as an ordinary model-owned criterion.

- Execution rejects unauthorized targets when possible.
- Evidence records actual effects.
- Check rejects completion when effects exceed authorization.
- Act cannot waive the violation without explicit user authorization and a new phase.

## Implementation checklist

- [x] Define the phase-review output schema.
- [x] Update outer review to consume `PhaseOutcome` and cumulative phase evidence.
- [x] Add deterministic handling for system-verifiable criteria and invariants.
- [x] Separate phase completion from mission completion.
- [x] Define Act's strategic-decision schema and invocation conditions.
- [x] Add direct terminal routing after verified task completion.
- [x] Add a dedicated response-generation boundary with no tools.
- [x] Allow Plan to represent a conversational response as a one-stage phase using
      the registered `native.respond` capability.
- [x] Project successful V3 `native.respond` results into the ordinary outbox.
- [x] Ensure a response summarizes only verified outcomes and visible blockers.
- [x] Prevent another planning phase after terminal verification.
- [x] Preserve waiting/parking behavior for necessary user decisions.
- [x] Ensure steering amends outcomes only through the existing explicit contract path.
- [x] Update activity/UI events to distinguish tactical progress, phase review, and
      mission completion.

## Tests

- [x] A verified simple mutation routes directly to response and ends.
- [x] A completed phase with unmet task criteria creates another strategic phase.
- [x] A tactical fallback success does not force strategic replanning.
- [x] A scope violation prevents completion.
- [x] Earlier contradictory evidence remains visible during later phase review.
- [x] A failed phase produces a visible blocker rather than silence.
- [x] A required user decision parks the task and resumes safely.
- [x] Steering during execution triggers explicit contract amendment review.
- [x] Final response cannot precede the final mutation/verification action.
- [x] No extra Plan inference occurs after verified task completion.
- [x] A conversational request enters CAPD, receives a one-stage plan, and lets Jev
      select `native.respond` from the complete native-and-artifact registry.
- [x] Invalid phase contracts fail once with a visible controlled V3 error rather than
      consuming the queue retry budget.
- [x] Three completed phases without acceptance-criteria progress fail visibly instead
      of consuming the full outer-phase budget.
- [x] Phase history stores phase-local evidence and reconstructs a deduplicated
      cumulative view without exponential growth.
- [x] Exhausting the outer-phase budget persists a terminal failed checkpoint.

## Non-goals

- Default production rollout.
- Desktop visual polish beyond functional nested progress.
- Replacing immutable acceptance contracts.

## Exit criteria

- Outer CAPD operates at phase granularity.
- Verified tasks stop without redundant planning cycles.
- Noninterference is enforced as a system invariant.
- Every failure path produces a visible outcome, question, or retriable state.
- Representative simple tasks use one outer execution phase.

## Decisions made

- A response already produced by `native.respond` is preserved through terminal outer
  review and is not regenerated.
- Mutation authorization, affected paths, phase status, and structured verification
  flags are deterministic invariants. Semantic task criteria remain phase-review
  inference responsibilities.
- Response generation receives at most the six latest successful evidence entries plus
  the verified objective and review reason.
- An optional System One provider may perform semantic criterion/evidence decisions
  and recommend an Act route. Deterministic invariants remain authoritative, and Act
  accepts only conservative completion vetoes or replanning recommendations. See
  [Experimental System One Check and Act](06-system-one-check-act.md).

## Implementation log

- 2026-09-20 — Added phase-review and strategic-decision contracts, deterministic
  mutation-scope checks, cumulative evidence-based acceptance review, direct terminal
  routing, waiting/failure routes, and no-tools verified response generation. Full
  suite: 451 passed.
- 2026-09-20 — Added the response-only conversational route, prepared-response outbox
  projection, and controlled single-attempt handling for invalid System Two phase
  contracts after the live `Hola Alphonse!` regression exposed retry churn.
- 2026-09-20 — Fixed the direct-response terminal checkpoint mapping (`completed` task
  state to `done` persistence status) and added visible classification/response
  activity after a live greeting completed in memory but failed before delivery.
- 2026-09-20 — Removed the pre-CAPD direct-response gate. Conversational requests now
  follow CAPD: Plan defines a response phase, Jev evaluates every native and artifact
  tool, System Two composes the call, and Do executes `native.respond`. Removed canned
  user-facing response fallbacks.
- 2026-09-20 — Added `user_response` as an explicit phase side-effect class and exposed
  the exact side-effect vocabulary to Plan after a live greeting used the invented
  value `send_message_to_user`. Acceptance contracts now describe observable outcomes
  rather than forbidding the internal response tool, and terminal V3 errors report
  their real validation reason instead of blaming the configured model.
- 2026-09-20 — Replaced Plan's prose-only phase-shape description with the complete
  generic PhasePlan JSON Schema after a second live greeting omitted
  `completion.kind`. The schema enumerates every required nested field, completion
  kind, failure policy, capability, and side-effect value without task-specific routes.
- 2026-09-21 — Diagnosed the live studio-temperature failure. Plan inferred that the
  temperature was unavailable before Jev could evaluate the complete tool registry,
  then eight response-only phases repeated while one conjunctive acceptance criterion
  remained pending. Tightened Plan and acceptance-contract instructions, added a
  three-phase no-progress terminal guard, made phase history store only local evidence
  with deduplicated reconstruction (preventing 1/2/4/8 evidence amplification),
  persisted phase-budget failures as terminal checkpoints, and corrected their
  user-facing error classification.
