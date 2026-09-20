# Stage 4: Outer completion routing and response

Status: complete
Depends on: Stages 1 through 3

## Objective

Make outer CAPD evaluate one complete tactical phase, distinguish tactical failure
from strategic failure, and terminate immediately after verified task completion.

This stage removes the current pattern of returning to global Plan after every tool
call or after already-proven success.

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

- Final response generation is a terminal outer-controller operation with a dedicated
  no-tools inference purpose.
- Mutation authorization, affected paths, phase status, and structured verification
  flags are deterministic invariants. Semantic task criteria remain phase-review
  inference responsibilities.
- Response generation receives at most the six latest successful evidence entries plus
  the verified objective and review reason.

## Implementation log

- 2026-09-20 — Added phase-review and strategic-decision contracts, deterministic
  mutation-scope checks, cumulative evidence-based acceptance review, direct terminal
  routing, waiting/failure routes, and no-tools verified response generation. Full
  suite: 451 passed.
