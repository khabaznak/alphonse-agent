# Stage 4: Outer completion routing and response

Status: not started  
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

- [ ] Define the phase-review output schema.
- [ ] Update Check to consume `PhaseOutcome` and full bounded cumulative evidence.
- [ ] Add deterministic handling for system-verifiable criteria and invariants.
- [ ] Separate phase completion from mission completion.
- [ ] Define Act's strategic-decision schema and invocation conditions.
- [ ] Add direct terminal routing after verified task completion.
- [ ] Add a dedicated response-generation boundary with no tools.
- [ ] Ensure a response summarizes only verified outcomes and visible blockers.
- [ ] Prevent another planning phase after terminal verification.
- [ ] Preserve waiting/parking behavior for necessary user decisions.
- [ ] Ensure steering amends outcomes only through the existing explicit contract path.
- [ ] Update activity/UI events to distinguish tactical progress, phase review, and
      mission completion.

## Tests

- [ ] A verified simple mutation routes directly to response and ends.
- [ ] A completed phase with unmet task criteria creates another strategic phase.
- [ ] A tactical fallback success does not force strategic replanning.
- [ ] A scope violation prevents completion.
- [ ] Earlier contradictory evidence remains visible during later phase review.
- [ ] A failed phase produces a visible blocker rather than silence.
- [ ] A required user decision parks the task and resumes safely.
- [ ] Steering during execution triggers explicit contract amendment review.
- [ ] Final response cannot precede the final mutation/verification action.
- [ ] No extra Plan inference occurs after verified task completion.

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

## Open decisions

- Whether final response generation is an explicit graph node or a terminal Act
  operation with a separate no-tools inference purpose.
- Which criteria can be verified deterministically without a Check inference.
- How much phase evidence enters the response generator versus a compact verified-fact
  projection.

## Implementation log

- No implementation entries yet.

