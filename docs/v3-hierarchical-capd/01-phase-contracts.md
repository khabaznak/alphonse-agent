# Stage 1: Phase contracts and persisted state

Status: complete
Depends on: current V2 acceptance contracts and task checkpointing

## Objective

Define the durable, provider-independent contracts that separate strategic phases
from tactical actions. Persist enough state to pause, restart, steer, inspect, and
resume a phase without reconstructing it from model prose.

This stage must not change the default execution path.

## Deliverables

Introduce typed models equivalent to:

```python
PhasePlan
PhaseSubgoal
PhaseLimits
MutationScope
CompletionCondition
TacticalState
TacticalAction
PhaseEvidence
PhaseOutcome
```

The exact Python representation may use dataclasses or validated dictionaries, but
serialization must be explicit and stable.

## Required contract semantics

### `PhasePlan`

- Version and unique phase ID.
- A concise phase objective tied to the immutable task criteria.
- Ordered subgoals with stable IDs.
- Phase-level call and duration limits.
- Authorized capabilities.
- Authorized mutation scope.
- Completion and escalation conditions.
- Creation time and originating outer-plan decision.

### `PhaseSubgoal`

- Objective and typed required output.
- Dependencies on earlier subgoals.
- Allowed capabilities, side effects, and per-subgoal limits.
- Completion predicate.
- Failure policy: stop, local fallback, wait, or return for strategic review.

### `TacticalState`

- Current phase and subgoal.
- Bound outputs from completed subgoals.
- Revealed capabilities and concrete tools.
- Ordered tactical actions and results.
- Remaining budgets and deadline.
- Steering/cancellation markers.
- Current terminal or nonterminal phase status.

### `PhaseEvidence`

- Append-only action references.
- Structured results and failures.
- Actual affected paths and external side effects.
- Before/after hashes or provider-native equivalents when available.
- Verification observations and contradictions.
- A bounded prompt projection that does not destroy the full checkpoint/audit record.

## State transitions

Define and validate at least:

```text
planned -> running
running -> subgoal_complete
running -> waiting_user
running -> blocked
running -> budget_exhausted
running -> cancelled
subgoal_complete -> running(next subgoal)
subgoal_complete -> phase_complete(last subgoal)
```

Terminal states must be explicit. Invalid transitions must fail closed.

## Implementation checklist

- [x] Choose package location for the V3 intelligence engine without duplicating V2
      infrastructure.
- [x] Define versioned phase and tactical-state schemas.
- [x] Define stable enums for statuses, failure policies, and side-effect classes.
- [x] Add validation for duplicate IDs, missing dependencies, dependency cycles,
      negative budgets, and invalid mutation scopes.
- [x] Add JSON-safe serialization and restoration.
- [x] Add a bounded prompt projection separate from the full audit representation.
- [x] Add V3 fields to task checkpoints behind an explicit engine/schema version.
- [x] Define how immutable acceptance-criterion IDs are referenced by a phase.
- [x] Define how subgoal outputs are typed and bound for dependent actions.
- [x] Define how cumulative V2 evidence is imported into a new V3 phase.
- [x] Add a compatibility adapter for a legacy one-call plan where appropriate.
- [x] Add structured logging fields for phase ID, subgoal ID, action ID, and budgets.
- [x] Keep the V2 planner and executor unchanged by default.

## Tests

- [x] Round-trip every V3 state model through JSON/checkpoint persistence.
- [x] Reject invalid subgoal dependency graphs.
- [x] Reject unsafe absolute or parent-traversing mutation paths; runtime project-root
      authorization remains an executor responsibility.
- [x] Verify state-transition rules and terminal-state immutability.
- [x] Verify budget decrement and deadline restoration after restart.
- [x] Verify full evidence remains append-only while prompt projection is bounded.
- [x] Verify steering metadata survives checkpoint restoration.
- [x] Verify a legacy one-call plan maps only to a one-subgoal compatibility phase.
- [x] Verify old V2 checkpoints still load unchanged.

## Non-goals

- Executing multiple tactical actions.
- Dynamic tool selection.
- Changing outer graph routing.
- Desktop UI work beyond any schema fixtures required for compatibility.

## Exit criteria

- All V3 contracts are documented and covered by unit tests.
- V3 state can be interrupted and restored losslessly.
- Invalid plans cannot enter execution.
- V2 behavior and the full existing test suite remain unchanged.
- A code review can answer exactly what is strategic state, tactical state, evidence,
  and authorization state.

## Decisions made

- Use dependency-free dataclasses with explicit validation and serialization.
- Persist deadlines as timezone-aware absolute timestamps.
- Use stable semantic output-type identifiers with explicit typed binding; introduce
  JSON Schema later only where a tool family needs field-level validation.
- Stamp the engine on task state; active V2 tasks remain V2 unless a future explicit,
  tested migration is requested.

## Implementation log

- 2026-09-20 — Added `intelligence/v3` contracts, V3 fields in `TaskState`, legacy
  one-call adaptation, absolute phase deadlines, typed bindings, bounded projections,
  and contract/checkpoint tests. Full suite: 428 passed.
