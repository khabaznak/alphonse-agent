# Stage 1: Phase contracts and persisted state

Status: not started  
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

- [ ] Choose package location for the V3 intelligence engine without duplicating V2
      infrastructure.
- [ ] Define versioned phase and tactical-state schemas.
- [ ] Define stable enums for statuses, failure policies, and side-effect classes.
- [ ] Add validation for duplicate IDs, missing dependencies, dependency cycles,
      negative budgets, and invalid mutation scopes.
- [ ] Add JSON-safe serialization and restoration.
- [ ] Add a bounded prompt projection separate from the full audit representation.
- [ ] Add V3 fields to task checkpoints behind an explicit engine/schema version.
- [ ] Define how immutable acceptance-criterion IDs are referenced by a phase.
- [ ] Define how subgoal outputs are typed and bound for dependent actions.
- [ ] Define how cumulative V2 evidence is imported into a new V3 phase.
- [ ] Add a compatibility adapter for a legacy one-call plan where appropriate.
- [ ] Add structured logging fields for phase ID, subgoal ID, action ID, and budgets.
- [ ] Keep the V2 planner and executor unchanged by default.

## Tests

- [ ] Round-trip every V3 state model through JSON/checkpoint persistence.
- [ ] Reject invalid subgoal dependency graphs.
- [ ] Reject mutation scopes outside the authorized project.
- [ ] Verify state-transition rules and terminal-state immutability.
- [ ] Verify budget decrement and deadline restoration after restart.
- [ ] Verify full evidence remains append-only while prompt projection is bounded.
- [ ] Verify steering metadata survives checkpoint restoration.
- [ ] Verify a legacy one-call plan maps only to a one-subgoal compatibility phase.
- [ ] Verify old V2 checkpoints still load unchanged.

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

## Open decisions

- Dataclasses versus a dedicated validation library.
- Whether deadlines persist as absolute timestamps or remaining duration plus restart
  policy.
- Whether subgoal output types use a fixed enum, JSON Schema, or both.
- Whether an active V2 task may opt into V3 or must finish on its original engine.

## Implementation log

- No implementation entries yet.

