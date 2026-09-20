# Stage 2: Bounded tactical phase executor

Status: not started  
Depends on: Stage 1

## Objective

Implement the inner tactical loop inside Do. It executes a validated phase, selects or
derives concrete actions, observes results, completes subgoals, and performs bounded
local recovery without re-entering outer CAPD after every tool call.

## Responsibility boundary

The tactical executor may:

- Choose a concrete action permitted by the current subgoal.
- Execute a revealed and authorized tool.
- Bind structured results for later subgoals.
- Retry or choose an allowed fallback within budget.
- Stop and return structured phase evidence.

It may not:

- Add, remove, weaken, or satisfy task acceptance criteria.
- Broaden mutation scope or authorize a new external side effect.
- Change the phase objective materially.
- Declare mission success.
- Hide failed or partial actions.
- Continue after cancellation, steering interruption, or exhausted limits.

## Execution algorithm

```text
load validated phase and tactical state
while phase is nonterminal:
    check cancellation, steering, deadline, and call budget
    resolve the current subgoal
    determine whether a static next action is already executable
    otherwise request one tactical action from bounded inference
    validate action against tools, capabilities, dependencies, and scope
    execute exactly once
    append action and result evidence
    evaluate the local completion predicate
    complete subgoal, recover locally, wait, or stop for outer review
return PhaseOutcome and PhaseEvidence
```

## Static and adaptive execution

- Static actions are used when tool arguments are already known.
- Adaptive action selection is used when later arguments depend on observed results.
- Provider/program execution may optimize a phase, but it must use the same invocation,
  authorization, budget, evidence, and interruption boundaries.
- Program success never substitutes for child-call evidence.

## Local recovery policy

Local recovery is appropriate when it preserves the same objective and authorization,
for example:

```text
artifact index has no match -> try allowed bounded project-file search
```

Return to outer Check/Act when:

- Authoritative records conflict.
- The user must choose among materially different interpretations.
- A new mutation target or external effect is required.
- The phase objective is invalidated.
- Repeated failures exceed the configured threshold.
- The phase or subgoal budget is exhausted.

## Implementation checklist

- [ ] Add a `PhaseExecutor` independent of the V2 one-call Do implementation.
- [ ] Add a tactical-action inference purpose and strict structured output schema.
- [ ] Validate every selected action before invocation.
- [ ] Reuse `ToolInvocationService` as the only tool execution boundary.
- [ ] Decrement budgets for attempted calls, including failures.
- [ ] Check cancellation and queued steering between every action.
- [ ] Persist state and evidence after every action.
- [ ] Implement typed subgoal-output binding.
- [ ] Implement dependency resolution and skip prevention.
- [ ] Implement local completion predicates.
- [ ] Implement bounded local fallback selection.
- [ ] Make all unexpected failures visible in `PhaseOutcome`.
- [ ] Ensure successful writes require structured verification or a separate read-back.
- [ ] Add a feature flag/engine selector so V2 remains the default initially.
- [ ] Emit nested activity events for phase, subgoal, and tactical action.

## Tests

- [ ] Execute a static two-action read/verify phase.
- [ ] Execute an adaptive search/edit/verify phase with output binding.
- [ ] Stop dependent actions after a failed prerequisite.
- [ ] Recover locally through one authorized fallback.
- [ ] Return for strategic review when scope must expand.
- [ ] Enforce total, subgoal, and duration budgets.
- [ ] Preserve nonzero Bash failure semantics inside direct and program execution.
- [ ] Interrupt safely on steering and resume under outer control.
- [ ] Cancel between actions without running another tool.
- [ ] Restart from a checkpoint without repeating a completed side effect.
- [ ] Reject an action that was not revealed or authorized.
- [ ] Preserve every partial result when a later action fails.

## Non-goals

- Full progressive tool revealing; Stage 2 may use a fixed test shortlist.
- Mission-completion routing.
- Making V3 the default engine.
- Broad parallel execution.

## Exit criteria

- One Do phase can complete locate/update/verify without global CAPD between calls.
- Checkpoints resume without duplicated writes.
- Tactical failures and recoveries are fully auditable.
- The executor cannot change strategic outcomes or mutation authorization.
- Focused tests and the full suite pass with V2 still available.

## Open decisions

- Whether local completion predicates are deterministic functions, schemas interpreted
  by the executor, or a narrowly scoped inference.
- Whether tactical inference should keep a provider conversation handle or receive a
  compact reconstructed state each time.
- Whether program mode remains an optimization under `PhaseExecutor` or becomes one
  tactical action type.

## Implementation log

- No implementation entries yet.

