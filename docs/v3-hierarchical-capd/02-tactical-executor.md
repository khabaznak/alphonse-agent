# Stage 2: Bounded tactical phase executor

Status: complete
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
    use the phase-wide System One tool palette
    ask bounded System Two inference to compose the next concrete invocation
    validate action against tools, capabilities, dependencies, and scope
    execute exactly once
    append action and result evidence
    evaluate hard completion invariants, then ask System One about semantic completion
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

- [x] Add a `PhaseExecutor` independent of the V2 one-call Do implementation.
- [x] Add a tactical-action inference purpose and strict structured output validation.
- [x] Use one parallel Jev Noul question per registry tool to select a phase-wide
      semantic palette; leave concrete invocation composition to System Two.
- [x] Use Jev for semantic inner-Do completion checks after operational success.
- [x] Validate every selected action before invocation.
- [x] Reuse `ToolInvocationService` as the only tool execution boundary.
- [x] Decrement budgets for attempted calls, including failures.
- [x] Check cancellation and queued steering between every action.
- [x] Persist state and evidence after every action.
- [x] Implement typed subgoal-output binding.
- [x] Implement dependency resolution and skip prevention.
- [x] Implement local completion predicates.
- [x] Implement bounded local fallback selection.
- [x] Make all unexpected failures visible in `PhaseOutcome`.
- [x] Ensure successful writes require structured verification or a separate read-back
      completion condition.
- [x] Keep V2 available as an explicit rollback engine while V3 becomes the default
      for newly ingested tasks.
- [x] Emit nested activity events for phase, subgoal, and tactical action.

## Tests

- [x] Execute a bounded multi-action read/verify phase.
- [x] Execute an adaptive search/edit/verify phase with output binding.
- [x] Stop dependent actions after a failed prerequisite.
- [x] Recover locally through one authorized fallback.
- [x] Return for strategic review when scope must expand.
- [x] Enforce total, subgoal, and duration budgets.
- [x] Preserve the shared nonzero Bash failure semantics through
      `ToolInvocationService`.
- [x] Interrupt safely on steering and resume under outer control.
- [x] Cancel between actions without running another tool.
- [x] Restart from a checkpoint without repeating a completed side effect.
- [x] Reject an action that was not revealed or authorized.
- [x] Preserve every partial result when a later action fails.

## Non-goals

- Full progressive tool revealing; Stage 2 may use a fixed test shortlist.
- Mission-completion routing.
- Broad parallel execution.

## Exit criteria

- One Do phase can complete locate/update/verify without global CAPD between calls.
- Checkpoints resume without duplicated writes.
- Tactical failures and recoveries are fully auditable.
- The executor cannot change strategic outcomes or mutation authorization.
- Focused tests and the full suite pass with V2 still available.

## Decisions made

- Local completion predicates are deterministic and declared in the subgoal contract.
- Tactical inference receives a compact reconstructed state; no provider-specific
  conversation handle is required for correctness.
- Program mode remains an optional execution optimization under the shared invocation
  boundary; child-call evidence remains authoritative.

## Implementation log

- 2026-09-20 — Added the bounded `PhaseExecutor`, tactical inference purpose,
  per-action checkpoint/evidence recording, typed bindings, local fallback, scope/tool
  rejection, steering/cancellation boundaries, and nested progress events. Full suite:
  437 passed.
- 2026-09-20 — Added System One tactical tool choice, per-transition durable
  checkpoints, bounded project search/read tools, and hard protection for `.alphonse`
  internal state. System Two now generates arguments for one selected tool rather
  than reconsidering the complete reveal set.
