# Stage 2: Tactical phase executor

Status: complete
Depends on: Stage 1

## Objective

Implement the inner tactical loop inside Do. It executes a validated phase, selects or
derives concrete actions, observes results, completes subgoals, and performs
local recovery without re-entering outer CAPD after every tool call.

## Responsibility boundary

The tactical executor may:

- Choose a concrete action permitted by the current subgoal.
- Execute a revealed and authorized tool.
- Bind structured results for later subgoals.
- Retry or choose an allowed fallback when Jev judges it useful and safe.
- Stop and return structured phase evidence.

It may not:

- Add, remove, weaken, or satisfy task acceptance criteria.
- Broaden mutation scope or authorize a new external side effect.
- Change the phase objective materially.
- Declare mission success.
- Hide failed or partial actions.
- Continue after cancellation or steering interruption.

Registered project artifacts are fully trusted, admin-managed capabilities. They must
still be selected from the registry, revealed for the current subgoal, and invoked
through the shared schema-validation and evidence boundary, but the executor does not
apply the generic native-tool `read_only` authorization gate to them. Native tools
continue to be checked against the phase's declared side-effect permissions.

`native.bash` is also available to V3 as the trusted `local_shell` capability. It may
run direct CLI, filesystem, process, build, test, diagnostic, and artifact-authoring
commands. Its calls honor an explicit caller-supplied timeout when present, retain
bounded output capture, and are recorded through the normal tactical evidence path;
the executor does not apply the generic read-only gate to Bash.

Bash has no product-imposed default or maximum execution timeout. A caller may set
an explicit positive timeout when the command itself warrants a deadline; otherwise
the command runs until it exits or the task is cancelled. Output capture remains
bounded.

## Execution algorithm

```text
load validated phase and tactical state
while phase is nonterminal:
    check cancellation and steering
    resolve the current subgoal
    determine whether a static next action is already executable
    use the phase-wide System One tool palette
    ask System Two to compose the next concrete invocation
    validate action against tools, capabilities, dependencies, and scope
    execute exactly once
    append action and result evidence
    evaluate hard completion invariants, then ask System One about semantic completion
    persist System One's progress verdict with the action evidence
    stop for outer replan if an equivalent call returns equivalent evidence and is
        again judged incomplete
    complete subgoal, recover locally, wait, or stop for outer review
return PhaseOutcome and PhaseEvidence
```

## Static and adaptive execution

- Static actions are used when tool arguments are already known.
- Adaptive action selection is used when later arguments depend on observed results.
- Provider/program execution may optimize a phase, but it must use the same invocation,
  authorization, evidence, and interruption boundaries.
- Program success never substitutes for child-call evidence.

## Local recovery policy

Local recovery is appropriate when it preserves the same objective and authorization,
for example:

```text
artifact index has no match -> try allowed project-file search
artifact adapter returns the wrong shape -> invoke or inspect its CLI with native.bash
```

Operational success and semantic progress are separate. A zero-exit tool call may
still fail to advance the subgoal. Jev's resolved progress verdict is durable phase
evidence and is supplied to the next tactical selection. The executor does not use a
fixed retry count, but it does stop when the same tool and arguments return the same
result and Jev again judges it incomplete; that repetition contributes no new evidence.

Return to outer Check/Act when:

- Authoritative records conflict.
- The user must choose among materially different interpretations.
- A new mutation target or external effect is required.
- The phase objective is invalidated.
- Jev determines that further authorized work is not worthwhile.

## Implementation checklist

- [x] Add a `PhaseExecutor` independent of the V2 one-call Do implementation.
- [x] Add a tactical-action inference purpose and strict structured output validation.
- [x] Use one parallel Jev Noul question per registry tool to select a phase-wide
      semantic palette; leave concrete invocation composition to System Two.
- [x] Use Jev for semantic inner-Do completion checks after operational success.
- [x] Validate every selected action before invocation.
- [x] Reuse `ToolInvocationService` as the only tool execution boundary.
- [x] Check cancellation and queued steering between every action.
- [x] Persist state and evidence after every action.
- [x] Implement typed subgoal-output binding.
- [x] Implement dependency resolution and skip prevention.
- [x] Implement local completion predicates.
- [x] Implement local fallback selection.
- [x] Make all unexpected failures visible in `PhaseOutcome`.
- [x] Ensure successful writes require structured verification or a separate read-back
      completion condition.
- [x] Keep V2 available as an explicit rollback engine while V3 becomes the default
      for newly ingested tasks.
- [x] Emit nested activity events for phase, subgoal, and tactical action.

## Tests

- [x] Execute a multi-action read/verify phase.
- [x] Execute an adaptive search/edit/verify phase with output binding.
- [x] Stop dependent actions after a failed prerequisite.
- [x] Recover locally through one authorized fallback.
- [x] Return for strategic review when scope must expand.
- [x] Continue across multiple actions without planner-authored limits.
- [x] Preserve the shared nonzero Bash failure semantics through
      `ToolInvocationService`.
- [x] Interrupt safely on steering and resume under outer control.
- [x] Cancel between actions without running another tool.
- [x] Restart from a checkpoint without repeating a completed side effect.
- [x] Reject an action that was not revealed or authorized.
- [x] Allow trusted registered artifacts without imposing native-tool read-only
      classification on them.
- [x] Preserve every partial result when a later action fails.
- [x] Persist Jev's semantic progress verdict with each action and prevent equivalent
      successful-but-incomplete evidence from looping indefinitely.
- [x] Allow a CLI-backed artifact failure to pivot to an already revealed
      `native.bash` capability.

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

- 2026-09-20 — Added the `PhaseExecutor`, tactical inference purpose,
  per-action checkpoint/evidence recording, typed bindings, local fallback, scope/tool
  rejection, steering/cancellation boundaries, and nested progress events. Full suite:
  437 passed.
- 2026-09-20 — Added System One tactical tool choice, per-transition durable
  checkpoints, project search/read tools, and hard protection for `.alphonse`
  internal state. System Two now generates arguments for one selected tool rather
  than reconsidering the complete reveal set.
- 2026-09-23 — Removed fixed tactical retry, phase-call, and phase-duration limits.
  Tool-owned execution timeouts and output bounds remain part of each tool contract.
- 2026-09-24 — Made Jev's per-action progress judgment durable tactical evidence.
  Equivalent successful calls that return equivalent incomplete results now stop for
  outer replan, while changed results and different Bash/CLI strategies remain allowed.
