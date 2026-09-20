# Alphonse V3: Hierarchical CAPD

Status: proposed  
Owner: Alphonse project  
Last updated: 2026-09-20

## Purpose

V3 changes CAPD's unit of work from one tactical tool selection to one meaningful,
bounded execution phase. An outer CAPD loop owns task outcomes and strategy. A
smaller tactical executor inside Do selects and runs concrete tools until the phase
is complete, blocked, interrupted, or out of budget.

This is a replacement intelligence engine built on the V2 runtime, not a rewrite of
queues, projects, memory sessions, integrations, tools, authorization, or IPC.

## Why V3 exists

V2 repeatedly runs global Plan, Do, Check, and Act around individual tool calls. That
causes excessive model calls, repeated acceptance review, short planning horizons,
and weak continuity between related actions. A task such as locating one record,
updating it, and reading it back should be one phase, not three or more CAPD cycles.

## Target architecture

```text
Message and bounded memory
          |
          v
Outer CAPD
  Check task/steering
  Act defines or amends immutable outcomes
  Plan selects a bounded phase
          |
          v
Tactical Phase Executor inside Do
  Resolve current subgoal
  Reveal relevant tools
  Choose and execute the next action
  Inspect local result and recover within policy
  Return structured phase evidence
          |
          v
Outer Check
  Evaluate the complete phase once
  Complete, replan strategically, wait, or fail safely
```

The tactical executor is CAPD-like but is not a recursive copy of the outer graph. It
cannot change acceptance criteria, expand authorized side effects, declare mission
success, or exceed its phase limits.

## Global principles

- Acceptance criteria remain immutable except for explicit user steering.
- The outer loop owns strategy; the inner loop owns mechanics.
- A phase has an objective, subgoals, completion conditions, budgets, and mutation
  scope.
- Tool outputs flow between subgoals through typed, persisted bindings.
- Tool availability is progressively revealed from capability metadata.
- Writes are sequential, authorized, and evidenced by verified read-back.
- Read-only work may be parallel only when the tool descriptor explicitly allows it.
- Steering and cancellation are checked between tactical actions.
- Local recovery is bounded; material strategy changes return to outer Act.
- Overall completion requires phase-level evidence and a final outer Check.
- The final user response is produced only after task completion is verified.

## Delivery stages

1. [Phase contracts and persisted state](01-phase-contracts.md)
2. [Bounded tactical phase executor](02-tactical-executor.md)
3. [Progressive tool revealing](03-progressive-tool-revealing.md)
4. [Outer completion routing and response](04-outer-routing.md)
5. [Compatibility, observability, evaluation, and rollout](05-rollout.md)
6. [Experimental System One Check and Act](06-system-one-check-act.md)

Stages are ordered. A later stage may be prototyped early, but it must not be declared
complete before its dependencies and exit gates pass.

## V3 success measures

Use a replayable evaluation set containing at least:

- A one-record project mutation.
- A mutation requiring discovery before editing.
- A medical-history task requiring prior-memory retrieval.
- An attachment task where OCR is relevant.
- A task where OCR is not relevant and must not be revealed.
- A tool failure with a valid local fallback.
- A conflict that requires strategic replanning.
- A task requiring a user decision.
- Steering during tactical execution.
- Restart and resume in the middle of a phase.

Compare V3 with V2 on:

- Correct task completion.
- Unrelated mutations.
- Silent failures.
- Model calls per completed task.
- Tool calls per completed task.
- Total input/output tokens.
- Time to first visible progress and final response.
- Percentage of tool schemas exposed per tactical decision.
- Recovery behavior and user-question quality.

V3 is not ready as the default unless correctness and safety are no worse than V2 and
the representative simple-task set shows a material reduction in global model calls
and latency.

## Compatibility strategy

- Add an engine selector rather than cloning the entire application.
- Reuse V2 `CoreMessage`, queues, projects, sessions, memory, tools, and integrations.
- Version V3 task checkpoints explicitly.
- Adapt legacy one-call `plan_json` checkpoints into one-subgoal legacy phases, or let
  already-running V2 tasks finish under V2.
- Keep public daemon and IPC methods compatible unless a versioned addition is needed.
- Do not silently reinterpret an active V2 task as a richer V3 plan.

## Durable working convention

Each stage file contains a checklist and exit criteria. During implementation:

1. Mark completed checklist items in the relevant file.
2. Add a dated note under `Implementation log` with the commit and important decisions.
3. Record unresolved design choices under `Open decisions`; do not leave them only in a
   conversation.
4. Run the stage's focused tests and the full suite before closing the stage.
5. Update this overview when stage boundaries or dependencies change.

## Current starting point

V2 already provides several V3 foundations:

- Immutable acceptance contracts.
- Append-only cumulative execution evidence.
- Verified exact-text mutation with hashes, diff, and post-write read-back.
- Direct and program execution modes.
- Tool descriptors with capabilities and `read_only` metadata.
- Task checkpointing, steering ingestion, cancellation, and UI activity events.

The next implementation task is Stage 1: define and test the V3 phase/state contracts
without changing default V2 behavior.
