# PDCA Slice Queue Orchestration Plan

## Context

Alphonse currently runs PDCA execution inside a LangGraph invocation with a recursion limit safety fuse.
For long-running tasks and multi-user fairness, we want cooperative multitasking:

- execute work in short PDCA slices
- persist checkpointed task state between slices
- yield to other pending user work
- resume safely without state drift or duplication

This requires explicit coordination between:

- **LangGraph PDCA loop** (task planning/execution/progress)
- **DDFSM** (heart-level signal routing and lifecycle orchestration)
- **durable task queue/checkpoint store** (resume + fairness + idempotency)

## Goals

1. Enable long-running tasks without premature interruption from hop count.
2. Ensure fair scheduling across multiple users/conversations.
3. Keep safety guarantees via progress/budget gates and hard stop fuse.
4. Preserve deterministic resume semantics (no mixed/duplicated slices).
5. Make execution lifecycle observable and debuggable.

## Non-goals (initial rollout)

1. Full parallel execution of multiple slices at once.
2. Rewriting core PDCA planning logic.
3. Replacing the signal bus transport mechanism.

## Current baseline (important)

1. DDFSM states currently in DB: `idle`, `dnd`, `error`, `onboarding`, `shutting_down`.
2. `dnd` appears legacy in FSM routing (no transitions into it in current DB); communication policy still uses preference `communication_mode in {"dnd","sleep"}`.
3. Heart consumes signals serially from bus.
4. PDCA state exists in cognition state (`task_state`) but there is no dedicated round-robin slice queue.

## Proposed DDFSM state model

Keep:

- `idle`
- `error`
- `shutting_down`

Add:

- `executing`
- `persisting_slice`
- `rehydrating_slice`
- `waiting_user`

Deprecate:

- `dnd` as an FSM state (retain communication preference semantics separately)

## Proposed signal set

New signals:

1. `pdca.slice.requested`
2. `pdca.slice.rehydrate_requested`
3. `pdca.slice.completed`
4. `pdca.slice.persisted`
5. `pdca.waiting_user`
6. `pdca.resume_requested`
7. `pdca.failed`

Existing signals remain (e.g. inbound messages, timer/job signals, shutdown).

## Transition skeleton

1. `idle` + `pdca.slice.requested` -> `rehydrating_slice`
2. `rehydrating_slice` + `action.succeeded` -> `executing`
3. `executing` + `pdca.slice.completed` -> `persisting_slice`
4. `persisting_slice` + `pdca.slice.persisted` -> `idle`
5. `executing` + `pdca.waiting_user` -> `waiting_user`
6. `waiting_user` + `{telegram|cli|api}.message_received` -> `rehydrating_slice`
7. `*` + `pdca.failed` -> `error`
8. `*` + `shutdown_requested` -> `shutting_down`

## Durable storage design

### `pdca_tasks`

- `task_id` (PK)
- `owner_id`
- `conversation_key`
- `session_id`
- `status` (`queued|running|waiting_user|done|failed|paused`)
- `priority` (int)
- `next_run_at`
- `lease_until`
- `worker_id`
- `slice_cycles`
- `max_cycles`
- `max_runtime_seconds`
- `token_budget_remaining`
- `failure_streak`
- `last_error`
- `created_at`
- `updated_at`

### `pdca_checkpoints`

- `task_id` (PK/FK)
- `state_json` (serialized cortex state)
- `task_state_json` (serialized PDCA state)
- `version` (optimistic concurrency)
- `updated_at`

### `pdca_events` (append-only)

- `event_id` (PK)
- `task_id`
- `event_type`
- `payload_json`
- `created_at`

## Slice execution protocol

1. Pick runnable task from queue (fair order + lease acquire).
2. Load checkpoint and rehydrate state.
3. Invoke PDCA for one bounded slice (`N` cycles or budget threshold).
4. If terminal (`done|failed|waiting_user`), update queue status accordingly.
5. If still running, persist checkpoint, emit `pdca.slice.persisted`, set `next_run_at`, requeue.
6. Release lease.

## Fair scheduling policy (initial)

Order by:

1. `priority DESC`
2. `next_run_at ASC`
3. `updated_at ASC`

This yields practical round-robin fairness while allowing priority override.

## Safety policy

Primary controls:

1. Progress gate (must make net progress within window).
2. No-progress cycle cap.
3. Failure streak cap.
4. Task wall-clock budget.
5. Token budget.

Emergency fuse:

1. LangGraph recursion limit remains finite and high enough for a slice.

## Idempotency and anti-duplication rules

1. Exactly one active lease per task.
2. Checkpoint writes require matching `version` (compare-and-swap).
3. Ignore stale `pdca.resume_requested` if checkpoint version moved forward.
4. All events include `correlation_id`, `task_id`, and `slice_id`.

## Integration points

1. `alphonse/agent/cortex/graph.py`
   - retain recursion limit as hard fuse
   - invoke graph against rehydrated checkpoint state
2. `alphonse/agent/cortex/task_mode/*`
   - expose cycle/slice boundaries for yield decisions
3. `alphonse/agent/actions/handle_incoming_message.py`
   - enqueue/update PDCA task instead of long inline run
4. `alphonse/agent/nervous_system/seed.sql`
   - add new states/signals/transitions
5. new store/service modules in `alphonse/agent/nervous_system/` or `alphonse/agent/services/`

## Rollout phases

### Phase 1: Data + interfaces (no behavior change)

1. Add schema tables for queue/checkpoints/events.
2. Add store APIs with tests.
3. Add feature flag `ALPHONSE_PDCA_SLICING_ENABLED=0`.

### Phase 2: Scheduler scaffold

1. Add queue runner with lease + fair selection.
2. Add emit/log instrumentation.
3. Keep existing inline flow as default path.

### Phase 3: Slice execution behind flag

1. Route eligible requests to queue/slice flow.
2. Persist/rehydrate checkpoints.
3. Keep fallback to legacy path on error.

### Phase 4: DDFSM lifecycle migration

1. Add new states/signals/transitions.
2. Move orchestration actions to new state model.
3. deprecate/remove `dnd` FSM state after observability confirms no usage.

### Phase 5: Hardening

1. Budget tuning (token/runtime/failure).
2. Backpressure + starvation monitoring.
3. Recovery drills (restart mid-slice, duplicate signal, stale lease).

## Test plan

1. Queue fairness across 2+ users/tasks.
2. Checkpoint resume preserves `cycle_index`, `facts`, `plan` step statuses.
3. Lease contention prevents double execution.
4. Waiting-user park/resume correctness.
5. Crash/restart recovery rehydrates last committed checkpoint.
6. Stale resume signal does not rollback state.
7. DDFSM transition coverage for new lifecycle states.

## Open decisions

1. Should queue tasks be keyed by `conversation_key` or explicit `task_id` lineage per user intent?
2. Do we allow preemption only at progress-critic boundaries, or also after each tool execution?
3. What is the default `slice_cycles` (candidate: 3-5)?
4. Which tasks get higher priority (user-interactive vs background jobs)?
5. Should waiting-user tasks auto-expire after inactivity window?

## Initial implementation checklist

1. Create DB migration for `pdca_tasks`, `pdca_checkpoints`, `pdca_events`.
2. Add typed store module with CRUD + lease methods.
3. Add queue runner skeleton service.
4. Add minimal signal/action wiring for `pdca.slice.requested` and `pdca.slice.persisted`.
5. Add feature flag guard and metrics.
6. Add integration tests for enqueue -> slice -> persist -> resume flow.
