# Observability and Harness Strategy

This document defines Alphonse's graph-layer observability model, logging strategy, and operational harness.

## Goals

- Make failures diagnosable quickly by correlation id.
- Keep runtime logs readable in production.
- Preserve enough history for trend analysis and self-improvement loops.
- Keep observability storage isolated from the nervous-system operational DB.

## Data Plane Separation

Alphonse uses a dedicated observability SQLite database.

- Env: `ALPHONSE_OBSERVABILITY_DB_PATH`
- Default: `agent/nervous_system/db/observability.db`

This avoids coupling trace-volume growth with operational state tables.

## Event Contract

Graph/task-mode emits structured events through `log_task_event(...)` with required fields:

- `ts`
- `level`
- `event`
- `correlation_id`
- `channel`
- `user_id`
- `node`
- `cycle`
- `status`

Optional fields include:

- `tool`
- `error_code`
- `latency_ms`
- domain-specific metadata

All events are emitted to runtime logs and persisted to the observability DB.

## Storage Schema

`trace_events`

- Canonical event stream.
- Indexed for:
  - `(correlation_id, created_at)`
  - `(event, created_at)`
  - `(level, created_at)`
  - `(channel, created_at)`

`trace_daily_rollups`

- Daily aggregate counters by `(day, event, level)`.
- Updated on every insert.
- Used for long-term trend visibility while raw rows are pruned.

## Retention and Rotation Policy

Policy is enforced by store maintenance:

- Non-error TTL: `ALPHONSE_OBSERVABILITY_NON_ERROR_TTL_DAYS` (default `14`)
- Error TTL: `ALPHONSE_OBSERVABILITY_ERROR_TTL_DAYS` (default `30`)
- Row cap: `ALPHONSE_OBSERVABILITY_MAX_ROWS` (default `1_000_000`)
- Maintenance interval: `ALPHONSE_OBSERVABILITY_MAINTENANCE_SECONDS` (default `21600`)

When the cap is exceeded, oldest rows are deleted first.

## Runtime Noise Control

Telegram polling noise is reduced:

- Empty `getUpdates` polls log at `DEBUG`.
- Non-empty polls log at `INFO`.
- Periodic summary at `INFO` reports:
  - `polls`
  - `empty_polls`
  - `updates_received`

This keeps signal-to-noise high while preserving diagnostics.

## Graph Observability Coverage

Current PDCA/task-mode instrumentation emits events at key transitions:

- proposal creation
- validation pass/fail
- tool success/failure
- ask-user transitions
- state updates
- completion/failure outcomes

## Harness Usage

Operationally, debugging should start from `correlation_id`:

1. Find all `trace_events` for a correlation id ordered by `created_at`.
2. Identify first failure/route divergence event.
3. Cross-check runtime logs for transport/integration details.
4. Use rollups to detect recurring tool or node-level failure patterns.

## Future: Alphonse Self-Inspection

To let Alphonse improve from its own behavior safely:

1. Add read-only observability query tools scoped to summarized windows.
2. Feed aggregates and recent error slices into planning context.
3. Require deterministic policy gates before any self-modification action.

This enables learning loops without giving unrestricted access to raw infrastructure internals.
