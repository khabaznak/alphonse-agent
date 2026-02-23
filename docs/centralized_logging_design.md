# Centralized Logging and Observability

## Goal
- Route agent flow logs through one strategy and one abstraction.
- Persist structured events for debugging long-running tasks.
- Avoid ad-hoc, module-specific logging styles in core orchestration paths.

## Log Manager
- Module: `alphonse/agent/observability/log_manager.py`
- Entry points:
  - `get_log_manager()` for structured event emission.
  - `get_component_logger(component)` for drop-in `logger.info(...)` compatibility while centralizing output.

## Event sinks
- Runtime sink: standard Python logging.
- Structured sink: `observability.db` (`trace_events`) via `write_task_event(...)`.

## Standard event shape
- Core: `event`, `level`, `component`, `message`.
- Correlation: `correlation_id`, `channel`, `user_id`.
- Execution: `node`, `cycle`, `status`, `tool`, `error_code`, `latency_ms`.
- Extended payload: free-form metadata stored in `detail_json`.
- Adapter normalization:
  - `StructuredLoggerAdapter` auto-parses `key=value` tokens from log text.
  - It maps common fields into canonical columns (`correlation_id`, `cycle`, `tool`, etc.).
  - It also accepts explicit `extra={...}` data to override/augment parsed values.

## Migration strategy
1. Core loop first: Heart, incoming message handling, task-mode orchestration.
2. High-churn services next: tools and action handlers.
3. Remaining modules: swap `logging.getLogger(...)` for `get_component_logger(...)`.
4. Keep third-party/framework logs unchanged.
5. For remaining legacy log lines, rely on adapter parsing (`key=value`) until explicit event emissions are added.

## Failure guarantees
- Emit structured failure events on exceptions (`emit_exception(...)`).
- Include exception class, message, and stack excerpt.
- Ensure long-running flows write terminal diagnostics even on abort paths.
