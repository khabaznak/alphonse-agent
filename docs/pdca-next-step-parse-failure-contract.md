# PDCA Next-Step Parse Failure Contract

Date: 2026-02-25
Owner: task-mode
Scope: `next_step_node` behavior when LLM proposal output is invalid/non-parseable.

## Problem Statement
When the planner returns invalid JSON/schema for `NextStepProposal`, the current behavior can route to user-question paths that imply missing user input. This is incorrect: parse failure is an internal planner degradation, not a user slot-fill condition.

## Goals
- Treat invalid planner output as internal degradation.
- Avoid asking user clarifying questions for formatter/contract failures.
- Produce complete diagnostics for admin/provider tuning.
- Preserve deterministic, bounded recovery flow.

## Non-Goals
- Solving all model quality issues.
- Auto-switching providers in this phase.

## Recovery Policy
1. Attempt planner output normally.
2. If invalid, perform bounded retry with validator feedback.
3. If still invalid, mark degraded and fail gracefully.

Retry budget:
- `max_attempts = 2` (initial + 1 repair attempt).
- Optional third attempt can be enabled later behind config.

## State Contract
On terminal parse failure in `next_step_node`:
- `task_state.status = "failed"`
- `task_state.pdca_phase = "plan"`
- `task_state.last_validation_error = {
    "reason": "next_step_parse_failed",
    "attempts": <int>,
    "schema": "NextStepProposal",
    "provider": <string|None>,
    "model": <string|None>
  }`
- `task_state.execution_eval` MAY include degradation metadata but MUST NOT imply user-actionable retry.
- `task_state.next_user_question = None`

Rationale:
- `failed` is accurate for internal degradation.
- `waiting_user` is reserved for true user dependency (secrets/confirmation/input).

## User-Facing Behavior
Responder should return a graceful internal-error message, no clarifying question:
- Example: "I hit an internal planning error and paused this task. I logged diagnostics for the admin."

Message requirements:
- Transparent about internal issue.
- No blame on user.
- No request for JSON/format input.

## Observability Contract
Emit structured event on each invalid attempt and on terminal degradation.

### Event A: `graph.next_step.parse_invalid`
Fields:
- `correlation_id`
- `cycle`
- `attempt`
- `provider`
- `model`
- `parse_error_type`
- `validation_errors` (list)
- `raw_output_preview` (truncated/redacted)

### Event B: `graph.next_step.degraded`
Fields:
- `correlation_id`
- `cycle`
- `attempts`
- `reason = "next_step_parse_failed"`
- `provider`
- `model`
- `status = "failed"`

### Redaction Rules
- Truncate raw output previews.
- Redact known secrets/tokens from logged text.
- Do not log full credential-bearing prompts.

## Node Routing Rules
- `route_after_next_step`:
  - if `status == "failed"` -> `respond_node`
  - if `status == "waiting_user"` -> `respond_node`
  - else -> `execute_step_node`

- `route_after_act` remains unchanged for non-parse paths.

## Compatibility Notes
- Existing tests expecting parse failure -> `waiting_user` + fallback question must be updated.
- Any responder dependency on `next_user_question` when status is `failed` should be removed.

## Metrics
Add counters:
- `pdca_next_step_parse_invalid_total`
- `pdca_next_step_degraded_total`
- dimensions: provider, model

## Rollout Plan
Phase 1:
- Implement bounded retry + degraded fail contract.
- Keep existing happy path unchanged.
- Add/adjust tests.

Phase 2:
- Add dashboard + alert threshold for repeated degradation by model/provider.

## Test Cases (Minimum)
1. Invalid JSON then valid repair output -> continues normally.
2. Invalid JSON for all attempts -> status becomes `failed`, routes to `respond_node`.
3. Parse degradation response is non-question, non-slot-fill.
4. Observability events include validation details and attempt counts.
