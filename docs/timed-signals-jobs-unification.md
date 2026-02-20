# Timed Signals + Jobs Unification

## Goal
Use a single scheduling primitive (`timed_signals`) and treat jobs as compiled recurring timed signals.

## Core model
- `timed_signals` is the only scheduler queue.
- `job_create` writes `jobs.json` (human/admin control) and also upserts a `timed_signals` row (`signal_type=job_trigger`).
- `TimerSense` is the only clock-driven trigger path.
- `timed_signal.fired` is dispatched by `mind_layer` + `dispatch_mode`:
  - `subconscious + deterministic`: direct deterministic handling.
  - `conscious + graph`: routed into the LLM graph (`api.message_received`).

## Mind layers
- Subconscious:
  - deterministic, no planner reasoning required.
  - examples: technical housekeeping triggers, internal maintenance dispatch.
- Conscious:
  - requires inference/planning.
  - examples: reminders/jobs phrased as intentional prompts to Alphonse.

## You Just Remembered technique
At creation time (reminder/job), Alphonse stores:
- `source_instruction`: user-origin request.
- `agent_internal_prompt`: LLM paraphrase in first person and original user language.
- `prompt_artifact_id`: persistent artifact reference.

This is what future executions consume when routed through the graph.

## Delivery semantics
- Execution model: **at-least-once**.
- Duplicate protection: idempotency key at execution layer (job + run slot).
- Late trigger policy:
  - baseline acceptable lag: `30 minutes`.
  - recurring catch-up window: `max(30 minutes, 5% of recurrence period)`.
  - if late beyond window:
    - recurring signal: skip missed occurrence and reschedule next RRULE occurrence.
    - one-shot signal: mark failed (`missed dispatch window`).

## Why this design
- Single source of truth for scheduling lifecycle.
- Predictable trigger behavior across reminders and jobs.
- Better recovery after downtime with explicit catch-up rules.
- Cleaner separation between deterministic reflexes and LLM reasoning paths.
