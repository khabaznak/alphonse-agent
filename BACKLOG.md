# BACKLOG

## Terminology
- `Skills` = markdown instructions for coding agents.
- `Abilities` = Alphonse runtime intent capabilities (JSON/code), executed via tools.
- [ ] Standardize naming in docs/code/comments to use `Abilities` for runtime capabilities and avoid `Skills` ambiguity.

## API / Delivery
- [ ] Replace blocking `POST /agent/message` wait-response with async flow:
  - `POST /agent/message` -> `202 Accepted` + `correlation_id`
  - `GET /agent/message/{correlation_id}` for completion/result
- [ ] Add configurable wait env vars for API gateway calls instead of hardcoded values:
  - `ALPHONSE_API_MESSAGE_WAIT_SECONDS`
  - `ALPHONSE_API_STATUS_WAIT_SECONDS`
  - `ALPHONSE_API_TIMED_SIGNALS_WAIT_SECONDS`
- [ ] Add endpoint-level timeout logging (requested timeout, elapsed wait, correlation_id).

## Channel Architecture
- [ ] Unify API/web request-response and SSE/event delivery semantics under one contract.
- [ ] Document channel behavior matrix (telegram, cli, webui, api) for sync vs async replies.

## Message Pipeline
- [ ] Reduce remaining magic thresholds/confidence defaults in routing and centralize policy.
- [ ] Ensure clarify prompts are always context-aware and locale-consistent.
- [ ] Add regression suite for confirmation loops (`yes/si/sí`) during slot continuation.
- [ ] Refactor error apology responses to be LLM-generated per user language/preference, with minimal safe fallback strings only when model/provider fails.

## Observability
- [ ] Add trace events for `api.message_received` lifecycle:
  - received
  - dispatched
  - response published
  - timed out
- [ ] Add correlation-id debug endpoint/filter for faster incident triage.

## Integration Roadmap
- [ ] Implement `HumeIntegrationPlan.md` Phase 1A (provider + feature flags).
