# OpenCode LLM Replacement Plan (Alphonse)

## Goal
- Make the current local LLM path (`mistral:7b-instruct` via Ollama) replaceable with an OpenCode-backed provider, with minimal surface-area changes in Alphonse.
- Deliver this as a drop-in provider swap first, without changing cognition behavior.

## What We Confirmed (Current State)
- Alphonse currently builds an Ollama client from env vars in `/Users/alex/Code Projects/atrium-server/alphonse/agent/cognition/skills/interpretation/skills.py`.
- The LLM interface contract is effectively `complete(system_prompt, user_prompt) -> str`.
- Message handling currently calls `_build_llm_client()` and falls back to `None` on provider construction failure in `/Users/alex/Code Projects/atrium-server/alphonse/agent/actions/handle_incoming_message.py`.
- Default model remains `mistral:7b-instruct` in:
  - `/Users/alex/Code Projects/atrium-server/config/alphonse.yaml`
  - `/Users/alex/Code Projects/atrium-server/alphonse/config/__init__.py`

## OpenCode Modes (From Docs)
### 1) Serve mode (recommended for Alphonse/Python)
- `opencode serve` runs a headless HTTP server (default `127.0.0.1:4096`) with OpenAPI docs at `/doc`.
- Supports auth via `OPENCODE_SERVER_PASSWORD` (and optional username override).
- Exposes session/message endpoints we can call directly from Python.

### 2) SDK mode
- JS/TS SDK (`@opencode-ai/sdk`) can:
  - Start server + client (`createOpencode()`), or
  - Connect as client-only (`createOpencodeClient({ baseUrl })`) to an existing server.
- Since Alphonse is Python, SDK mode implies a Node sidecar or wrapper service.

## Decision
- Use **OpenCode serve mode** as the primary integration path.
- Do not require a Node sidecar in v1.
- Keep SDK mode as an optional future path for richer orchestration.

## Target Architecture (Drop-in)
1. Add new provider client: `OpenCodeClient` with `complete(system_prompt, user_prompt) -> str`.
2. `OpenCodeClient` talks to `opencode serve` over HTTP:
   - Create/reuse session
   - Send message via session message endpoint
   - Return assistant text
3. Swap provider selection in `_build_llm_client()` using config/env toggle.
4. Keep downstream cognition code unchanged.

## Provider Selection Plan
- Add explicit provider selection env/config:
  - `ALPHONSE_LLM_PROVIDER=ollama|openai|opencode` (default `ollama`)
- Add OpenCode settings:
  - `OPENCODE_BASE_URL` (default `http://127.0.0.1:4096`)
  - `OPENCODE_SERVER_USERNAME` (default `opencode`)
  - `OPENCODE_SERVER_PASSWORD` (required when server auth enabled)
  - `OPENCODE_MODEL` (format `provider_id/model_id`, example: `ollama/mistral:7b-instruct`)
  - `OPENCODE_TIMEOUT_SECONDS`
- Keep existing Ollama env vars untouched for backward compatibility.

## Phased Execution Plan
### Phase 0: Interface freeze
- Lock the drop-in contract (`complete(system_prompt, user_prompt)`).
- Add a small provider factory abstraction so caller code no longer imports Ollama-specific builders.

### Phase 1: OpenCode provider implementation
- Create `/Users/alex/Code Projects/atrium-server/alphonse/agent/cognition/providers/opencode.py`.
- Implement:
  - HTTP auth handling (basic auth if configured)
  - Session creation/reuse policy
  - Message submission and assistant text extraction
  - Timeouts + structured error logging

### Phase 2: Wiring
- Update `/Users/alex/Code Projects/atrium-server/alphonse/agent/actions/handle_incoming_message.py` to build provider via selector, not Ollama-only.
- Preserve existing failure mode (`None` provider fallback) for safe rollout.
- Add docs for launching OpenCode server locally.

### Phase 3: Tests
- Unit tests for:
  - provider selection logic
  - OpenCode payload mapping and response parsing
  - auth headers/basic auth behavior
  - error paths and timeout handling
- Integration test (mocked HTTP) to verify complete end-to-end `complete()` semantics.

### Phase 4: Rollout
- Start in dev with fixed model.
- Add optional canary env toggle to switch select environments/channels.
- Measure:
  - response latency
  - provider error rate
  - fallback frequency (`no_llm_client`)

## Open Questions to Resolve Before Build
- Session strategy:
  - One ephemeral session per request, or
  - Persistent session per conversation key for context continuity.
- Prompt mapping:
  - Keep separate `system` + `user` parts exactly, or prepend system into message body when needed by endpoint behavior.
- Model source of truth:
  - Set in OpenCode config only, or allow override from Alphonse env per request.

## Out of Scope for This First Pass
- Rewriting cognition prompts.
- Changing tool policy/planning loop behavior.
- Multi-provider smart routing in one request.

## Acceptance Criteria
- Switching `ALPHONSE_LLM_PROVIDER=opencode` routes all `complete()` calls through OpenCode with no cognition callsite changes.
- Switching back to `ollama` requires no code changes.
- Existing tests remain green; new provider tests cover happy path + failures.

## Source References
- OpenCode Server docs: https://opencode.ai/docs/server/
- OpenCode SDK docs: https://opencode.ai/docs/sdk/
- OpenCode Providers docs: https://opencode.ai/docs/providers/
- OpenCode Models docs: https://opencode.ai/docs/models/
