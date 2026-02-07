# Prompt/Text Source Inventory (Phase 1)

Date: 2026-02-07

## Goal

Move all user-facing text and LLM instructions to nerve-db-backed templates. If nerve-db is unavailable, fail fast with graceful shutdown messaging; do not continue with a broad conversational fallback mode.

## Scope Reviewed

- `alphonse/agent/cognition/prompt_store.py`
- `alphonse/agent/cognition/localization.py`
- `alphonse/agent/cognition/plan_executor.py`
- `alphonse/agent/cortex/graph.py`

## Quick Signal Counts

Approximate string-literal density (`quoted strings with spaces`):

- `prompt_store.py`: 59
- `localization.py`: 94
- `plan_executor.py`: 38
- `graph.py`: 170

These counts are heuristic but useful for prioritization.

## Findings

### 1) `prompt_store.py` is currently both runtime store and seed content owner

Why it smells:

- File has hardcoded seed text, detector rules, and prompt templates.
- This is useful for bootstrap, but still code-owned content.

Keep:

- Store API (`get_template`, `upsert_template`, selection scoring).
- Versioning/audit behavior.

Refactor/Nuke:

- Move large seed blocks into migration seed payloads (DB-first seed), not inline Python tuples.
- Keep only a minimal bootstrap seed set in code for catastrophic DB bootstrap.

### 2) `localization.py` is mostly legacy template ownership

Why it smells:

- Contains many direct templates and key-specific branch logic.
- Duplicates content that should be in PromptStore.

Keep:

- Small compatibility shim while migration finishes.

Refactor/Nuke:

- Migrate all keys to PromptStore and route through `ResponseComposer`.
- Reduce `localization.py` to deterministic safety fallback module only.

### 3) `plan_executor.py` still embeds direct user-facing literals

Examples:

- Pairing responses (`"Pairing not found."`, `"Missing OTP."`).
- Policy and error strings (`"No estoy autorizado..."`, `"Lo siento..."`).
- LAN status messages.

Keep:

- Policy enforcement, typed plan validation, dispatch pipeline.

Refactor/Nuke:

- Replace direct strings with `ResponseSpec` keys and compose via PromptStore-backed path.
- Keep only minimal safety fallback literals in one dedicated module.

### 4) `graph.py` still has many response-key decisions and fallback paths

Why it smells:

- Routing is map+catalog based, but response fallback logic still includes hardcoded control branches.
- Many key-level branch points are fine, but final wording should always be PromptStore-driven.

Keep:

- Routing and slot/FSM control flow.

Refactor/Nuke:

- Centralize unknown/clarify/policy fallback rendering in one response policy layer.
- Keep graph focused on state transitions and plan emission, not message text concerns.

## Recommended Nuke Order

1. `plan_executor.py` user-facing literals -> `ResponseSpec` keys.
2. `graph.py` fallback response assembly -> response policy/composer layer.
3. `localization.py` deprecate to safety-only fallback.
4. `prompt_store.py` split seed content from runtime logic.

## Safety Baseline (must remain in code)

Keep only deterministic startup/runtime failure handling for brain availability checks.

- `system.unavailable.catalog`
- `system.unavailable.prompt_store`
- `system.unavailable.nerve_db`

Runtime policy:

- If `nerve-db` schema or connectivity checks fail, log explicit reason and terminate agent startup gracefully.
- If a mid-run critical `nerve-db` failure happens, emit a structured fatal event and stop processing new messages.
- Do not silently switch to broad localization/template fallback for normal responses when the brain DB is unavailable.

## Acceptance Criteria for Phase 2

- New PRs cannot add user-facing literals outside approved safety fallback module.
- All response keys used by graph/executor resolve via PromptStore in normal operation.
- DB-down path is explicit and deterministic.
