# Nuke Plan

## Status Legend
- `[x]` completed
- `[~]` in progress
- `[ ]` pending

## Goal
Make `nerve-db` the source of truth for behavior/config text and reduce hardcoded coupling in response and policy paths.

## Completed
- `[x]` Graph response-key rendering goes through `ResponseComposer`.
- `[x]` `plan_executor.py` major user-facing literals migrated to response keys.
- `[x]` `localization.py` template tables removed; `render_message` now delegates to safe fallback layer.
- `[x]` Startup brain health checks added (`intent catalog` + `prompt store` availability).
- `[x]` Deterministic safe fallback module created (`safe_fallbacks.py`).

## Current Phase
- `[x]` Phase 1: Principal scopes and preference precedence.
  - Added principal scope support for `system` and `office` in schema + migration.
  - Added preference resolution precedence helper:
    - `person -> channel_chat -> office -> household -> system -> default`.
  - Wired precedence helper into active read paths (`incoming`, `executor`, `timer`, `daily report`).
  - Added tests for precedence behavior.

- `[~]` Phase 2: PromptStore cardinality reduction.

## Next
- `[ ]` Phase 2: PromptStore cardinality reduction.
  - Shift from high-selector combinatorics to mostly `key + locale (+ variant)`.
  - Keep selector dimensions for explicit overrides only.

- `[ ]` Phase 3: Policy engine pluginization.
  - Introduce rule interface and registry.
  - Move Telegram-specific checks from core engine into integration rule providers.

- `[ ]` Phase 4: Prompt seed ownership cleanup.
  - Move seed payload out of runtime code paths into explicit migration/seed assets.
  - Keep runtime store logic only in `prompt_store.py`.

- `[ ]` Phase 5: Guardrails.
  - Test to fail if new user-facing literal blocks are added outside `safe_fallbacks.py`.
  - Runtime tests for unavailable-brain behavior.
