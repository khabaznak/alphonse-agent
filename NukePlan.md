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

- `[x]` Phase 3: Policy engine pluginization.
  - Introduced rule interface and provider loading in `PolicyEngine`.
  - Moved Telegram-specific checks out of core engine into `policy/rules/telegram.py`.

- `[x]` Phase 4: Prompt seed ownership cleanup.
  - Moved prompt seed payload from `prompt_store.py` into explicit seed asset `nervous_system/resources/prompt_templates.seed.json`.
  - Kept runtime store logic in `prompt_store.py`; it now only loads and applies seed rows.

- `[x]` Phase 5: Guardrails.
  - Added test that fails on hardcoded runtime user-response literals outside `safe_fallbacks.py`.
  - Added startup runtime test for graceful exit when brain health is unavailable.
