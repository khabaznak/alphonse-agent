# Directory Audit (2026-02-25)

## Scope
Audit of potentially duplicated, legacy, or placeholder directories under `/Users/alex/Code Projects/atrium-server`.

## Method
- Enumerated directories and duplicate names.
- Excluded obvious runtime/cache noise (`.git`, `__pycache__`, venv internals) from architecture decisions.
- Checked import usage and git-tracked files.

## Findings

### 1) Duplicate `nervous_system` directories
- Paths:
  - `alphonse/nervous_system`
  - `alphonse/agent/nervous_system`
- Evidence:
  - `alphonse/nervous_system` has `0` files and `0B` size.
  - Active code imports target `alphonse.agent.nervous_system.*`.
  - A stale namespace reference previously existed in `alphonse/README.md` and was corrected to `alphonse.agent.nervous_system`.
- Recommendation: **Delete `alphonse/nervous_system`** and update stale docs/import examples.
- Risk: **Low**.

### 2) Empty interface placeholders
- Paths:
  - `alphonse/agent/extremities/interfaces/devices`
  - `alphonse/agent/extremities/interfaces/webhooks`
  - `alphonse/agent/extremities/interfaces/integrations/cameras`
  - `alphonse/agent/extremities/interfaces/integrations/home-assistant`
  - `alphonse/agent/extremities/interfaces/integrations/notifications`
- Evidence:
  - All are empty (0 files).
- Recommendation: **Keep only if intentional roadmap placeholders**. Otherwise remove and recreate when implemented.
- Risk: **Low**.

### 3) `alphonse/tools` vs `alphonse/agent/tools`
- Paths:
  - `alphonse/tools`
  - `alphonse/agent/tools`
- Evidence:
  - `alphonse/tools` has only a thin CLI wrapper (`local_audio_output.py`) and package init.
  - `alphonse/agent/tools` contains the real tool implementations and registry.
  - README uses `python -m alphonse.tools.local_audio_output`.
- Recommendation: **Keep for now as compatibility shim**.
- Merge option: move wrapper to `scripts/` and update docs/entrypoints.
- Risk: **Medium** if removed now (breaks existing command usage).

### 4) `brain` vs `agent/cognition` skill-like structures
- Paths:
  - `alphonse/brain/graphs`, `alphonse/brain/skills`
  - `alphonse/agent/cognition/skills`, `alphonse/agent/cognition/narration`
- Evidence:
  - `alphonse/brain/*` is still imported by active modules (`alphonse/brain/orchestrator.py`, `alphonse/agent/lan/api.py`) and tests.
  - `agent/cognition/*` is the newer active path for planning/narration.
  - There is naming overlap (`skills`, `narration`) but not direct duplication of code.
- Recommendation: **Do not merge/delete yet**. Treat as a staged-architecture boundary.
- Risk: **High** if merged without migration plan.

### 5) Runtime/state directories in workspace
- Paths:
  - `data/jobs` (empty currently)
  - `alphonse/agent/nervous_system/db/*` (runtime DB files; currently ignored)
- Evidence:
  - These are runtime artifacts, not source.
  - `.gitignore` already excludes `data/` and `alphonse/agent/nervous_system/db/`.
- Recommendation: **Keep out of source control**; optionally standardize cleanup scripts.
- Risk: **Low**.

### 6) Local venv directory noise
- Path:
  - `.venv_pptx` (~53MB, ~973 files)
- Evidence:
  - Local environment folder under repo root.
  - Not tracked in git currently.
- Recommendation: add `.venv_pptx/` to `.gitignore` to prevent accidental adds.
- Risk: **Low**.

## Deletion/Merge Candidate Summary

### Safe delete now
- `alphonse/nervous_system`
- Optional: empty placeholder interface directories (if not intentionally reserved)

### Keep (for now)
- `alphonse/tools` (compatibility wrapper)
- `alphonse/brain/*` and `alphonse/agent/cognition/*` split until explicit migration

### Follow-up hygiene
- Update stale namespace references in `alphonse/README.md` from `alphonse.nervous_system` to `alphonse.agent.nervous_system`.
- Add `.venv_pptx/` to `.gitignore`.

## Suggested phased cleanup
1. Remove `alphonse/nervous_system` and fix stale docs.
2. Decide whether empty interface placeholders stay (roadmap) or go (clean tree).
3. Keep wrappers and legacy brain modules until a migration RFC exists.
4. Add `.venv_pptx/` ignore rule.

## Remaining Work Plan

### Phase A: Compatibility shim decision (`alphonse/tools`)
1. Inventory all invocations of `python -m alphonse.tools.local_audio_output` in docs, scripts, and automations.
2. Define target entrypoint:
   - Option A: keep shim permanently as stable compatibility API.
   - Option B: migrate to `python -m alphonse.agent.tools.local_audio_output` or dedicated script under `scripts/`.
3. If migrating, add a deprecation window:
   - Keep shim with warning logs for one release cycle.
   - Update docs/CI/examples first.
4. Remove shim only after usage reaches zero.

### Phase B: `brain` -> `agent/cognition` consolidation plan
1. Write an RFC that defines final ownership boundaries:
   - Which modules remain in `alphonse/brain`.
   - Which modules move into `alphonse/agent/cognition`.
2. Build dependency map for current `alphonse/brain/*` imports (especially `alphonse/brain/orchestrator.py` and LAN integration call sites).
3. Execute migration in small slices:
   - Move one subdomain at a time.
   - Add temporary compatibility imports where needed.
   - Keep behavior parity with tests per slice.
4. Add explicit “cutover complete” criteria:
   - No runtime imports from deprecated `alphonse/brain/*` paths.
   - All docs and tests updated.
5. Only then remove deprecated `brain` paths.

### Phase C: Runtime hygiene tooling
1. Add a small maintenance command (or make target) to clear local runtime artifacts safely:
   - `data/*` transient files
   - optional local DB snapshots/backups older than retention threshold
2. Document retention policy:
   - what is safe to delete
   - what should be preserved for debugging/auditing
3. Ensure maintenance commands never touch source files and are opt-in.
