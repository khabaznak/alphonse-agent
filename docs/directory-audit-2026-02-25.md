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
