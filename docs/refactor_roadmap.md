# Refactor Roadmap

## Goal

Separate Alphonse agent core from presentation layers and treat all UIs as extremities.

## Phase 1 — Boundary Clarity (no behavior changes)
- Document core vs extremities vs infrastructure.
- Ensure all I/O channels flow through the interpreter/router.
- Mark the Web UI as an extremity in docs and README.

## Phase 2 — Structural Separation
- Move agent core modules under `alphonse/core/`.
- Move UI routes/templates under a Web UI module (in-process extremity).
- Keep FastAPI as infrastructure that mounts extremities.

## Phase 3 — Externalize the Web UI
- Introduce a clean API boundary (`/messages`, `/status`).
- Replace internal calls with HTTP, even locally.
- Allow the Web UI to run as a separate service.

## Phase 4 — Cleanup
- Remove legacy Alphonse Agent-era artifacts from core.
- Rename repo and update docs/scripts as needed.

## Guardrails
- Do not bypass the interpreter/router.
- Do not mutate FSM state directly from extremities.
- Keep the same message path across channels.
