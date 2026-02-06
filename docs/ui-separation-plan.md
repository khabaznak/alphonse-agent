# UI Separation Refactor Plan

## Goals
- Move the WebUI into its own standalone project while keeping the agent service focused on runtime + APIs.
- Preserve the existing HTMX UI behavior.
- Establish explicit HTTP boundaries between UI and agent data/operations.

## Current Coupling (Summary)
- `interfaces/http/main.py` serves HTML templates, static assets, and API routes in the same FastAPI app.
- UI templates live under `interfaces/http/templates/` and static assets under `interfaces/static/`.
- UI handlers directly import and call core repositories and stores, e.g.:
- `core.repositories.*`
- `core.identity_store`, `core.nerve_store`, `core.settings_store`
- UI also directly accesses agent internals for status and API calls (via `ALPHONSE_API_BASE_URL`).

## Target Architecture
- **Agent Service (atrium-server)**
- Owns data stores and business logic.
- Exposes a versioned HTTP API for all UI actions and reads.
- **UI Service (new repo, e.g. `alphonse-ui`)**
- Standalone FastAPI + HTMX server.
- Contains templates and static assets.
- Talks only to Agent HTTP API.
- No direct imports of `core` or `alphonse` packages.

## High-Level Migration Strategy
- Phase 1: Stabilize and expand HTTP API in agent.
- Phase 2: Create UI service that mirrors existing UX but uses only HTTP.
- Phase 3: Remove UI code from agent repo.

## Phase 1: Stabilize Agent API
### 1.1 Inventory and Map UI Needs
- Source: `interfaces/http/main.py`
- Identify all routes that currently call `core.*` or `alphonse.*`.
- Define equivalent REST endpoints in agent API.

### 1.2 Create Agent API Endpoints
Add endpoints under `/api/v1` (or similar). Suggested coverage:
- `GET /api/v1/settings` `POST /api/v1/settings` `PATCH /api/v1/settings/{id}` `DELETE /api/v1/settings/{id}`
- `GET /api/v1/family` `POST /api/v1/family` `PATCH /api/v1/family/{id}`
- `GET /api/v1/push/devices` `POST /api/v1/push/devices` `DELETE /api/v1/push/devices/{id}`
- `GET /api/v1/identity/persons|groups|channels|prefs|presence` and CRUD for each
- `GET /api/v1/nerve/signals|states|transitions|senses|queue|trace` and CRUD for mutable entities
- `GET /api/v1/plans/kinds|versions|executors|instances` and CRUD for mutable entities
- `POST /api/v1/nerve/inspector/resolve` (state+signal -> resolved transition)

Notes:
- The existing `/api/*` in `interfaces/http/routes/api.py` can be moved/extended, but should be versioned.
- Use consistent response envelopes, e.g. `{ \"data\": ... }` for lists/objects.
- Preserve any `HX-Trigger` semantics in UI by using UI-side triggers instead of server-side triggers if possible.

### 1.3 Auth and Security
- Keep `x-alphonse-api-token` (or standardize to `Authorization: Bearer ...`).
- Ensure agent API returns 401/403 with useful error messages.

### 1.4 Backward Compatibility
- Keep the existing UI working during Phase 1 by calling in-process functions.
- Add tests for the new API endpoints (pytest).

## Phase 2: Create the UI Project
### 2.1 Create New Repo
- New repo name: `alphonse-ui` (or `atrium-ui`).
- Base stack: FastAPI + Jinja + HTMX.

### 2.2 Move UI Assets
- Move templates from `interfaces/http/templates/`.
- Move static files from `interfaces/static/`.
- Keep the structure intact to avoid template breakage.

### 2.3 Create UI Service App
- New `main.py` with app wiring:
- `Jinja2Templates` pointing to templates directory.
- `StaticFiles` mounted at `/static`.
- Route structure mirrors existing UI endpoints.

### 2.4 Implement API Client
- Replace direct imports of `core.*` with HTTP client calls.
- Use a small internal client module, e.g. `ui_client.py`.
- Environment vars in UI:
- `ALPHONSE_API_BASE_URL` (required)
- `ALPHONSE_API_TOKEN` (optional)

### 2.5 UI Route Refactor
Convert each UI endpoint to use the API client:
- `GET /family` -> `GET /api/v1/family`
- `POST /family` -> `POST /api/v1/family`
- `GET /nerve/*` -> `GET /api/v1/nerve/*`
- `POST /nerve/*` -> `POST /api/v1/nerve/*`
- Same pattern for identity, plans, settings, etc.

### 2.6 UI Build and Run
- New run command: `uvicorn ui.main:app --reload` (example).
- Provide `.env.example` for UI with API base/token.

## Phase 3: Remove UI From Agent Repo
- Delete `interfaces/http/templates/` and `interfaces/static/` from agent repo.
- Reduce `interfaces/http/main.py` to API-only, or remove entirely.
- Keep `interfaces/http/routes/api.py` (or move to `interfaces/http/routes/v1.py`).
- Update README with two services:
- Agent service: `uvicorn interfaces.http.main:app --reload`
- UI service: `uvicorn ui.main:app --reload`

## Suggested File/Module Layout (New UI Repo)
- `ui/main.py`
- `ui/routes/*.py`
- `ui/clients/alphonse_api.py`
- `ui/templates/*`
- `ui/static/*`

## Migration Checklist
- Agent API endpoints exist for all current UI actions.
- UI routes no longer import `core.*` or `alphonse.*`.
- Both services can run independently using `.env` configuration.
- Manual verification of all UI pages.

## Risks and Mitigations
- **Risk:** UI depends on internal DBs (identity/nerve) that lack stable API.
- **Mitigation:** Implement API endpoints first; enforce UI-only HTTP access.
- **Risk:** Two services increase dev complexity.
- **Mitigation:** Provide a `docker-compose` or `make dev` later, or a simple `scripts/dev.sh`.

## Immediate Next Step
- I can draft the API endpoint list in detail and start implementing the agent-side `/api/v1` routes first.
