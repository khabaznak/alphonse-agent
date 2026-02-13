## Project scope
- Alphonse Agent is a FastAPI service that exposes Alphonse status and controls over HTTP APIs.

## Local setup
- Run the server with `uvicorn interfaces.http.main:app --reload`.
- Tests live in `tests/` and use pytest.

## Code style
- Prefer small, explicit functions and avoid introducing unnecessary dependencies.

## Repo layout
- `interfaces/http/` holds legacy HTTP entrypoints.
- `core/context/` provides time + system context snapshots.
- `alphonse/` contains cognition, voice, and persona assets.
