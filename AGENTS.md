## Project scope
- Atrium is a FastAPI service that exposes Rex status over HTTP and a minimal HTMX UI.

## Local setup
- Run the server with `uvicorn interfaces.http.main:app --reload`.
- Tests live in `tests/` and use pytest.

## Code style
- Prefer small, explicit functions and avoid introducing new dependencies for UI.
- Use HTMX patterns for UI interactions; avoid custom JavaScript unless necessary.

## Repo layout
- `interfaces/http/` holds the web entrypoints.
- `core/context/` provides time + system context snapshots.
- `rex/` contains cognition, voice, and persona assets.
