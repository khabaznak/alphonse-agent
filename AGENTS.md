## Project scope
- Alphonse Agent is a FastAPI service that exposes Alphonse status and controls over HTTP APIs.

## Local setup
- Run the full agent with `python -m alphonse.agent.main`.
- Tests live in `tests/` and use pytest.

## Code style
- Prefer small, explicit functions and avoid introducing unnecessary dependencies.

## Repo layout
- `alphonse/infrastructure/` holds the active FastAPI API surface.
- `alphonse/` contains cognition, voice, and persona assets.
