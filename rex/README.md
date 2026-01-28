# Rex

## Nervous system DB

- Set `NERVE_DB_PATH` in `rex/agent/.env` to point at the SQLite DB.
- Relative paths are resolved from `rex/agent/`.
- A starter template is in `rex/agent/.env.example`.

## FSM schema and seed

- Apply schema: `python rex/nervous_system/migrate.py`
- Seed data: `python rex/nervous_system/seed.py`
