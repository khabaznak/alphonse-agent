# Alphonse

Alphonse is a local-first personal agent. This repository contains the v2
runtime and desktop client.

## Run locally

Install the Python dependencies, then start the daemon:

```bash
python -m alphonse.agent_v2.daemon
```

In another terminal, start the terminal interface:

```bash
python -m alphonse.agent_v2.tui
```

The installed `alphonse` command provides the same v2 lifecycle controls:

```bash
alphonse start
alphonse status
alphonse stop
```

Runtime data is stored under `~/.alphonse/` by default. Its canonical database
is `alphonse-v2.sqlite3`; user and project files are kept in `users/`.

## Desktop client

The desktop client in `desktop/` starts and connects to the v2 daemon. Configure
integrations, users, scheduled tasks, and settings there or through the TUI.

## Repository layout

- `alphonse/agent_v2/` — v2 daemon, core, integrations, and interfaces
- `desktop/` — Tauri desktop application
- `tests/` — v2 test suite
- `docs/` — v2 design and operational documentation
