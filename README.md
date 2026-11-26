# Alphonse

Alphonse is a local-first household assistant. This repository contains the
Python daemon and Desktop app.

## Run locally

Install the Python dependencies, then launch the Desktop app from `desktop/`:

```bash
npm install
npm run tauri dev
```

Desktop connects to the daemon and starts it if necessary. See the
[Desktop development guide](desktop/README.md) for prerequisites. The installed
`alphonse` command also provides daemon lifecycle controls:

```bash
alphonse start
alphonse status
alphonse stop
```

Runtime data is stored under `~/.alphonse/` by default. Its canonical database
is `alphonse-v2.sqlite3`; user and project files are kept in `users/`.

## Desktop client

The Desktop app in `desktop/` starts and connects to the daemon. Configure
integrations, users, scheduled tasks, and settings there.

The current daemon runs the owner-only simpler conversation loop. It does not
start the legacy scheduler, event workers, integration listeners, or ordinary
outbox dispatcher. Existing scheduled and queued work is preserved but is not
automatically resumed. The owner can use one-time Desktop reminders, Gmail Inbox
counts and [content search](docs/gmail-readonly.md#desktop-search-and-summaries),
and a private [Google Calendar agenda](docs/family-calendar.md#owner-desktop-agenda)
from an authorized calendar. See the [transition inventory](docs/agent-loop/retirement-transition-2026-11-15.md)
before upgrading or recovering older work.

## Repository layout

- `alphonse/agent_v2/` — daemon, core and integrations
- `desktop/` — Tauri desktop application
- `tests/` — Python test suite
- `docs/` — design and operational documentation
