# Alphonse Agent

Alphonse Agent is a self-hosted domestic infrastructure designed to host **Alphonse**,
a resident digital butler that is proactive, educational, and protective.

Alphonse Agent is not a smart-home gadget, nor a generic assistant.
It is the persistent environment in which Alphonse exists, observes, learns, and serves.

---

## What is Alphonse?

Alphonse is the resident butler of Alphonse Agent.

Alphonse:
- observes household state without intruding,
- provides context before advice,
- educates through opportunity,
- protects by detecting anomalies, not by enforcing behavior.

Alphonse is governed by an explicit constitution that defines his role, limits, tone,
and ethical orientation.

---

## What Alphonse Agent Is

Alphonse Agent is:

- a **local-first** system
- designed to be **self-hosted**
- owned and controlled by the household
- extensible through multiple interfaces (apps, voice, services)
- built to evolve gradually and deliberately

Alphonse Agent prioritizes trust, clarity, and restraint over automation volume.

---

## What Alphonse Agent Is Not

Alphonse Agent is not:

- a cloud-dependent assistant
- a surveillance system
- a command-and-control AI
- a replacement for human judgment

---

## Repository Structure

This repository hosts the **Alphonse Agent**, the runtime environment where Alphonse lives.

Key files:

- `CONSTITUTION.md` — Alphonse’s founding charter and behavioral contract
- `README.md` — This document
- `docs/` — Design notes, vision, and long-term ideas
- `docs/architecture.md` — Alphonse architecture and boundaries
- `docs/channel_integration_blueprint.md` — Reusable channel integration scaffolding (Telegram reference for Discord/others)
- `docs/refactor_roadmap.md` — Cleanup and separation roadmap
- `docs/message_io_contract.md` — Normalized inbound/outbound adapter contract

Alphonse HTTP chat integration (MVP) uses `POST /agent/message`.
Web UI outbound push can subscribe to `GET /agent/events` (SSE).

Configuration is driven by environment variables in `alphonse/agent/.env`.
Provider routing is controlled by:

- `ALPHONSE_LLM_PROVIDER` (`opencode`, `ollama`, `openai`, `llamafarm`)
- provider-specific base URL/model/auth environment variables

Code will be introduced incrementally once identity and boundaries are clearly defined.

---

## Status

This project is in its foundational phase.

The current focus is:
- defining identity,
- establishing ethical and behavioral constraints,
- and creating a stable base for future capabilities.

---

## Running Locally

Expose Alphonse Agent to the local network with:

```bash
python -m alphonse.agent.main
```

## Timed Signals / Scheduler

TimedSignals fire via the Timer sense and the heart loop. You can run them
in-process (with `alphonse.agent.main`) or as a dedicated loop via the CLI.

### Run Telegram bot + scheduler (single process)

```bash
export TELEGRAM_BOT_TOKEN=...
export TELEGRAM_ALLOWED_CHAT_IDS=123456789
python -m alphonse.agent.main
```

### CLI harness (local testing)

Start the full agent loop directly:

```bash
python -m alphonse.agent.main
```

Send a message into the same cortex pipeline as Telegram:

```bash
python -m alphonse.agent.cli say "Recuérdame hacer ejercicio en 1 min" --chat-id local --channel cli
```

Start the interactive CLI REPL:

```bash
python -m alphonse.agent.cli repl
```

Inside the REPL you can inspect or change routing strategy:

```text
alphonse> routing get
alphonse> routing set multi_pass
```

Inside the REPL you can also manage a **managed** agent process (started by the REPL):

```text
alphonse> agent start
alphonse> agent status
alphonse> agent restart
alphonse> agent stop
```

Run the dispatcher loop (separate process):

```bash
python -m alphonse.agent.cli run-scheduler
```

Check timed signal status:

```bash
python -m alphonse.agent.cli status
```

### Local Audio Output Tool (POC)

Manual run:

```bash
python -m alphonse.tools.local_audio_output --text "Hola mundo"
```

Example tool-call payload:

```json
{
  "tool": "local_audio_output.speak",
  "args": { "text": "Hello World", "blocking": false }
}
```

### SSH Terminal Tool (`ssh_terminal`)

Run remote SSH commands through Paramiko using the `ssh_terminal` tool.

Prerequisites:

- Install dependency: `paramiko` (already included in `requirements.txt`)
- Enable tool in env:

```bash
ALPHONSE_ENABLE_SSH_TERMINAL=true
```

Optional env controls:

```bash
ALPHONSE_SSH_TERMINAL_DEFAULT_TIMEOUT_SECONDS=30
ALPHONSE_SSH_TERMINAL_MAX_TIMEOUT_SECONDS=600
ALPHONSE_SSH_TERMINAL_CONNECT_TIMEOUT_SECONDS=10
ALPHONSE_SSH_TERMINAL_ALLOW_AGENT=true
ALPHONSE_SSH_TERMINAL_LOOK_FOR_KEYS=true
ALPHONSE_SSH_TERMINAL_STRICT_HOST_KEY=false
ALPHONSE_SSH_TERMINAL_KNOWN_HOSTS_PATH=
```

Example tool-call payload:

```json
{
  "tool": "ssh_terminal",
  "args": {
    "host": "192.168.1.20",
    "username": "pi",
    "command": "uname -a",
    "timeout_seconds": 30
  }
}
```

### Scratchpad Storage Root

Scratchpad files can be pinned to a single workspace directory:

```bash
ALPHONSE_SCRATCHPAD_ROOT=/Users/alex/Code\ Projects/alphonse-workdirs/dumpster/scratchpad
```

If unset, Alphonse uses:
1. sandbox alias `dumpster` + `/scratchpad` (when enabled)
2. fallback `data/scratchpad`

### Jobs Storage Root

Scheduled job files can be pinned to the same workdir strategy:

```bash
ALPHONSE_JOBS_ROOT=/Users/alex/Code\ Projects/alphonse-workdirs/dumpster/jobs
```

If unset, Alphonse uses:
1. sandbox alias `dumpster` + `/jobs` (when enabled)
2. fallback `data/jobs`

Password auth example:

```json
{
  "tool": "ssh_terminal",
  "args": {
    "host": "10.0.0.15",
    "username": "admin",
    "password": "REDACTED",
    "command": "systemctl status ssh",
    "connect_timeout_seconds": 8
  }
}
```

### Acceptance milestones (must pass)

Marker 1 — TimedSignals end-to-end
- Telegram: "Recuérdame irme a bañar en 1 min" schedules, then reminder arrives after 1 minute (3/3)
- CLI: schedule reminder for 1 min, then CLI prints reminder (3/3)

Marker 2 — No amnesia in clarifications
- User: "Recuérdame bañarme" → Assistant: "¿Cuándo?" → User: "en 10 min" → schedules successfully (Telegram + CLI)

Marker 3 — Plan schema stable
- Cortex returns structured result (`reply_text` + plan(s)) and logs show it

Marker 4 — Policy hook
- Only configured Telegram chat IDs can schedule reminders

---

## Preferences (per chat)

Alphonse stores user/chat preferences in the nerve DB so they persist across restarts.
Deployment defaults come from environment settings, but per-chat overrides live in SQLite.

Examples (Telegram or CLI):

- "Háblame de tú" → sets `address_style=tu`
- "Háblame de usted" → sets `address_style=usted`
- "Habla en español" / "Speak English" → sets `locale`
- "Sé más formal" / "Be more casual" → sets `tone`

Preferences are keyed per principal (currently `channel_chat`), and the renderer uses them
for reminder phrasing immediately after they are set.

## Onboarding

Alphonse uses a two-phase onboarding model:

- Primary onboarding (out-of-box):
  - Runs once to bootstrap the first admin user.
  - Captures initial display name and records global bootstrap completion.
- Secondary onboarding (subsequent users):
  - Runs per new user/channel after primary onboarding is complete.
  - Captures per-user profile defaults and links identity progressively.

Primary onboarding and secondary onboarding are intentionally separated so each can evolve
independently without mixing first-run bootstrap concerns with household growth flows.

### Introduce + Authorize (Telegram)

You can introduce and authorize a new user directly inside a Telegram group chat.
Alphonse uses the replied-to user's Telegram `user_id` as the stable channel address.

Flow:

1. In a Telegram group with Alphonse, reply to the new user's message.
2. Say: "Alphonse, please meet Gaby" (or "Introduce and authorize Gaby on Telegram").
3. Alphonse will:
   - Create the user record if needed.
   - Link the user's Telegram `user_id` to the `channels` registry.
   - Mark the channel as enabled for communication.

If you do not reply to a message, Alphonse will ask for the Telegram chat id.

### Onboarding + Location Persistence (nerve-db)

New persistence tables:

- `onboarding_profiles`
- `location_profiles`
- `device_locations`

These are managed via store modules:

- `/Users/alex/Code Projects/alphonse-agent/alphonse/agent/nervous_system/onboarding_profiles.py`
- `/Users/alex/Code Projects/alphonse-agent/alphonse/agent/nervous_system/location_profiles.py`

### Onboarding + Location API Endpoints

Onboarding:

- `GET /agent/onboarding/profiles`
- `GET /agent/onboarding/profiles/{principal_id}`
- `POST /agent/onboarding/profiles`
- `DELETE /agent/onboarding/profiles/{principal_id}`

Locations:

- `GET /agent/locations`
- `GET /agent/locations/{location_id}`
- `POST /agent/locations`
- `DELETE /agent/locations/{location_id}`

Device location stream/snapshots:

- `GET /agent/device-locations`
- `POST /agent/device-locations`

### Tool Configs (Secrets / API Keys)

Store tool API keys or configs in `nerve-db` and manage them via:

- `GET /agent/tool-configs`
- `GET /agent/tool-configs/{config_id}`
- `POST /agent/tool-configs`
- `DELETE /agent/tool-configs/{config_id}`

CLI:

```bash
python -m alphonse.agent.cli tool-configs list --tool-key geocoder
python -m alphonse.agent.cli tool-configs upsert --tool-key geocoder --name google --config-json '{"api_key":"..."}'
python -m alphonse.agent.cli tool-configs show <config_id>
python -m alphonse.agent.cli tool-configs delete <config_id>
```

### Google Geocoding (optional)

If you want to normalize addresses into lat/lng, set:

`GOOGLE_MAPS_API_KEY`

The geocoder tool is registered as `geocoder` and uses the Google Maps Geocoding API.

### CLI Commands

Onboarding profile CRUD:

```bash
python -m alphonse.agent.cli onboarding list --state in_progress
python -m alphonse.agent.cli onboarding show <principal_id>
python -m alphonse.agent.cli onboarding upsert <principal_id> --state in_progress --primary-role admin --next-steps home_location work_location
python -m alphonse.agent.cli onboarding delete <principal_id>
```

Location profile CRUD + device positions:

```bash
python -m alphonse.agent.cli locations list --principal-id <principal_id>
python -m alphonse.agent.cli locations upsert <principal_id> --label home --address-text "123 Main St" --lat 20.67 --lng -103.35
python -m alphonse.agent.cli locations device-add <device_id> --principal-id <principal_id> --lat 20.68 --lng -103.34 --source alphonse_link
python -m alphonse.agent.cli locations device-list --device-id <device_id>
python -m alphonse.agent.cli locations delete <location_id>
```

---

## LangGraph Cortex

Alphonse's conversation orchestration runs in `alphonse/agent/cortex/graph.py`.
Session state is persisted per chat in SQLite using the `cortex_sessions` table
inside the nerve DB.

To add a new intent:

1. Update `alphonse/agent/cortex/intent.py` with classification and slot logic.
2. Add a response or execution path in `alphonse/agent/cortex/graph.py`.
3. Wire any new tools in `alphonse/agent/extremities/`.

---

## Configuration

Runtime behavior is configured via `alphonse/agent/.env` and defaults in code.

In `production`, set `OPENAI_API_KEY` for the OpenAI provider.

---

## Supabase Integration

Set these environment variables (see `alphonse/agent/.env.example`):

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` (preferred) or `SUPABASE_ANON_KEY`
- `ALPHONSE_WEBHOOK_SECRET` (optional; required for webhook auth)
- `FCM_CREDENTIALS_JSON` (Firebase service account JSON, for push notifications)
- `VAPID_PRIVATE_KEY` (Web Push private key)
- `VAPID_PUBLIC_KEY` (Web Push public key)
- `VAPID_EMAIL` (Web Push contact email)

Generate VAPID keys with your preferred Web Push tooling and export:
- `VAPID_PUBLIC_KEY`
- `VAPID_PRIVATE_KEY`

Legacy push-device endpoints under `/api/*` were removed with the legacy
`interfaces/` service. Use the active `/agent/*` API in
`alphonse/infrastructure/api.py`.

---

## Notification Worker

Run the separate notification worker to dispatch due events:

```bash
python workers/notification_worker.py
```

---

## Habit crystallization (MVP)

Alphonse can crystallize a habit on `pairing.requested`. The flow uses LangGraph:
router → planner → executor, then persists the plan/run/receipts in the nerve DB.

Quick dev trigger:

```bash
curl -s -X POST http://<alphonse_host>:8001/pair/start \
  -H "Content-Type: application/json" \
  -d '{"device_name":"Test Phone","device_platform":"android"}'
```

Habits and runs live in the nerve DB (SQLite). See tables `habits`, `plan_runs`,
and `delivery_receipts`.

---

## Philosophy

Alphonse Agent is built with the belief that:

> A system that knows when to remain silent
> is more intelligent than one that speaks constantly.

---

## License

License to be defined.
