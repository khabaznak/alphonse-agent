# Atrium Server

Atrium is a self-hosted domestic infrastructure designed to host **Alphonse**,
a resident digital butler that is proactive, educational, and protective.

Atrium is not a smart-home gadget, nor a generic assistant.
It is the persistent environment in which Alphonse exists, observes, learns, and serves.

---

## What is Alphonse?

Alphonse is the resident butler of Atrium.

Alphonse:
- observes household state without intruding,
- provides context before advice,
- educates through opportunity,
- protects by detecting anomalies, not by enforcing behavior.

Alphonse is governed by an explicit constitution that defines his role, limits, tone,
and ethical orientation.

---

## What Atrium Is

Atrium is:

- a **local-first** system
- designed to be **self-hosted**
- owned and controlled by the household
- extensible through multiple interfaces (apps, voice, services)
- built to evolve gradually and deliberately

Atrium prioritizes trust, clarity, and restraint over automation volume.

---

## What Atrium Is Not

Atrium is not:

- a cloud-dependent assistant
- a surveillance system
- a command-and-control AI
- a replacement for human judgment

---

## Repository Structure

This repository hosts the **Atrium Server**, the core environment where Alphonse lives.

Key files:

- `CONSTITUTION.md` — Alphonse’s founding charter and behavioral contract
- `README.md` — This document
- `docs/` — Design notes, vision, and long-term ideas
- `docs/architecture.md` — Alphonse architecture and boundaries
- `docs/refactor_roadmap.md` — Cleanup and separation roadmap

Configuration lives in `config/alphonse.yaml`. The `mode` controls which inference
provider Alphonse uses:

- `test` uses Ollama locally.
- `production` uses an online provider (OpenAI by default) and expects an
  `OPENAI_API_KEY` environment variable.

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

Expose Atrium to the local network with:

```bash
uvicorn interfaces.http.main:app --host 0.0.0.0 --port 8000
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

Inside the REPL you can inspect intent routing:

```text
alphonse> debug intent "Cuál es mi nombre?"
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

`config/alphonse.yaml` controls runtime behavior. The key switch is `mode`:

```yaml
mode: test

providers:
  test:
    type: ollama
  production:
    type: openai
```

In `production`, set `OPENAI_API_KEY` for the OpenAI provider.

---

## Supabase Integration

Set these environment variables (see `.env.example`):

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY` (preferred) or `SUPABASE_ANON_KEY`
- `ATRIUM_WEBHOOK_SECRET` (optional; required for webhook auth)
- `FCM_CREDENTIALS_JSON` (Firebase service account JSON, for push notifications)
- `VAPID_PRIVATE_KEY` (Web Push private key)
- `VAPID_PUBLIC_KEY` (Web Push public key)
- `VAPID_EMAIL` (Web Push contact email)

Generate VAPID keys with:

```bash
python scripts/generate_vapid_keys.py
```

API endpoints:

- `POST /api/family-events`
- `GET /api/family-events`
- `GET /api/family-events/{id}`
- `PATCH /api/family-events/{id}`
- `DELETE /api/family-events/{id}`
- `POST /api/webhooks/family-events`
- `POST /api/push-devices`
- `DELETE /api/push-devices/{id}`

`/api/push-devices` accepts `platform` values like `android` or `web`.
For web push, send the subscription object as `token`.

`owner_id` for push devices references the `family` table.

Webhook auth (optional): if `ATRIUM_WEBHOOK_SECRET` is set, include the
`X-Atrium-Webhook-Secret` header in webhook requests.

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

Atrium is built with the belief that:

> A system that knows when to remain silent
> is more intelligent than one that speaks constantly.

---

## License

License to be defined.
