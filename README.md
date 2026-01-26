# Atrium Server

Atrium is a self-hosted domestic infrastructure designed to host **Rex**,
a resident digital butler that is proactive, educational, and protective.

Atrium is not a smart-home gadget, nor a generic assistant.
It is the persistent environment in which Rex exists, observes, learns, and serves.

---

## What is Rex?

Rex is the resident butler of Atrium.

Rex:
- observes household state without intruding,
- provides context before advice,
- educates through opportunity,
- protects by detecting anomalies, not by enforcing behavior.

Rex is governed by an explicit constitution that defines his role, limits, tone,
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

This repository hosts the **Atrium Server**, the core environment where Rex lives.

Key files:

- `CONSTITUTION.md` — Rex’s founding charter and behavioral contract
- `README.md` — This document
- `docs/` — Design notes, vision, and long-term ideas

Configuration lives in `config/rex.yaml`. The `mode` controls which inference
provider Rex uses:

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

---

## Configuration

`config/rex.yaml` controls runtime behavior. The key switch is `mode`:

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

Webhook auth (optional): if `ATRIUM_WEBHOOK_SECRET` is set, include the
`X-Atrium-Webhook-Secret` header in webhook requests.

---

## Notification Worker

Run the separate notification worker to dispatch due events:

```bash
python workers/notification_worker.py
```

---

## Philosophy

Atrium is built with the belief that:

> A system that knows when to remain silent
> is more intelligent than one that speaks constantly.

---

## License

License to be defined.
