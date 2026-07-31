# Alphonse Architecture

## Overview

Alphonse is the agent runtime. It interprets messages, transitions state, and produces
intentions. Everything else is I/O. Telegram, Web UI, CLI, and future A2A channels are
extremities that send normalized messages into Alphonse and receive formatted responses.

The same message should take the same path through the interpreter regardless of channel.

## Boundaries

### Agent Core (Alphonse)
- Interpreter, FSM, signals, senses, actions, intentions, narration.
- No HTTP, templates, or UI dependencies.
- Owns the message-in to routing-decision flow.

### Extremities (I/O Channels)
- Telegram, Web UI, CLI, A2A, and future integrations.
- Responsible for receiving messages, normalizing into `MessageEvent`, calling the
  interpreter, and formatting outputs.
- Must not mutate FSM state directly or bypass the interpreter.

### Infrastructure
- Server hosting, config, env, logging, workers, and storage.
- Provides runtime and transport wiring, not intelligence.

## Web UI Positioning

The Web UI is an extremity. It observes and interacts with Alphonse the same way as
Telegram. It must not bypass the interpreter/router.

Short-term it can remain in-process as an extremity; long-term it should be a separate
service that communicates via a clear API boundary.

## Naming and Mental Model

Alphonse is the brain. Extremities are senses and hands. The interpreter routes all
messages. UIs are clients, not owners.

## Proposed Structure

```
alphonse/
  core/
    interpretation/
    nervous_system/
    actions/
    intentions/
    mediation/
    nervous_system/senses/
  extremities/
    telegram/
    webui/
    cli/
    a2a/
  infrastructure/
    server/
    workers/
    config/
interfaces/
  webui/
scripts/
docs/
```

## Channel Integration Blueprint

For a reusable implementation template (adapter, policy, ToolSpec/runtime tool wiring,
deterministic auth gates, timed follow-ups, and testing), see:

- `docs/channel_integration_blueprint.md`

## Observability and Harness

For graph-layer observability design, retention policy, and operational diagnostics:

- `docs/observability_and_harness.md`

## V2 Persistence

Alphonse v2 uses one local, WAL-enabled SQLite database for relational state. Separate
tables retain their own schemas and lifecycles: identity and settings, projects and
channel sessions, inbound and outbound queues, conversation events and cursors,
questions and task checkpoints, scheduled tasks and executions, automations,
communication threads, integrations, asset metadata, and artifacts.

SQLite conversation events are the canonical user-visible timeline. Markdown ledgers
are intentionally a different projection: compactable model context stored on disk
under a user/project scope. Binary attachments, project files, user/project context,
and agent configuration also remain on disk; relational tables store their metadata
and paths.

The daemon evaluates a fixed 30-day retention policy daily for terminal operational
rows. Pending and retryable work, conversation events, and Markdown memory are never
removed by this policy. Checkpoints store actionable task state but do not embed the
Markdown ledger; resumed tasks reload the current scoped ledger when processing
starts.

Users and projects can enqueue concurrently. SQLite WAL, a busy timeout, consistent
transactions, and per-scope ledger locks protect those writes. CAPD task execution is
still serial; parallel CAPD processing is a separate future capability.
