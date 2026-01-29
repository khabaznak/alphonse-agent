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
    senses/
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
