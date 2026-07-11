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
- `alphonse/agent_v2/daemon.py` — Long-lived v2 daemon host
- `alphonse/agent_v2/runtime.py` — Shared v2 runtime construction
- `alphonse/agent_v2/integrations/` — Optional v2 provider integrations

Configuration is driven by environment variables in `alphonse/agent/.env`.
Provider routing is controlled by:

- `ALPHONSE_LLM_PROVIDER` (`openai`, `openai_codex`, `github_copilot`, `opencode`, `ollama`, `llamafarm`)
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

## Alphonse v2 Runtime

Alphonse v2 uses a daemon-owned runtime. The daemon owns CAPD processing, the
scheduled-task worker, optional integrations, durable inbound messages, and
outbound delivery. The TUI is an interface client and does not need to remain
open for Telegram messages or scheduled tasks to be processed.

### Start the v2 daemon

Run the daemon as a foreground process:

```bash
python -m alphonse.agent_v2.daemon
```

After installing the project in the virtual environment, the daemon can also
be managed with the short lifecycle commands:

```bash
source .venv/bin/activate
alphonse start
alphonse start tui
alphonse status
alphonse stop
```

`alphonse start` runs the daemon in the background and waits for its local
health socket. `alphonse start tui` starts or reuses the daemon and then opens
the TUI while leaving the daemon running when the TUI closes. `alphonse status`
reports daemon, queue, scheduler, and outbound status. Daemon output is
written to `~/.alphonse/v2-daemon.log`.

The daemon uses a local Unix socket for interface communication. Its default
runtime files are stored under `~/.alphonse/`:

- `v2-daemon.sock` — local daemon IPC socket
- `v2-messages.sqlite3` — durable inbound message queue
- `v2-outbox.sqlite3` — durable outbound message queue
- `v2-scheduled-tasks.sqlite3` — scheduled task definitions and executions
- `v2-integrations.sqlite3` — integration configuration and local secrets
- `v2-inference.sqlite3` — daemon-wide inference provider and model selection
- `agent-config/` — editable `CoreContext.md` and `Philosophy.md` snapshots

Override paths with these environment variables when needed:

```dotenv
ALPHONSE_V2_SOCKET_PATH=
ALPHONSE_V2_MESSAGES_DB_PATH=
ALPHONSE_V2_OUTBOX_DB_PATH=
ALPHONSE_V2_SCHEDULE_DB_PATH=
ALPHONSE_V2_INTEGRATIONS_DB_PATH=
ALPHONSE_V2_AGENT_CONFIG_DIR=
```

### Start the v2 TUI

In a second terminal:

```bash
python -m alphonse.agent_v2.tui
```

The TUI attaches to an already-running local daemon. If no daemon is running,
it starts an embedded daemon for the default out-of-the-box experience.

Use `/integrations` inside the TUI to configure optional providers. TUI setup
and configuration changes are applied to the daemon-owned runtime.

Use `/model-provider` to select the inference provider and `/model` to select
the model used by new v2 tasks. The initial provider is OpenAI Codex: Alphonse
reads the locally signed-in Codex CLI model catalog and validates a choice with
the installed CLI before saving it. The selection applies to TUI, Telegram, and
scheduled tasks; a CAPD task already in progress keeps its original model.

Use `/agent-config` to edit the global `Core Context` or `Philosophy` markdown
used by v2 CAPD prompts. These files live under `~/.alphonse/agent-config/` by
default; set `ALPHONSE_V2_AGENT_CONFIG_DIR` to relocate them. The daemon reads
them at startup, so save changes and then run `alphonse stop` followed by
`alphonse start` before they affect new tasks.

### Configure Telegram in v2

1. Start the v2 daemon.
2. Start the v2 TUI.
3. Enter `/integrations`.
4. Select Telegram and configure an opaque integration id such as `telegram-home`.
5. Enter the bot token, Telegram user id, allowed chat ids, and enable the integration.
6. Send a fresh message to the bot.

Telegram supports text messages and text responses in the current v2 slice. The
integration also projects presence through typing indicators and message reactions.
Provider polling, CAPD processing, and outbound delivery continue when the TUI is closed
as long as the daemon remains running.

### v2 scheduled tasks

Scheduled tasks created through Telegram preserve their originating channel, so
the triggered response can return to Telegram. The v2 scheduler is part of the
daemon process:

```bash
python -m alphonse.agent_v2.daemon
```

The daemon must remain running for scheduled tasks to trigger. The task definition,
execution record, inbound message, and outbound response are stored separately so
work can be inspected after a restart.

The v2 daemon is the active development runtime. The legacy commands below use
the v1 Heart and timed-signal pipeline and are retained for v1 compatibility.

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

### CLI logs

The CLI keeps user-facing REPL output separate from runtime logs. By default, CLI logs are written to `alphonse/agent/logs/cli.log`; this includes output from a managed agent started with `alphonse> agent start`.

Useful REPL commands:

```text
alphonse> logs status
alphonse> logs path
alphonse> logs off
alphonse> logs file
alphonse> logs stderr
```

The singular form also works, for example `log status`.

Persistent defaults live in `alphonse/agent/.env`:

```dotenv
ALPHONSE_CLI_LOG_ENABLED=true
ALPHONSE_CLI_LOG_DESTINATION=file
ALPHONSE_CLI_LOG_FILE=agent/logs/cli.log
ALPHONSE_CLI_LOG_LEVEL=INFO
```

### Admin LLM Provider Auth

Alphonse supports several LLM provider auth paths. V1 uses exactly one active
provider selected by `ALPHONSE_LLM_PROVIDER`; it does not automatically fall
back between providers.

Supported provider IDs:

- `openai` — OpenAI REST API with `OPENAI_API_KEY`
- `openai_codex` — ChatGPT/Codex subscription access through the official Codex CLI session
- `github_copilot` — GitHub Copilot access through GitHub OAuth/device auth and `COPILOT_GITHUB_TOKEN`
- `opencode`
- `ollama`
- `llamafarm`

List current auth status with secrets redacted:

```bash
python -m alphonse.agent.cli llm-auth list
```

Print the exact env lines needed to select a provider:

```bash
python -m alphonse.agent.cli llm-auth select --provider openai
python -m alphonse.agent.cli llm-auth select --provider openai_codex
python -m alphonse.agent.cli llm-auth select --provider github_copilot
```

Smoke test a configured provider:

```bash
python -m alphonse.agent.cli llm-auth smoke --provider openai
python -m alphonse.agent.cli llm-auth smoke --provider openai_codex
python -m alphonse.agent.cli llm-auth smoke --provider github_copilot
```

The same commands are available inside the interactive REPL:

```text
alphonse> llm-auth list
alphonse> llm-auth select --provider openai_codex
alphonse> llm-auth smoke --provider openai_codex
```

`llm-auth select` prints dotenv lines; it does not edit `.env` automatically.
Apply the selected env values in `alphonse/agent/.env`, then restart the running
Alphonse process or the managed REPL agent:

```text
alphonse> agent restart
```

For `openai_codex`, install the official Codex CLI and authenticate once as the
admin before selecting the provider:

```bash
codex login
# or
codex login --device-auth
```

For `github_copilot`, complete the GitHub OAuth/device auth flow and put the
resulting token in ignored env as `COPILOT_GITHUB_TOKEN`. Do not store OpenAI,
Codex, or Copilot credentials in SQLite or tool configs.

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

### Qwen TTS Backend Setup

Use this when you want Alphonse local audio tools to run with Qwen3-TTS instead of macOS `say`.

1. Activate the same virtualenv used to run Alphonse:

```bash
cd "/Users/alex/Code Projects/atrium-server"
source .venv/bin/activate
```

2. Install dependencies into that env:

```bash
pip install -U qwen-tts soundfile transformers torchaudio
```

3. Configure `alphonse/agent/.env`:

```bash
ALPHONSE_TTS_BACKEND=qwen
ALPHONSE_QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
ALPHONSE_QWEN_TTS_DEVICE_MAP=auto
ALPHONSE_QWEN_TTS_DTYPE=float16
ALPHONSE_QWEN_TTS_SPEAKER=Ryan
ALPHONSE_QWEN_TTS_LANGUAGE=Auto
# optional style/tone guidance:
ALPHONSE_QWEN_TTS_INSTRUCT=
```

4. Smoke test the backend:

```bash
python -c "import qwen_tts, soundfile, transformers, torchaudio; print('qwen-tts deps ok')"
```

```bash
python - <<'PY'
from alphonse.agent.tools.local_audio_output import LocalAudioOutputRenderTool
print(LocalAudioOutputRenderTool().execute(text="Hola, soy Alphonse.", format="m4a"))
PY
```

```bash
python - <<'PY'
from alphonse.agent.tools.local_audio_output import LocalAudioOutputSpeakTool
print(LocalAudioOutputSpeakTool().execute(text="Hello, this is Alphonse.", blocking=True))
PY
```

If dependencies/model are missing, the tool returns deterministic code `qwen_backend_unavailable`.

Fallback for continuity:

```bash
ALPHONSE_TTS_BACKEND=say
```

#### High-Quality Stability Ladder (macOS arm64)

Run tier validation locally and keep the highest tier that passes repeatability and latency budgets.

Tier profiles:

- `stable`: `0.6B + cpu + float32`
- `balanced`: `1.7B + auto + float16`
- `strict`: `balanced` + voice instruction style

Run 10 sequential calls with a 10s p95 budget:

```bash
python -m alphonse.tools.qwen_tts_stability_check --tier stable --runs 10 --blocking --p95-budget-seconds 10
python -m alphonse.tools.qwen_tts_stability_check --tier balanced --runs 10 --blocking --p95-budget-seconds 10
python -m alphonse.tools.qwen_tts_stability_check --tier strict --runs 10 --blocking --p95-budget-seconds 10
```

Exit code `0` means pass, `1` means fail. The command prints JSON with `median_seconds`, `p95_seconds`, and failure details.

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

### Tool Configs (Secrets / API Keys)

Store tool API keys or configs in `nerve-db` and manage them via CLI:

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

### SearXNG Web Search (optional)

Alphonse includes two read-only web tools:

- `web.search` performs structured search through the SearXNG HTTP Search API.
- `web.fetch` retrieves readable text from a specific `http` or `https` URL.

`web.search` is a client only. It does not start or install SearXNG. You must run a
SearXNG service separately and point Alphonse at it. For a local Docker Desktop
setup, the expected endpoint is usually `http://127.0.0.1:8080`.

Configure the Alphonse process environment, usually in `alphonse/agent/.env`, then
restart Alphonse so the running process loads the values:

```dotenv
SEARXNG_BASE_URL=http://127.0.0.1:8080
SEARXNG_TIMEOUT_SECONDS=10
ALPHONSE_WEB_FETCH_TIMEOUT_SECONDS=10
ALPHONSE_WEB_FETCH_MAX_CHARS=12000
```

The SearXNG instance must also allow JSON output, because `web.search` always calls
SearXNG with `format=json`. In the SearXNG `settings.yml`, include `json` under
`search.formats`:

```yaml
search:
  formats:
    - html
    - json
```

For a local SearXNG Docker compose setup, define the host expected by the compose
file in the SearXNG project's own `.env`:

```dotenv
SEARXNG_HOST=127.0.0.1
```

Then start or restart SearXNG from that SearXNG project directory:

```bash
docker compose up -d
```

Validate the service before retrying Alphonse:

```bash
curl 'http://127.0.0.1:8080/search?q=stoic&format=json'
```

Common failure meanings:

- `searxng_base_url_missing`: the running Alphonse process did not load `SEARXNG_BASE_URL`.
- Connection refused from `127.0.0.1:8080`: SearXNG is not running, Docker Desktop is not running, or the service is on a different port.
- `403 Forbidden` from the curl command: SearXNG is running, but JSON output is not enabled.
- Docker `Cannot connect to the Docker daemon`: Docker Desktop is not running. On macOS, ignore Linux-only `usermod` instructions.

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

## Notification Worker

Run the separate notification worker to dispatch due events:

```bash
python workers/notification_worker.py
```

---

## Philosophy

Alphonse Agent is built with the belief that:

> A system that knows when to remain silent
> is more intelligent than one that speaks constantly.

---

## License

License to be defined.
