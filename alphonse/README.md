# Alphonse

## Nervous system DB

- Set `NERVE_DB_PATH` in `alphonse/agent/.env` to point at the SQLite DB.
- Relative paths are resolved from `alphonse/agent/`.
- A starter template is in `alphonse/agent/.env.example`.
- Runtime DB files live under `alphonse/agent/nervous_system/db/` and are ignored.

## FSM schema and seed

- Apply schema: `python alphonse/agent/nervous_system/migrate.py`
- Seed data: `python alphonse/agent/nervous_system/seed.py`

## Running Alphonse

- From repo root: `python -m alphonse.agent.main`
- From `alphonse/agent/`: `PYTHONPATH=../.. python -m alphonse.agent.main`

## Actions, Intentions, Extremities

Alphonse separates decision logic from communication.

- **Actions** execute pure logic and return an `ActionResult` describing intent.
- **Intentions** represent semantic goals (e.g., notify the household).
- **Narration** optionally explains an intention in human language.
- **Extremities** are IO endpoints that manifest intentions (push, logs, etc.).

### Example flow

Doorbell pressed while user sleeping:

1. Signal: `doorbell_pressed`
2. FSM: transitions to `household_sleeping`
3. Action: decides to `NOTIFY_HOUSEHOLD` with `urgency=high`
4. Intention: `NOTIFY_HOUSEHOLD` payload includes who/where
5. Narration: optionally produces a calm message
6. Extremity: push notification delivers the message

### Cognitive mapping

- Signals = stimuli
- FSM = nervous system
- Actions = motor planning
- Intentions = semantic goals
- Narration = social cognition
- Extremities = physical expression

## How to add a new Sense

1) Create a new module under `alphonse/agent/nervous_system/senses/` that subclasses `Sense`.
2) Define `key`, `name`, `source_type`, and `signals`.
3) Implement a background producer with `start()` and `stop()` that emits to the `Bus`.
4) Run `python alphonse/agent/nervous_system/migrate.py` and `python alphonse/agent/nervous_system/seed.py` or call `register_senses()` and `register_signals()` to register metadata.

Minimal example:

```python
from __future__ import annotations

import threading
import time

from alphonse.nervous_system.senses.base import Sense, SignalSpec
from alphonse.nervous_system.senses.bus import Bus, Signal


class ExampleSense(Sense):
    key = "example"
    name = "Example Sense"
    source_type = "service"
    signals = [
        SignalSpec(key="example_pulse", name="Example Pulse"),
    ]

    def __init__(self) -> None:
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, bus: Bus) -> None:
        def run() -> None:
            while not self._stop_event.is_set():
                bus.emit(Signal(type="example_pulse", source=self.key))
                time.sleep(5)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
```

## Service Integrations as Tools

Integration APIs should be exposed as capability tools for planning and execution.
Rule: if Alphonse integrates with a service, that service's API surface should map to tool-like operations.

Examples:
- Telegram integration can expose tools such as:
  - `getTelegramMessageFile(file_id: string)` -> resolves Telegram file metadata/path for download.
  - `downloadTelegramFile(file_path: string)` -> retrieves raw bytes for transcription or vision processing.
- Similar mapping should apply to other integrations (email, calendar, web providers, etc.).

Guidelines:
- Keep tool names explicit and action-oriented.
- Keep parameter names aligned to provider API fields.
- Return typed data objects (metadata + payload references), not UI text.
- Document each integration's tool surface in the same style.

### Telegram raw update shape

Telegram updates are JSON objects. The root includes `update_id` and one event payload such as
`message`, `edited_message`, `callback_query`, or `message_reaction`.

Media messages include `file_id` handles, not local file paths.
