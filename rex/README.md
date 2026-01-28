# Rex

## Nervous system DB

- Set `NERVE_DB_PATH` in `rex/agent/.env` to point at the SQLite DB.
- Relative paths are resolved from `rex/agent/`.
- A starter template is in `rex/agent/.env.example`.
- Runtime DB files live under `rex/nervous_system/db/` and are ignored.

## FSM schema and seed

- Apply schema: `python rex/nervous_system/migrate.py`
- Seed data: `python rex/nervous_system/seed.py`

## Running Rex

- From repo root: `python -m rex.agent.main`
- From `rex/agent/`: `PYTHONPATH=../.. python -m rex.agent.main`

## Actions, Intentions, Extremities

Rex separates decision logic from communication.

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

1) Create a new module under `rex/senses/` that subclasses `Sense`.
2) Define `key`, `name`, `source_type`, and `signals`.
3) Implement a background producer with `start()` and `stop()` that emits to the `Bus`.
4) Run `python rex/nervous_system/migrate.py` and `python rex/nervous_system/seed.py` or call `register_senses()` and `register_signals()` to register metadata.

Minimal example:

```python
from __future__ import annotations

import threading
import time

from rex.senses.base import Sense, SignalSpec
from rex.senses.bus import Bus, Signal


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
