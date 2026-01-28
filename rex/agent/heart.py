"""Core agent heartbeat loop."""

from __future__ import annotations

from dataclasses import dataclass

from rex.nervous_system.ddfsm import CurrentState, DDFSM
from rex.senses.bus import Bus

SHUTDOWN = "SHUTDOWN"
RUNNING = "RUNNING"


@dataclass
class HeartConfig:
    tick_seconds: float = 0.1
    nervous_system_db_path: str | None = None
    initial_state_id: int = 1


class Heart:
    def __init__(
        self,
        config: HeartConfig | None = None,
        *,
        bus: Bus,
        ddfsm: DDFSM,
        state: CurrentState | None = None,
        ctx: object | None = None,
    ) -> None:
        self.config = config or HeartConfig()
        self.bus = bus
        self.ddfsm = ddfsm
        self.state = state or CurrentState(id=self.config.initial_state_id)
        self.ctx = ctx
        self.signal = RUNNING

    def run(self) -> None:
        """Run the vital loop until a shutdown is requested or received.

        The heart blocks on the signal bus with a timeout. When the timeout
        expires with no signals, `tick()` is called for housekeeping.
        """
        while self.signal != SHUTDOWN:
            signal = self.bus.get(timeout=self.config.tick_seconds)
            if signal is None:
                # Heartbeat / housekeeping tick when no signals arrive.
                self.tick()
                continue
            if signal.type == SHUTDOWN:
                break
            outcome = self.ddfsm.handle(self.state, signal, self.ctx)
            if outcome:
                next_id = getattr(outcome, "next_state_id", None) or getattr(outcome, "id", None) or self.state.id
                next_key = getattr(outcome, "next_state_key", None) or getattr(outcome, "key", None) or self.state.key
                next_name = getattr(outcome, "next_state_name", None) or getattr(outcome, "name", None) or self.state.name

                self.state = CurrentState(
                    id=next_id,
                    key=next_key,
                    name=next_name,
                )

    def tick(self) -> None:
        """One iteration of the heart loop."""
        # Placeholder for future work.
        return None

    def stop(self) -> None:
        self.signal = SHUTDOWN
