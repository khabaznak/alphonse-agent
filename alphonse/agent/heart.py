"""Core agent heartbeat loop."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging

from alphonse.agent.runtime import get_runtime
from alphonse.agent.cognition.intentions.intent_pipeline import (
    IntentPipeline,
    build_default_pipeline_with_bus,
)
from alphonse.agent.nervous_system.ddfsm import CurrentState, DDFSM
from alphonse.agent.nervous_system.senses.bus import Bus

SHUTDOWN = "SHUTDOWN"
RUNNING = "RUNNING"

logger = logging.getLogger(__name__)


@dataclass
class HeartConfig:
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
        pipeline: IntentPipeline | None = None,
    ) -> None:
        self.config = config or HeartConfig()
        self.bus = bus
        self.ddfsm = ddfsm
        self.state = state or CurrentState(id=self.config.initial_state_id)
        self.ctx = ctx
        self.signal = RUNNING
        self._runtime = get_runtime()
        self._runtime.update_state(self.state.id, self.state.key, self.state.name)
        self.pipeline = pipeline or build_default_pipeline_with_bus(self.bus)

    def run(self) -> None:
        """Run the vital loop until a shutdown is requested or received.
        """
        while self.signal != SHUTDOWN:
            signal = self.bus.get(timeout=None)
            logger.info(
                "Heart received signal state=%s/%s signal=%s",
                self.state.id,
                self.state.key,
                signal.type,
            )
            self._runtime.update_signal(signal.type, signal.source)
            if signal.type == SHUTDOWN:
                break
            outcome = self.ddfsm.handle(self.state, signal, self.ctx)
            logger.info(
                "Heart transition outcome matched=%s action=%s next_state=%s",
                getattr(outcome, "matched", None),
                getattr(outcome, "action_key", None),
                getattr(outcome, "next_state_key", None),
            )
            if outcome:
                self.pipeline.handle(
                    outcome.action_key,
                    {
                        "state": self.state,
                        "signal": signal,
                        "outcome": outcome,
                        "ctx": self.ctx,
                    },
                )
                next_id = getattr(outcome, "next_state_id", None) or getattr(outcome, "id", None) or self.state.id
                next_key = getattr(outcome, "next_state_key", None) or getattr(outcome, "key", None) or self.state.key
                next_name = getattr(outcome, "next_state_name", None) or getattr(outcome, "name", None) or self.state.name

                self.state = CurrentState(
                    id=next_id,
                    key=next_key,
                    name=next_name,
                )
                self._runtime.update_state(self.state.id, self.state.key, self.state.name)

    def stop(self) -> None:
        self.signal = SHUTDOWN
