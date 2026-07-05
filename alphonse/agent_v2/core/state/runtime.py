"""Global state store for the Alphonse v2 core."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock

from alphonse.agent_v2.core.state.ddfsm import AVAILABLE
from alphonse.agent_v2.core.state.ddfsm import CurrentState
from alphonse.agent_v2.core.state.ddfsm import DDFSM
from alphonse.agent_v2.core.state.ddfsm import TransitionOutcome
from alphonse.agent_v2.core.state.ddfsm import build_default_ddfsm


@dataclass
class StateStore:
    _current: CurrentState
    _lock: RLock = field(default_factory=RLock, repr=False)

    def snapshot(self) -> CurrentState:
        with self._lock:
            return self._current

    def set(self, state: CurrentState) -> None:
        with self._lock:
            self._current = state

    def apply(self, outcome: TransitionOutcome) -> CurrentState:
        if not outcome.matched:
            return self.snapshot()
        if outcome.next_state_id is None or outcome.next_state_key is None or outcome.next_state_name is None:
            return self.snapshot()
        state = CurrentState(
            id=outcome.next_state_id,
            key=outcome.next_state_key,
            name=outcome.next_state_name,
        )
        self.set(state)
        return state

    def reset(self, fsm: DDFSM | None = None, key: str = AVAILABLE) -> CurrentState:
        machine = fsm or build_default_ddfsm()
        state = machine.current_state_for_key(key)
        self.set(state)
        return state


_DEFAULT_FSM = build_default_ddfsm()
State = StateStore(_DEFAULT_FSM.current_state_for_key(AVAILABLE))


def get_state() -> CurrentState:
    return State.snapshot()


def set_state(state: CurrentState) -> None:
    State.set(state)


def reset_state(key: str = AVAILABLE, *, fsm: DDFSM | None = None) -> CurrentState:
    return State.reset(fsm=fsm, key=key)

