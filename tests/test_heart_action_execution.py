from __future__ import annotations

from dataclasses import dataclass

from alphonse.agent.actions.runtime import ActionExecutionRuntime
from alphonse.agent.heart import Heart
from alphonse.agent.nervous_system.ddfsm import CurrentState, TransitionOutcome
from alphonse.agent.nervous_system.senses.bus import Bus, Signal


class _FakeRuntime:
    def __init__(self) -> None:
        self.states: list[tuple[int | None, str | None, str | None]] = []
        self.signals: list[tuple[str | None, str | None]] = []

    def update_state(self, state_id, state_key, state_name) -> None:  # noqa: ANN001
        self.states.append((state_id, state_key, state_name))

    def update_signal(self, signal_type, source) -> None:  # noqa: ANN001
        self.signals.append((signal_type, source))


@dataclass
class _FakeDDFSM:
    outcome: TransitionOutcome

    def handle(self, state: CurrentState, signal: Signal) -> TransitionOutcome:
        _ = state, signal
        return self.outcome


class _FakeActionRuntime(ActionExecutionRuntime):
    def __init__(self) -> None:
        self.calls: list[tuple[str | None, dict]] = []

    def execute(self, action_key: str | None, context: dict):  # noqa: ANN201
        self.calls.append((action_key, dict(context)))
        return None


def test_heart_executes_selected_action_directly(monkeypatch) -> None:
    fake_runtime = _FakeRuntime()
    monkeypatch.setattr("alphonse.agent.heart.get_runtime", lambda: fake_runtime)

    outcome = TransitionOutcome(
        matched=True,
        reason="MATCH",
        action_key="handle_conscious_message",
        next_state_id=2,
        next_state_key="executing",
        next_state_name="Executing",
    )
    bus = Bus()
    action_runtime = _FakeActionRuntime()
    heart = Heart(
        bus=bus,
        ddfsm=_FakeDDFSM(outcome=outcome),
        state=CurrentState(id=1, key="idle", name="Idle"),
        action_runtime=action_runtime,
    )

    bus.emit(Signal(type="sense.cli.message.user.received", payload={"text": "hello"}, source="cli"))
    bus.emit(Signal(type="SHUTDOWN", payload={}, source="test"))

    heart.run()

    assert len(action_runtime.calls) == 1
    action_key, context = action_runtime.calls[0]
    assert action_key == "handle_conscious_message"
    assert context["signal"].type == "sense.cli.message.user.received"
    assert heart.state.key == "executing"
