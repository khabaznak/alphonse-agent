from __future__ import annotations

from alphonse.agent.cli import _handle_repl_message_command
from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.nervous_system.senses.cli import build_cli_user_message_signal


class _FakePipeline:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def handle(self, action_key: str | None, context: dict) -> None:
        self.calls.append({"action_key": action_key, "context": dict(context)})


def test_build_cli_user_message_signal_matches_envelope_contract() -> None:
    signal = build_cli_user_message_signal(
        text="hello from repl",
        correlation_id="corr-cli-1",
        user_name="Alex",
        metadata={"source": "test"},
    )
    assert signal.type == "sense.cli.message.user.received"
    assert signal.source == "cli"
    assert signal.correlation_id == "corr-cli-1"
    payload = signal.payload
    assert payload["schema_version"] == "1.0"
    assert payload["channel"]["type"] == "cli"
    assert payload["channel"]["target"] == "cli"
    assert payload["content"]["text"] == "hello from repl"
    assert payload["metadata"]["source"] == "test"


def test_repl_message_command_emits_signal_and_invokes_pipeline(capsys) -> None:
    bus = Bus()
    pipeline = _FakePipeline()

    handled = _handle_repl_message_command("message hello world", bus=bus, pipeline=pipeline)

    assert handled is True
    emitted = bus.get(timeout=0.01)
    assert emitted is not None
    assert emitted.type == "sense.cli.message.user.received"
    assert emitted.payload["content"]["text"] == "hello world"
    assert len(pipeline.calls) == 1
    assert pipeline.calls[0]["action_key"] == "handle_conscious_message"
    out = capsys.readouterr().out
    assert "Queued sense.cli.message.user.received corr=" in out


def test_repl_message_command_rejects_empty_text(capsys) -> None:
    handled = _handle_repl_message_command("message", bus=Bus(), pipeline=_FakePipeline())

    assert handled is True
    out = capsys.readouterr().out
    assert "Usage: message <text>" in out


def test_repl_message_command_handles_missing_bus(capsys) -> None:
    handled = _handle_repl_message_command("message hello", bus=None, pipeline=None)

    assert handled is True
    out = capsys.readouterr().out
    assert "Unable to queue message: REPL message bus is unavailable." in out
