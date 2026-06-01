from __future__ import annotations

import logging
from pathlib import Path

import pytest

from alphonse.agent.cli import _handle_repl_message_command
from alphonse.agent.cli import build_parser
from alphonse.agent.cli import _handle_repl_logs_command
from alphonse.agent.cli import _configure_logging
from alphonse.agent.cli import _get_cli_logging_state
from alphonse.agent.cli import _resolve_cli_log_file
from alphonse.agent.cli import _set_cli_log_destination
from alphonse.agent.cli import AgentSupervisor
import alphonse.agent.cli as cli_module
from alphonse.agent.nervous_system.migrate import apply_schema
from alphonse.agent.nervous_system.seed import (
    BOOTSTRAP_CLI_SERVICE_USER_ID,
    apply_seed,
)
from alphonse.agent.nervous_system.senses.bus import Bus
from alphonse.agent.nervous_system.senses.cli import build_cli_user_message_signal


class _FakeActionRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def execute(self, action_key: str | None, context: dict) -> None:
        self.calls.append({"action_key": action_key, "context": dict(context)})


def _reset_logging() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass
    _set_cli_log_destination("file")
    cli_module._CLI_LOG_DESTINATION_OVERRIDE = None
    cli_module._CLI_LOG_ENABLED_OVERRIDE = None


@pytest.fixture(autouse=True)
def _clean_cli_logging():
    _reset_logging()
    yield
    _reset_logging()


def test_build_cli_user_message_signal_matches_canonical_contract() -> None:
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
    assert payload["contract_type"] == "canonical_inbound_event"
    assert payload["service_key"] == "cli"
    assert payload["provider_user_id_from"] == BOOTSTRAP_CLI_SERVICE_USER_ID
    assert payload["channel_target"] == "cli"
    assert payload["text"] == "hello from repl"
    assert payload["metadata"]["source"] == "test"


def test_build_cli_user_message_signal_includes_bootstrap_identity(tmp_path: Path, monkeypatch) -> None:
    db_path = tmp_path / "nerve-db"
    monkeypatch.setenv("NERVE_DB_PATH", str(db_path))
    apply_schema(db_path)
    apply_seed(db_path)

    signal = build_cli_user_message_signal(
        text="hello from repl",
        correlation_id="corr-cli-identity",
        metadata={"source": "test"},
    )

    payload = signal.payload
    assert payload["provider_user_id_from"] == BOOTSTRAP_CLI_SERVICE_USER_ID
    assert payload["display_name"] == "Alex"
    assert payload["metadata"]["bootstrap_admin_user_id"]
    assert payload["metadata"]["bootstrap_admin_user_id"] != "owner-1"


def test_repl_message_command_emits_signal_and_invokes_runtime(capsys) -> None:
    bus = Bus()
    action_runtime = _FakeActionRuntime()

    handled = _handle_repl_message_command("message hello world", bus=bus, action_runtime=action_runtime)

    assert handled is True
    emitted = bus.get(timeout=0.01)
    assert emitted is not None
    assert emitted.type == "sense.cli.message.user.received"
    assert emitted.payload["text"] == "hello world"
    assert len(action_runtime.calls) == 1
    assert action_runtime.calls[0]["action_key"] == "handle_conscious_message"
    out = capsys.readouterr().out
    assert "Queued sense.cli.message.user.received corr=" in out


def test_repl_message_command_rejects_empty_text(capsys) -> None:
    handled = _handle_repl_message_command("message", bus=Bus(), action_runtime=_FakeActionRuntime())

    assert handled is True
    out = capsys.readouterr().out
    assert "Usage: message <text>" in out


def test_repl_message_command_handles_missing_bus(capsys) -> None:
    handled = _handle_repl_message_command("message hello", bus=None, action_runtime=None)

    assert handled is True
    out = capsys.readouterr().out
    assert "Unable to queue message: REPL message bus is unavailable." in out


def test_repl_logs_off_disables_handlers(monkeypatch, capsys) -> None:
    _reset_logging()
    monkeypatch.setenv("ALPHONSE_CLI_LOG_ENABLED", "true")

    handled = _handle_repl_logs_command("logs off")

    assert handled is True
    out = capsys.readouterr().out
    assert "CLI logs disabled" in out
    assert _get_cli_logging_state().destination == "none"
    assert any(isinstance(handler, logging.NullHandler) for handler in logging.getLogger().handlers)


def test_repl_logs_file_configures_file_handler(monkeypatch, tmp_path: Path, capsys) -> None:
    _reset_logging()
    log_path = tmp_path / "cli.log"
    monkeypatch.setenv("ALPHONSE_CLI_LOG_ENABLED", "false")
    monkeypatch.setenv("ALPHONSE_CLI_LOG_FILE", str(log_path))

    handled = _handle_repl_logs_command("logs file")

    assert handled is True
    out = capsys.readouterr().out
    assert f"CLI logs enabled: file {log_path}" in out
    assert _get_cli_logging_state().enabled is True
    assert any(isinstance(handler, logging.FileHandler) for handler in logging.getLogger().handlers)
    assert log_path.parent.exists()


def test_repl_logs_stderr_configures_stream_handler(monkeypatch, capsys) -> None:
    _reset_logging()
    monkeypatch.setenv("ALPHONSE_CLI_LOG_ENABLED", "true")

    handled = _handle_repl_logs_command("logs stderr")

    assert handled is True
    out = capsys.readouterr().out
    assert "CLI logs enabled: stderr" in out
    handlers = logging.getLogger().handlers
    assert any(isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler) for handler in handlers)


def test_repl_logs_path_and_status(monkeypatch, tmp_path: Path, capsys) -> None:
    _reset_logging()
    log_path = tmp_path / "cli.log"
    monkeypatch.setenv("ALPHONSE_CLI_LOG_ENABLED", "true")
    monkeypatch.setenv("ALPHONSE_CLI_LOG_FILE", str(log_path))

    assert _handle_repl_logs_command("logs path") is True
    assert _handle_repl_logs_command("logs status") is True

    out = capsys.readouterr().out
    assert f"CLI log file: {log_path}" in out
    assert f"CLI logs: file {log_path}" in out


def test_repl_logs_invalid_command_prints_usage(capsys) -> None:
    _reset_logging()

    handled = _handle_repl_logs_command("logs nope")

    assert handled is True
    out = capsys.readouterr().out
    assert "Usage: logs off|file|stderr|path|status" in out


def test_repl_log_singular_alias_is_supported(monkeypatch, tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "cli.log"
    monkeypatch.setenv("ALPHONSE_CLI_LOG_ENABLED", "true")
    monkeypatch.setenv("ALPHONSE_CLI_LOG_FILE", str(log_path))

    assert _handle_repl_logs_command("log file") is True
    assert _handle_repl_logs_command("log status") is True

    out = capsys.readouterr().out
    assert f"CLI logs enabled: file {log_path}" in out
    assert f"CLI logs: file {log_path}" in out
    assert "CLI logs enabled for REPL session" in log_path.read_text(encoding="utf-8")


def test_configure_logging_can_use_env_defaults(monkeypatch, tmp_path: Path) -> None:
    _reset_logging()
    log_path = tmp_path / "logs" / "cli.log"
    monkeypatch.setenv("ALPHONSE_CLI_LOG_ENABLED", "true")
    monkeypatch.setenv("ALPHONSE_CLI_LOG_DESTINATION", "file")
    monkeypatch.setenv("ALPHONSE_CLI_LOG_FILE", str(log_path))
    monkeypatch.setenv("ALPHONSE_CLI_LOG_LEVEL", "WARNING")

    _configure_logging(None)

    root = logging.getLogger()
    assert root.level == logging.WARNING
    assert any(isinstance(handler, logging.FileHandler) for handler in root.handlers)
    assert log_path.parent.exists()


def test_parser_log_level_default_uses_cli_env(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_CLI_LOG_LEVEL", "ERROR")

    args = build_parser().parse_args(["status"])

    assert args.log_level == "ERROR"


def test_cli_log_file_relative_path_resolves_under_alphonse_root(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_CLI_LOG_FILE", "agent/logs/cli.log")

    resolved = _resolve_cli_log_file()

    assert str(resolved).endswith("/alphonse/agent/logs/cli.log")


def test_agent_supervisor_redirects_managed_agent_to_log_file(monkeypatch, tmp_path: Path, capsys) -> None:
    log_path = tmp_path / "agent.log"
    monkeypatch.setenv("ALPHONSE_CLI_LOG_ENABLED", "true")
    monkeypatch.setenv("ALPHONSE_CLI_LOG_FILE", str(log_path))
    captured: dict[str, object] = {}

    class _FakeProcess:
        pid = 1234

        def poll(self):
            return None

    def _fake_popen(cmd, stdout, stderr, text):  # noqa: ANN001
        captured["cmd"] = list(cmd)
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        captured["text"] = text
        return _FakeProcess()

    monkeypatch.setattr(cli_module.subprocess, "Popen", _fake_popen)

    supervisor = AgentSupervisor()
    supervisor.start()

    out = capsys.readouterr().out
    assert "Agent started (pid=1234)." in out
    assert str(log_path) in out
    assert captured["stdout"] is captured["stderr"]
    assert getattr(captured["stdout"], "name", "") == str(log_path)
    assert captured["text"] is True
    supervisor._close_log_handle()


def test_agent_supervisor_discards_managed_agent_output_when_logs_off(monkeypatch) -> None:
    monkeypatch.setenv("ALPHONSE_CLI_LOG_ENABLED", "true")
    _set_cli_log_destination("none")
    captured: dict[str, object] = {}

    class _FakeProcess:
        pid = 1234

        def poll(self):
            return None

    def _fake_popen(cmd, stdout, stderr, text):  # noqa: ANN001
        captured["stdout"] = stdout
        captured["stderr"] = stderr
        _ = (cmd, text)
        return _FakeProcess()

    monkeypatch.setattr(cli_module.subprocess, "Popen", _fake_popen)

    AgentSupervisor().start()

    assert captured["stdout"] == cli_module.subprocess.DEVNULL
    assert captured["stderr"] == cli_module.subprocess.DEVNULL
