from __future__ import annotations

import subprocess

import pytest

from alphonse.agent_v2.code_mode_settings import CodeModeSettings
from alphonse.agent_v2.code_mode_settings import SQLiteCodeModeSettingsStore
from alphonse.agent_v2.core.programs import ProgramRunner
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def test_code_mode_settings_start_disabled_and_validate_bounds() -> None:
    store = SQLiteCodeModeSettingsStore(":memory:")
    assert store.get().enabled is False
    assert store.get().available is False
    with pytest.raises(ValueError, match="code_mode_timeout_seconds_invalid"):
        store.save(CodeModeSettings(timeout_seconds=301))
    with pytest.raises(ValueError, match="code_mode_max_parallel_calls_exceeds_tool_calls"):
        store.save(CodeModeSettings(max_tool_calls=2, max_parallel_calls=3))


def test_changing_docker_image_clears_verification() -> None:
    store = SQLiteCodeModeSettingsStore(":memory:")
    store.save(CodeModeSettings(enabled=True))
    assert store.mark_verification(ready=True).verification_ready is True
    assert store.save(CodeModeSettings(enabled=True, image="python:3.12-slim")).verification_ready is False


def test_unsafe_code_mode_settings_require_daemon_confirmation(tmp_path) -> None:
    users = V2UserStore(":memory:")
    admin = users.onboard(display_name="Admin", users_root=tmp_path / "users")
    runtime = build_runtime_host(user_store=users, code_mode_settings_store=SQLiteCodeModeSettingsStore(":memory:"))
    daemon = V2Daemon(runtime)
    with pytest.raises(ValueError, match="confirmation_required"):
        daemon.save_code_mode_settings(actor_user_id=admin.user_id, values={"network_disabled": False})
    saved = daemon.save_code_mode_settings(actor_user_id=admin.user_id, values={"network_disabled": False}, acknowledge_unsafe=True)
    assert saved["weakened_protections"] == ["network access"]


def test_runner_snapshots_persisted_resource_and_isolation_settings(monkeypatch) -> None:
    settings = CodeModeSettings(enabled=True, verification_ready=True, memory_mb=384, cpu_count=1.5, pid_limit=99, tmpfs_mb=96, network_disabled=False, read_only_filesystem=False, run_as_non_root=False, drop_all_capabilities=False, no_new_privileges=False)
    runner = ProgramRunner(settings_provider=lambda: settings)
    captured: dict[str, object] = {}
    monkeypatch.setattr(runner, "_available_for", lambda _settings: True)
    class FakeBridge:
        def __init__(self, *args, **kwargs): pass
        def start(self): pass
        def stop(self): pass
    monkeypatch.setattr("alphonse.agent_v2.core.programs.runner._BridgeServer", FakeBridge)
    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, '{"kind":"final","result":{}}\n', "")
    monkeypatch.setattr("alphonse.agent_v2.core.programs.runner.subprocess.run", fake_run)
    result = runner.run(source="async def main(tools):\n    return {}", invocation_service=object())
    command = captured["command"]
    assert result["status"] == "success"
    assert "--network" not in command and "--read-only" not in command and "--user" not in command
    assert ["--memory", "384m"] == command[command.index("--memory"):command.index("--memory") + 2]
    assert ["--pids-limit", "99"] == command[command.index("--pids-limit"):command.index("--pids-limit") + 2]
