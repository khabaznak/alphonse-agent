from __future__ import annotations

import json
import time

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry.native import BASH_TOOL_ID
from alphonse.agent_v2.core.tools.registry.native import BASH_TOOL_NAME
from alphonse.agent_v2.core.tools.registry.native import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native import execute_bash
from alphonse.agent_v2.core.tools.registry.native.bash import BASH_ARGUMENT_SCHEMA
from alphonse.agent_v2.core.tools.registry.native.bash import MAX_OUTPUT_CHARS


def test_native_registry_registers_bash_tool() -> None:
    registry = build_native_tool_registry()

    descriptor = registry.get(BASH_TOOL_NAME)

    assert descriptor is not None
    assert descriptor.tool_id == BASH_TOOL_ID
    assert descriptor.name == BASH_TOOL_NAME
    assert descriptor.argument_schema["required"] == ["command"]
    assert descriptor.capabilities == ("shell", "local_execution")




def test_bash_tool_executes_successful_command_and_captures_stdout() -> None:
    started_at = time.monotonic()
    result = execute_bash({"command": "printf hello"})
    elapsed = time.monotonic() - started_at

    assert result["exit_code"] == 0
    assert result["stdout"] == "hello"
    assert result["stderr"] == ""
    assert result["timed_out"] is False
    assert result["cwd"]
    assert result["timeout_seconds"] is None
    assert result["duration_ms"] >= 0
    assert elapsed < 2


def test_bash_tool_does_not_wait_for_background_process_pipe_eof() -> None:
    started_at = time.monotonic()
    result = execute_bash({"command": "sleep 1 &"})
    elapsed = time.monotonic() - started_at

    assert result["exit_code"] == 0
    assert result["timed_out"] is False
    assert elapsed < 0.5


def test_bash_tool_captures_stderr_and_nonzero_exit_without_raising() -> None:
    result = execute_bash({"command": "printf problem >&2; exit 7"})

    assert result["exit_code"] == 7
    assert result["stdout"] == ""
    assert result["stderr"] == "problem"
    assert result["timed_out"] is False


def test_bash_tool_enforces_timeout() -> None:
    started_at = time.monotonic()
    result = execute_bash({"command": "sleep 2", "timeout_seconds": 0.05})
    elapsed = time.monotonic() - started_at

    assert result["exit_code"] == -1
    assert result["timed_out"] is True
    assert "timed out" in result["stderr"]
    assert result["timeout_seconds"] == 0.05
    assert result["duration_ms"] < 2_000
    assert elapsed < 2


def test_bash_timeout_kills_children_that_hold_captured_pipes_open() -> None:
    started_at = time.monotonic()
    result = execute_bash(
        {
            "command": "trap '' TERM; sh -c 'trap \"\" TERM; sleep 60' & wait",
            "timeout_seconds": 0.05,
        }
    )
    elapsed = time.monotonic() - started_at

    assert result["timed_out"] is True
    assert elapsed < 2


def test_bash_timeout_terminates_entire_process_group(tmp_path) -> None:
    marker = tmp_path / "child-terminated"
    command = (
        "sh -c 'trap \"printf terminated > "
        f"{marker}"
        "; exit 0\" TERM; while :; do sleep 1; done' & wait"
    )

    result = execute_bash({"command": command, "timeout_seconds": 0.2})

    assert result["timed_out"] is True
    assert marker.read_text() == "terminated"


def test_bash_stops_running_process_when_cancellation_monitor_fails(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.tools.registry.native import bash
    original = bash.subprocess.Popen
    processes = []
    def start(*args, **kwargs):
        process = original(*args, **kwargs)
        processes.append(process)
        return process
    checks = 0
    def monitor():
        nonlocal checks
        checks += 1
        if checks == 1:
            return False
        raise RuntimeError('lost authority monitor')
    monkeypatch.setattr(bash.subprocess, 'Popen', start)
    result = bash.execute_bash({'command': 'sleep 60', 'cwd': str(tmp_path)},
                               context=SimpleNamespace(cancellation_checker=monitor))
    assert result['cancelled'] is True
    assert result['exit_code'] == -1
    assert len(processes) == 1 and processes[0].poll() is not None


def test_bash_stops_running_process_on_unexpected_output_error(monkeypatch, tmp_path):
    from alphonse.agent_v2.core.tools.registry.native import bash
    original_start = bash.subprocess.Popen
    original_collect = bash._collect_output_until_exit
    processes = []
    def start(*args, **kwargs):
        process = original_start(*args, **kwargs)
        processes.append(process)
        return process
    first = True
    def collect(*args, **kwargs):
        nonlocal first
        if first:
            first = False
            raise OSError('synthetic output failure')
        return original_collect(*args, **kwargs)
    monkeypatch.setattr(bash.subprocess, 'Popen', start)
    monkeypatch.setattr(bash, '_collect_output_until_exit', collect)
    with pytest.raises(OSError, match='synthetic output failure'):
        bash.execute_bash({'command': 'sleep 60', 'cwd': str(tmp_path)})
    assert len(processes) == 1 and processes[0].poll() is not None


def test_bash_tool_preserves_explicit_timeout_without_product_cap() -> None:
    result = execute_bash({"command": "printf uncapped", "timeout_seconds": 999})

    assert result["stdout"] == "uncapped"
    assert result["timeout_seconds"] == 999


def test_bash_schema_does_not_impose_timeout_default_or_maximum() -> None:
    timeout_schema = BASH_ARGUMENT_SCHEMA["properties"]["timeout_seconds"]

    assert "default" not in timeout_schema
    assert "maximum" not in timeout_schema
    assert timeout_schema["exclusiveMinimum"] == 0
    assert "Omit it" in timeout_schema["description"]


def test_bash_tool_rejects_blank_command() -> None:
    with pytest.raises(ValueError, match="bash_command_required"):
        execute_bash({"command": "   "})


def test_bash_tool_truncates_large_output() -> None:
    result = execute_bash({"command": f"printf '%*s' {MAX_OUTPUT_CHARS + 10} '' | tr ' ' x"})

    assert len(result["stdout"]) < MAX_OUTPUT_CHARS + 40
    assert result["stdout"].endswith("... [truncated]")


def test_bash_drains_both_flooded_streams_with_bounded_capture() -> None:
    result = execute_bash({'command': "python3 -c 'import sys; sys.stdout.write(\"x\"*5000000); sys.stderr.write(\"y\"*5000000)'",
                           'timeout_seconds': 10})
    assert result['exit_code'] == 0
    assert result['stdout'].endswith('... [truncated]')
    assert result['stderr'].endswith('... [truncated]')
    assert len(result['stdout']) < MAX_OUTPUT_CHARS + 40
    assert len(result['stderr']) < MAX_OUTPUT_CHARS + 40


def test_bash_withholds_overflowed_stream_if_secret_crosses_cutoff() -> None:
    from alphonse.agent_v2.core.tools.registry.native.bash import MAX_CAPTURE_BYTES, _execute_bash
    from alphonse.agent_v2.skill_bindings import ExecutionBindings
    secret = 'synthetic-private-token'
    prepared = ExecutionBindings({}, (secret,))
    command = f"python3 -c 'import sys; sys.stdout.write(\"x\"*{MAX_CAPTURE_BYTES - 3}+\"{secret}\")'"
    result = _execute_bash({'command': command}, context=None, prepared=prepared)
    assert result['stdout'] == '[output withheld: exceeded capture limit]\n... [truncated]'
    assert secret not in result['stdout']


def test_bash_timeout_with_unbounded_output_stops_and_reports_truncation() -> None:
    result = execute_bash({'command': "python3 -c 'import sys;\nwhile True: sys.stdout.write(\"x\"*8192); sys.stdout.flush()'",
                           'timeout_seconds': 0.1})
    assert result['timed_out'] is True
    assert result['stdout'].endswith('... [truncated]')
    assert len(result['stdout']) < MAX_OUTPUT_CHARS + 40


def test_unverifiable_process_group_cleanup_is_not_reported_as_success(monkeypatch):
    from types import SimpleNamespace
    from alphonse.agent_v2.core.tools.registry.native import bash
    monkeypatch.setattr(bash, '_signal_process_group', lambda *_: None)
    monkeypatch.setattr(bash, '_collect_output_until_exit', lambda *_args, **_kwargs: None)
    def forbidden(_pid):
        raise PermissionError('unverifiable')
    monkeypatch.setattr(bash, '_process_group_exists', forbidden)
    with pytest.raises(PermissionError):
        bash._stop_process_group(SimpleNamespace(pid=1), bash._OutputBuffer(), bash._OutputBuffer())


def test_bash_tool_honors_optional_cwd(tmp_path) -> None:
    (tmp_path / "marker.txt").write_text("ok")

    result = execute_bash({"command": "pwd; ls marker.txt", "cwd": str(tmp_path)})

    assert result["exit_code"] == 0
    assert str(tmp_path) in result["stdout"]
    assert "marker.txt" in result["stdout"]
    assert result["cwd"] == str(tmp_path)






def test_bash_tool_definition_executes_through_registry() -> None:
    registry = build_native_tool_registry()

    result = registry.execute(BASH_TOOL_ID, {"command": "printf registry"})

    assert result["stdout"] == "registry"
    assert result["timed_out"] is False


def test_bash_tool_definition_can_be_built_directly() -> None:
    definition = build_bash_tool_definition()

    assert definition.descriptor.tool_id == BASH_TOOL_ID
    assert "For standard package scripts supply skill_id" in definition.descriptor.description
    assert "artifact_id" not in definition.argument_schema["properties"]
    assert definition.argument_schema["properties"]["command"]["type"] == "string"
