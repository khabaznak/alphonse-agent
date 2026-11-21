"""Dojo executor controls; live container checks are separately opt-in."""
import subprocess
import sys
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation.dojo_bash import DojoBash, DojoBashUnavailable

IMAGE = 'sha256:' + 'a' * 64


def fixture_shell(monkeypatch, *, output_limit=16384, timeout_seconds=1):
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_bash.shutil.which', lambda _: '/usr/bin/docker')
    shell = DojoBash(image_id=IMAGE, docker_socket='/tmp/fixture.sock',
                     output_limit=output_limit, timeout_seconds=timeout_seconds)
    commands = []
    def control(args):
        commands.append(args)
        output = (IMAGE + '\n').encode() if args[:2] == ['image', 'inspect'] else b''
        return SimpleNamespace(returncode=0, stdout=output, stderr=b'')
    monkeypatch.setattr(shell, '_control', control)
    return shell, commands


def test_container_settings_and_cleanup(monkeypatch):
    shell, commands = fixture_shell(monkeypatch)
    with shell:
        create = next(command for command in commands if command[0] == 'create')
        assert '--pull=never' in create
        assert create[create.index('--network') + 1] == 'none'
        assert '--read-only' in create and '--cap-drop' in create
        assert '--user' in create and '--tmpfs' in create
        assert '-v' not in create and '--mount' not in create
    assert shell.cleanup_verified is True
    assert any(command[:2] == ['rm', '-f'] for command in commands)


def test_failed_preflight_never_creates_container(monkeypatch):
    shell, commands = fixture_shell(monkeypatch)
    monkeypatch.setattr(shell, '_control', lambda args: SimpleNamespace(returncode=1, stdout=b'', stderr=b''))
    with pytest.raises(DojoBashUnavailable, match='local_image_unavailable'):
        shell.__enter__()
    assert not commands


def test_output_is_bounded_while_command_runs(monkeypatch):
    shell, _ = fixture_shell(monkeypatch, output_limit=32)
    original = subprocess.Popen
    def fake_exec(_args, **kwargs):
        return original([sys.executable, '-c', 'import sys;sys.stdout.write("x"*10000)'], **kwargs)
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_bash.subprocess.Popen', fake_exec)
    with shell:
        result = shell.execute('example')
        assert result['status'] == 'completed' and result['exit_code'] == 0
        assert result['stdout'] == 'x' * 32 and result['truncated'] is True


def test_timeout_destroys_container_and_disallows_reuse(monkeypatch):
    shell, commands = fixture_shell(monkeypatch, timeout_seconds=.15)
    original = subprocess.Popen
    def fake_exec(_args, **kwargs):
        return original([sys.executable, '-c', 'import time;time.sleep(10)'], **kwargs)
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_bash.subprocess.Popen', fake_exec)
    with shell:
        result = shell.execute('example')
        assert result['status'] == 'timeout' and result['exit_code'] is None
        assert result['cleanup_verified'] is True
        with pytest.raises(DojoBashUnavailable, match='container_not_active'):
            shell.execute('again')
    assert any(command[:2] == ['rm', '-f'] for command in commands)


def test_cancel_reaps_client_that_exits_during_container_cleanup(monkeypatch):
    shell, _ = fixture_shell(monkeypatch)
    original_popen = subprocess.Popen
    child = {}
    def fake_exec(_args, **kwargs):
        child['process'] = original_popen([sys.executable, '-c', 'import time;time.sleep(10)'], **kwargs)
        return child['process']
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_bash.subprocess.Popen', fake_exec)
    close = shell.close
    def stop_container():
        process = child.get('process')
        if process is not None and process.poll() is None:
            process.terminate()
        return close()
    monkeypatch.setattr(shell, 'close', stop_container)
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_bash.os.killpg',
                        lambda *_: pytest.fail('already-exited client group was signaled'))
    calls = iter([False, True])
    with shell:
        result = shell.execute('example', cancelled=lambda: next(calls))
    assert result['status'] == 'cancelled'
    assert result['cleanup_verified'] is True


def test_cancel_reports_uncertain_if_live_client_cannot_be_signaled(monkeypatch):
    shell, _ = fixture_shell(monkeypatch)
    original_popen = subprocess.Popen
    child = {}
    def fake_exec(_args, **kwargs):
        child['process'] = original_popen([sys.executable, '-c', 'import time;time.sleep(10)'], **kwargs)
        return child['process']
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_bash.subprocess.Popen', fake_exec)
    def denied(*_args): raise PermissionError('fixture')
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_bash.os.killpg', denied)
    calls = iter([False, True])
    try:
        with shell:
            with pytest.raises(DojoBashUnavailable, match='docker_client_termination_unconfirmed'):
                shell.execute('example', cancelled=lambda: next(calls))
    finally:
        if child.get('process') is not None and child['process'].poll() is None:
            child['process'].kill()
            child['process'].wait(timeout=5)


def test_precancelled_command_is_not_dispatched(monkeypatch):
    shell, _ = fixture_shell(monkeypatch)
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_bash.subprocess.Popen',
                        lambda *_a, **_k: pytest.fail('cancelled command was dispatched'))
    with shell:
        result = shell.execute('example', cancelled=lambda: True)
        assert result['status'] == 'cancelled' and result['exit_code'] is None


def test_cleanup_failure_is_reported(monkeypatch):
    shell, _ = fixture_shell(monkeypatch)
    with shell:
        monkeypatch.setattr(shell, '_control', lambda *_: (_ for _ in ()).throw(DojoBashUnavailable()))
    assert shell.cleanup_verified is False


@pytest.mark.parametrize('image', ['python:latest', 'sha256:' + 'a' * 63, 'sha256:' + 'G' * 64])
def test_requires_immutable_image(monkeypatch, image):
    monkeypatch.setattr('alphonse.agent_v2.evaluation.dojo_bash.shutil.which', lambda _: '/usr/bin/docker')
    with pytest.raises(ValueError, match='image_id_required'):
        DojoBash(image_id=image, docker_socket='/tmp/fixture.sock')
