from pathlib import Path
import time

import pytest

from alphonse.agent_v2.coding_agent import CodexCodingAgent, IsolatedCodexSkillProvider


def executable(tmp_path, body):
    path = tmp_path / 'fake-codex'
    path.write_text('#!/usr/bin/env python3\n' + body, encoding='utf-8')
    path.chmod(0o700)
    return str(path)


def test_codex_adapter_passes_brief_on_stdin_and_uses_noninteractive_flags(tmp_path):
    binary = executable(tmp_path, "import json, sys\nfrom pathlib import Path\nPath('receipt.json').write_text(json.dumps({'args':sys.argv[1:],'brief':sys.stdin.read()}))\n")
    result = CodexCodingAgent(binary, model='fixture-model').run(
        workspace=tmp_path, brief='Build a fixture skill', timeout_seconds=5)
    import json
    receipt = json.loads((tmp_path / 'receipt.json').read_text())
    assert result.status == 'completed' and result.exit_code == 0
    assert receipt['brief'] == 'Build a fixture skill'
    assert receipt['args'] == ['exec', '--model', 'fixture-model', '--skip-git-repo-check',
                               '--ephemeral', '--ignore-user-config', '--sandbox', 'workspace-write', '-']
    assert 'Build a fixture skill' not in receipt['args']


def test_failed_process_is_not_reported_as_success(tmp_path):
    binary = executable(tmp_path, 'import sys\nsys.exit(7)\n')
    result = CodexCodingAgent(binary).run(workspace=tmp_path, brief='Make a fixture', timeout_seconds=5)
    assert result.status == 'failed' and result.exit_code == 7


def test_completed_parent_cannot_leave_a_same_group_child_writing(tmp_path):
    binary = executable(tmp_path, "import subprocess, sys\nsubprocess.Popen([sys.executable, '-c', \"import time; from pathlib import Path; time.sleep(.6); Path('late-write').write_text('unexpected')\"])\n")
    try:
        result = CodexCodingAgent(binary).run(workspace=tmp_path, brief='Make a fixture', timeout_seconds=5)
    except RuntimeError as exc:
        if str(exc) != 'coding_agent_cleanup_unconfirmed':
            raise
        pytest.skip('The outer execution sandbox denied process-group cleanup')
    assert result.status == 'completed'
    time.sleep(.8)
    assert not (tmp_path / 'late-write').exists()


def test_denied_process_cleanup_is_not_reported_as_success(monkeypatch):
    import alphonse.agent_v2.coding_agent as module
    from types import SimpleNamespace
    def deny(*args): raise PermissionError('denied')
    monkeypatch.setattr(module.os, 'killpg', deny)
    with pytest.raises(RuntimeError, match='cleanup_unconfirmed'):
        CodexCodingAgent._terminate(SimpleNamespace(pid=123))


@pytest.mark.parametrize('mode', ['cancelled', 'timed_out'])
def test_cancel_and_timeout_stop_process(tmp_path, mode):
    binary = executable(tmp_path, "import time\nfrom pathlib import Path\ntime.sleep(2)\nPath('late-write').write_text('unexpected')\n")
    started = time.monotonic()
    result = CodexCodingAgent(binary).run(workspace=tmp_path, brief='Make a fixture', timeout_seconds=.3 if mode == 'timed_out' else 5,
        cancelled=(lambda: time.monotonic() - started > .3) if mode == 'cancelled' else None)
    assert result.status == mode
    time.sleep(.1)
    assert not (tmp_path / 'late-write').exists()


def test_cancel_before_start_and_reject_invalid_inputs(tmp_path):
    agent = CodexCodingAgent(executable(tmp_path, 'raise RuntimeError()\n'))
    assert agent.run(workspace=tmp_path, brief='x', timeout_seconds=1, cancelled=lambda: True).status == 'cancelled'
    with pytest.raises(ValueError, match='brief_invalid'):
        agent.run(workspace=tmp_path, brief='x' * 9000, timeout_seconds=1)
    with pytest.raises(ValueError, match='absolute_executable'):
        CodexCodingAgent('codex')
    with pytest.raises(ValueError, match='workspace_invalid'):
        agent.run(workspace=tmp_path / 'missing', brief='x', timeout_seconds=1)


def test_isolated_provider_probes_before_running_and_cleans_auth(tmp_path, monkeypatch):
    import sys
    import alphonse.agent_v2.coding_agent as module
    monkeypatch.setattr(sys, 'platform', 'darwin')
    class Listener:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address): pass
        def listen(self, count): pass
        def getsockname(self): return ('127.0.0.1', 12345)
    monkeypatch.setattr(module.socket, 'socket', Listener)
    workspace = tmp_path / 'work'
    workspace.mkdir()
    auth = tmp_path / 'source-auth.json'
    auth.write_text('{"fixture": true}')
    binary = executable(tmp_path, "import os, pathlib, sys\n"
        "if sys.argv[1] == 'sandbox': pathlib.Path(sys.argv[-1]).write_text('ok')\n"
        "else:\n"
        " p=pathlib.Path(os.environ['CODEX_HOME']); assert (p/'auth.json').exists(); "
        "pathlib.Path('used-auth-home').write_text(str(p)); sys.stdin.read()\n")
    provider = IsolatedCodexSkillProvider(binary, auth)
    assert provider.run(workspace=workspace, name='fixture-reader', brief='A fixture', cancelled=lambda: False) == 'completed'
    isolated_home = Path((workspace / 'used-auth-home').read_text())
    assert not isolated_home.exists()
    assert auth.exists()


def test_isolated_provider_fails_closed_when_sandbox_probe_fails(tmp_path, monkeypatch):
    import sys
    import alphonse.agent_v2.coding_agent as module
    monkeypatch.setattr(sys, 'platform', 'darwin')
    class Listener:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def bind(self, address): pass
        def listen(self, count): pass
        def getsockname(self): return ('127.0.0.1', 12345)
    monkeypatch.setattr(module.socket, 'socket', Listener)
    workspace = tmp_path / 'work'
    workspace.mkdir()
    auth = tmp_path / 'source-auth.json'
    auth.write_text('{}')
    binary = executable(tmp_path, 'import sys\nsys.exit(9)\n')
    with pytest.raises(RuntimeError, match='isolation_probe_failed'):
        IsolatedCodexSkillProvider(binary, auth).run(workspace=workspace, name='fixture-reader', brief='x', cancelled=lambda: False)
    assert not list(tmp_path.glob('.codex-isolated-*'))
