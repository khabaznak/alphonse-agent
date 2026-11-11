from pathlib import Path
from types import SimpleNamespace
import subprocess
import os
import shutil
import time

import pytest

import alphonse.agent_v2.docker_coding_sandbox as module
from alphonse.agent_v2.docker_coding_sandbox import DockerCodingSandbox
from alphonse.agent_v2.docker_coding_agents import DockerCodexSkillProvider, DockerCodexVerifier


IMAGE = 'sha256:' + 'a' * 64
JOB = 'b' * 64


class Process:
    pid = 123456
    returncode = None
    def __init__(self, *, stalled=False): self.stalled = stalled
    def poll(self): return self.returncode
    def communicate(self, input=None, timeout=None):
        if self.stalled:
            raise subprocess.TimeoutExpired('docker', timeout)
        self.returncode = 0
        return (None, None)
    def wait(self): self.returncode = -9


def commands(monkeypatch, *, missing=True, healthy=True, stalled=False):
    events = []
    def run(args, **kwargs):
        events.append(args[1:3])
        if args[1:3] == ['container', 'inspect']:
            return SimpleNamespace(returncode=1 if missing else 0, stderr='No such container' if missing else '')
        if args[1] == 'info': return SimpleNamespace(returncode=0 if healthy else 1, stderr='')
        return SimpleNamespace(returncode=0, stderr='')
    def popen(args, **kwargs):
        events.append(args[1:3])
        return Process(stalled=stalled)
    monkeypatch.setattr(module.subprocess, 'run', run)
    monkeypatch.setattr(module.subprocess, 'Popen', popen)
    monkeypatch.setattr(module.os, 'killpg', lambda *args: None)
    return events


def test_container_is_removed_and_absence_verified_before_success(tmp_path, monkeypatch):
    events = commands(monkeypatch)
    runner = DockerCodingSandbox('/usr/local/bin/docker', IMAGE)
    assert runner.run(job_key=JOB, kind='verify', workspace=tmp_path,
                      command=['/bin/true']) == 'completed'
    assert [entry[0] for entry in events] == ['create', 'start', 'rm', 'container', 'info']


@pytest.mark.parametrize('missing,healthy', [(False, True), (True, False)])
def test_unconfirmed_container_removal_never_reports_success(tmp_path, monkeypatch, missing, healthy):
    commands(monkeypatch, missing=missing, healthy=healthy)
    runner = DockerCodingSandbox('/usr/local/bin/docker', IMAGE)
    with pytest.raises(RuntimeError, match='reconciliation_required'):
        runner.run(job_key=JOB, kind='verify', workspace=tmp_path, command=['/bin/true'])


def test_cancelled_job_is_destroyed_before_client_is_killed(tmp_path, monkeypatch):
    events = commands(monkeypatch, stalled=True)
    count = 0
    def cancelled():
        nonlocal count
        count += 1
        return count >= 3
    assert DockerCodingSandbox('/usr/local/bin/docker', IMAGE).run(job_key=JOB, kind='verify',
        workspace=tmp_path, command=['/bin/true'], cancelled=cancelled) == 'cancelled'
    assert [entry[0] for entry in events] == ['create', 'start', 'rm', 'container', 'info']


def test_container_create_timeout_cannot_start_or_claim_success(tmp_path, monkeypatch):
    events = []
    def create(args, **kwargs):
        events.append(args[1])
        raise subprocess.TimeoutExpired(args, 30)
    monkeypatch.setattr(module.subprocess, 'run', create)
    with pytest.raises(RuntimeError, match='reconciliation_required'):
        DockerCodingSandbox('/usr/local/bin/docker', IMAGE).run(job_key=JOB, kind='verify',
            workspace=tmp_path, command=['/bin/true'])
    assert events == ['create']


def test_existing_container_name_is_not_blindly_removed(tmp_path, monkeypatch):
    events = []
    def create(args, **kwargs):
        events.append(args[1])
        return SimpleNamespace(returncode=1, stderr='name already in use')
    monkeypatch.setattr(module.subprocess, 'run', create)
    with pytest.raises(RuntimeError, match='reconciliation_required'):
        DockerCodingSandbox('/usr/local/bin/docker', IMAGE).run(job_key=JOB, kind='verify',
            workspace=tmp_path, command=['/bin/true'])
    assert events == ['create']


def test_adapters_keep_task_text_and_checks_in_stdin(tmp_path):
    calls = []
    class Runner:
        def run(self, **kwargs):
            calls.append(kwargs)
            return 'completed'
    workspace = tmp_path / JOB
    workspace.mkdir()
    provider = DockerCodexSkillProvider(Runner(), tmp_path / 'auth')
    assert provider.run(workspace=workspace, name='fixture', brief='private fixture brief', cancelled=lambda: False) == 'completed'
    assert b'private fixture brief' in calls[0]['stdin']
    assert 'private fixture brief' not in repr(calls[0]['command'])
    package = workspace / 'fixture'
    package.mkdir()
    verifier = DockerCodexVerifier(Runner())
    assert verifier.verify(package=package, checks=['echo fixture'], cancelled=lambda: False)
    assert b'echo fixture' in calls[1]['stdin']
    assert calls[1]['kind'] == 'verify'


@pytest.mark.skipif(not os.environ.get('ALPHONSE_TEST_CODEX_CONTAINER_IMAGE'), reason='explicit container fixture image required')
@pytest.mark.parametrize('cancel', [False, True])
def test_real_container_stops_detached_descendants_before_return(tmp_path, cancel):
    docker = shutil.which('docker')
    assert docker is not None
    runner = DockerCodingSandbox(docker, os.environ['ALPHONSE_TEST_CODEX_CONTAINER_IMAGE'])
    child = "import time; from pathlib import Path; time.sleep(2); Path('late').write_text('unexpected')"
    program = ('import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",' + repr(child)
               + '],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); '
               + ('time.sleep(30)' if cancel else 'pass'))
    start = time.monotonic()
    status = runner.run(job_key=('c' if cancel else 'd') * 64, kind='verify', workspace=tmp_path,
        command=['/usr/bin/python3', '-c', program], timeout_seconds=15,
        cancelled=(lambda: time.monotonic() - start > 1) if cancel else lambda: False)
    assert status == ('cancelled' if cancel else 'completed')
    time.sleep(2)
    assert not (tmp_path / 'late').exists()
