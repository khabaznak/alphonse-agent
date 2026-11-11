"""Experimental container lifecycle for coding jobs; not enabled by default.

The caller supplies a verified image and an inner Codex sandbox policy. A container
exit, including forced removal, terminates detached children before publication.
Loss of Docker control is reported as uncertain, never as successful cleanup.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import signal
import subprocess
import time
from typing import Callable


class DockerCodingSandbox:
    def __init__(self, docker: str, image_id: str, *, seccomp: Path | None = None):
        if not Path(docker).is_absolute() or not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
            raise ValueError('coding_container_configuration_invalid')
        self.docker = docker
        self.image_id = image_id
        self.seccomp = seccomp or Path(__file__).parent / 'coding_sandbox' / 'seccomp.json'

    @staticmethod
    def _name(job_key: str, kind: str) -> str:
        if not re.fullmatch(r'[0-9a-f]{64}', job_key) or kind not in {'build', 'verify'}:
            raise ValueError('coding_container_identity_invalid')
        return f'alphonse-coding-{job_key}-{kind}'

    def run(self, *, job_key: str, kind: str, workspace: Path, command: list[str],
            stdin: bytes = b'', auth_file: Path | None = None,
            cancelled: Callable[[], bool] = lambda: False, timeout_seconds: float = 600) -> str:
        name = self._name(job_key, kind)
        workspace = Path(workspace)
        if (workspace.is_symlink() or not workspace.is_dir() or not workspace.is_absolute()
                or ',' in str(workspace) or os.getuid() == 0 or not 0 < timeout_seconds <= 3600):
            raise ValueError('coding_container_configuration_invalid')
        if not command or not all(isinstance(value, str) and value for value in command):
            raise ValueError('coding_container_command_invalid')
        if auth_file is not None:
            auth_file = Path(auth_file)
            if kind != 'build' or auth_file.is_symlink() or not auth_file.is_file() or ',' in str(auth_file):
                raise ValueError('coding_container_auth_invalid')
        if cancelled():
            return 'cancelled'
        args = [self.docker, 'create', '-i', '--name', name, '--pull=never',
                '--user', f'{os.getuid()}:{os.getgid()}', '--cap-drop=ALL',
                '--security-opt=no-new-privileges', f'--security-opt=seccomp={self.seccomp}',
                '--pids-limit=128', '--memory=1g', '--read-only',
                '--network=bridge' if kind == 'build' else '--network=none',
                '--tmpfs', f'/tmp:rw,nosuid,nodev,size=64m,uid={os.getuid()},gid={os.getgid()}',
                '--tmpfs', f'/codex-home:rw,nosuid,nodev,size=64m,uid={os.getuid()},gid={os.getgid()}',
                '--workdir', '/workspace', '--mount', f'type=bind,src={workspace},dst=/workspace',
                '-e', 'HOME=/codex-home', '-e', 'CODEX_HOME=/codex-home']
        if auth_file is not None:
            args.extend(['--mount', f'type=bind,src={auth_file},dst=/run/codex-auth.json,readonly'])
        args.extend([self.image_id, *command])
        # Reserve a named, stopped container before starting it. A timed-out
        # create cannot later begin executing; a duplicate name is never reused.
        try:
            created = subprocess.run(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RuntimeError('coding_container_reconciliation_required') from exc
        if created.returncode != 0:
            # A name collision may be an earlier unfinished container. Do not
            # remove it blindly or turn uncertain prior work into a new run.
            raise RuntimeError('coding_container_reconciliation_required')
        if cancelled():
            if not self.destroy(job_key=job_key, kind=kind):
                raise RuntimeError('coding_container_reconciliation_required')
            return 'cancelled'
        try:
            process = subprocess.Popen([self.docker, 'start', '-a', '-i', name], stdin=subprocess.PIPE,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            if not self.destroy(job_key=job_key, kind=kind):
                raise RuntimeError('coding_container_reconciliation_required') from exc
            raise RuntimeError('coding_container_start_failed') from exc
        status = 'failed'
        try:
            deadline = time.monotonic() + timeout_seconds
            data = stdin
            while True:
                if cancelled():
                    status = 'cancelled'
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    status = 'timed_out'
                    break
                try:
                    process.communicate(input=data, timeout=min(.2, remaining))
                except subprocess.TimeoutExpired:
                    data = None
                    continue
                status = 'completed' if process.returncode == 0 else 'failed'
                break
        finally:
            # Destroy the container before signaling the docker client: orphaned
            # sessions inside the container are not covered by a host killpg.
            removed = self.destroy(job_key=job_key, kind=kind)
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            if not removed:
                raise RuntimeError('coding_container_reconciliation_required')
        return status

    def destroy(self, *, job_key: str, kind: str) -> bool:
        """Remove the exact job container and independently confirm its absence."""
        name = self._name(job_key, kind)
        try:
            subprocess.run([self.docker, 'rm', '-f', name], capture_output=True, timeout=15, check=False)
            inspect = subprocess.run([self.docker, 'container', 'inspect', name],
                                     capture_output=True, text=True, timeout=15, check=False)
            health = subprocess.run([self.docker, 'info', '--format', '{{.ServerVersion}}'],
                                    capture_output=True, timeout=15, check=False)
        except (OSError, subprocess.TimeoutExpired):
            return False
        return inspect.returncode != 0 and 'No such container' in inspect.stderr and health.returncode == 0
