"""Disposable Docker-backed Bash for Dojo fixtures, not household execution.

The caller must authorize the synthetic task and inspect effects independently.
Containers reduce exposure but are not a complete security boundary.
"""
from __future__ import annotations

import math
import os
import re
import selectors
import shutil
import signal
import subprocess
import time
from uuid import uuid4
from typing import Callable


class DojoBashUnavailable(RuntimeError):
    """The disposable execution environment could not be established."""


class DojoBash:
    """One disposable workspace shared across commands, destroyed on close.

    The fixed container is unprivileged, has no host mounts or network and a
    bounded tmpfs workspace. The image ID and local Docker socket are trusted
    operator inputs. Abrupt process death can leave a stopped/running container;
    its sleep process exits after 300 seconds, but explicit cleanup is required.
    """
    def __init__(self, *, image_id: str, docker_socket: str, timeout_seconds: float = 15,
                 output_limit: int = 16384):
        if not isinstance(image_id, str) or not re.fullmatch(r'sha256:[0-9a-f]{64}', image_id):
            raise ValueError('dojo_bash_image_id_required')
        if not isinstance(docker_socket, str) or not docker_socket.startswith('/'):
            raise ValueError('dojo_bash_local_socket_required')
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (float, int))
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60):
            raise ValueError('dojo_bash_timeout_invalid')
        if type(output_limit) is not int or not 0 < output_limit <= 65536:
            raise ValueError('dojo_bash_output_limit_invalid')
        binary = shutil.which('docker')
        if not binary:
            raise DojoBashUnavailable('docker_unavailable')
        self.image_id = image_id
        self.timeout_seconds = float(timeout_seconds)
        self.output_limit = output_limit
        self.name = 'alphonse-dojo-' + uuid4().hex
        self._base = [binary, '--host', 'unix://' + docker_socket]
        self._active = False
        self._created = False
        self.cleanup_verified = None

    def _control(self, args: list[str]) -> subprocess.CompletedProcess:
        # Never include caller environment variables, Docker context or credentials.
        env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/nonexistent'}
        try:
            return subprocess.run(self._base + args, env=env, stdin=subprocess.DEVNULL,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15, check=False)
        except (OSError, subprocess.TimeoutExpired):
            raise DojoBashUnavailable('docker_control_unavailable') from None

    def __enter__(self):
        inspected = self._control(['image', 'inspect', '--format', '{{.Id}}', self.image_id])
        if inspected.returncode != 0 or inspected.stdout.decode(errors='replace').strip() != self.image_id:
            raise DojoBashUnavailable('local_image_unavailable')
        create = ['create', '--pull=never', '--name', self.name, '--network', 'none', '--read-only',
                  '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--pids-limit', '32',
                  '--memory', '128m', '--cpus', '1', '--user', '65534:65534',
                  '--tmpfs', '/tmp:rw,nosuid,nodev,size=16m,mode=1777',
                  '--tmpfs', '/workspace:rw,nosuid,nodev,size=16m,mode=1777',
                  '--workdir', '/workspace', '--env', 'HOME=/workspace',
                  '--env', 'PATH=/usr/local/bin:/usr/bin:/bin', '--entrypoint', '/bin/sleep',
                  self.image_id, '300']
        try:
            # A timed-out create can still have succeeded on the daemon.
            self._created = True
            result = self._control(create)
            if result.returncode != 0:
                raise DojoBashUnavailable('container_create_failed')
            if self._control(['start', self.name]).returncode != 0:
                raise DojoBashUnavailable('container_start_failed')
            self._active = True
            return self
        except Exception:
            self.close()
            raise

    def execute(self, command: str, *, cancelled: Callable[[], bool] | None = None) -> dict:
        if not self._active:
            raise DojoBashUnavailable('container_not_active')
        if not isinstance(command, str) or not command.strip() or len(command.encode()) > 16384:
            raise ValueError('dojo_bash_command_invalid')
        if cancelled is not None and not callable(cancelled):
            raise ValueError('dojo_bash_cancel_invalid')
        def is_cancelled():
            try:
                return cancelled is not None and cancelled() is not False
            except Exception:
                return True
        if is_cancelled():
            return {'status': 'cancelled', 'exit_code': None, 'stdout': '', 'stderr': '',
                    'truncated': False, 'cleanup_verified': self.cleanup_verified, 'duration_ms': 0}
        started = time.monotonic()
        env = {'PATH': '/usr/local/bin:/usr/bin:/bin', 'HOME': '/nonexistent'}
        args = self._base + ['exec', self.name, '/bin/bash', '--noprofile', '--norc', '-c', command]
        try:
            process = subprocess.Popen(args, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, start_new_session=True)
        except OSError:
            raise DojoBashUnavailable('docker_exec_unavailable') from None
        buffers = {'stdout': bytearray(), 'stderr': bytearray()}
        truncated = False
        outcome = None
        deadline = time.monotonic() + self.timeout_seconds
        with selectors.DefaultSelector() as selector:
            for stream, label in ((process.stdout, 'stdout'), (process.stderr, 'stderr')):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, label)
            while selector.get_map() or process.poll() is None:
                if is_cancelled():
                    outcome = 'cancelled'
                    break
                if time.monotonic() >= deadline:
                    outcome = 'timeout'
                    break
                for key, _ in selector.select(.05):
                    chunk = os.read(key.fd, 8192)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
                        continue
                    buf = buffers[key.data]
                    available = max(0, self.output_limit - len(buf))
                    buf.extend(chunk[:available])
                    truncated |= len(chunk) > available
            if outcome is not None:
                # Stopping the Docker CLI alone does not stop the container exec.
                self.close()
                try:
                    # Container removal normally causes the client to exit.
                    # Reap it first to avoid signaling an already-exited group.
                    process.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        # A group that may still be live is not a confirmed
                        # cleanup. Do not disguise it as clean cancellation.
                        if process.poll() is None:
                            raise DojoBashUnavailable('docker_client_termination_unconfirmed') from None
                    process.wait(timeout=5)
            else:
                process.wait(timeout=5)
        return {'status': outcome or ('completed' if process.returncode == 0 else 'failed'),
                'exit_code': None if outcome else process.returncode,
                'stdout': bytes(buffers['stdout']).decode('utf-8', errors='replace'),
                'stderr': bytes(buffers['stderr']).decode('utf-8', errors='replace'),
                'truncated': truncated, 'cleanup_verified': self.cleanup_verified,
                'duration_ms': int((time.monotonic() - started) * 1000)}

    def close(self) -> bool:
        self._active = False
        if not self._created:
            return self.cleanup_verified is True
        try:
            self._control(['rm', '-f', self.name])
            result = self._control(['container', 'ls', '-aq', '--filter', 'name=^/' + self.name + '$'])
            self.cleanup_verified = result.returncode == 0 and not result.stdout.strip()
        except DojoBashUnavailable:
            self.cleanup_verified = False
        self._created = not self.cleanup_verified
        return self.cleanup_verified

    def __exit__(self, *_):
        self.close()
