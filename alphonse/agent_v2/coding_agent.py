"""Codex process adapter for skill-building jobs.

This module does not publish packages or grant access to household resources.
Codex's workspace-write sandbox is not a confidentiality boundary: the model and
its subprocesses can still read host files available to the operating-system user.
The caller must supply a separately isolated execution environment before using
this adapter for household work.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import shutil
import tempfile
import time
from typing import Callable


MAX_BRIEF_BYTES = 8192
TERMINATION_GRACE_SECONDS = 1.0


@dataclass(frozen=True)
class CodingAgentResult:
    status: str
    exit_code: int | None = None


class CodexCodingAgent:
    """Run one noninteractive coding job inside a caller-provided environment.

    No shell is used, no user configuration is loaded, and no model output is
    treated as a verified result. The caller owns package checks and publication.
    """

    def __init__(self, executable: str, *, model: str = "", config: tuple[str, ...] = ()) -> None:
        if not executable or not Path(executable).is_absolute():
            raise ValueError("coding_agent_absolute_executable_required")
        self.executable = executable
        self.model = model
        self.config = config

    def run(
        self, *, workspace: Path, brief: str, timeout_seconds: float,
        cancelled: Callable[[], bool] | None = None,
        environment: dict[str, str] | None = None,
    ) -> CodingAgentResult:
        workspace = Path(workspace)
        if workspace.is_symlink() or not workspace.is_dir():
            raise ValueError("coding_agent_workspace_invalid")
        if not isinstance(brief, str) or not brief.strip() or len(brief.encode("utf-8")) > MAX_BRIEF_BYTES:
            raise ValueError("coding_agent_brief_invalid")
        if not 0 < timeout_seconds <= 3600:
            raise ValueError("coding_agent_timeout_invalid")
        if cancelled and cancelled():
            return CodingAgentResult("cancelled")
        command = [self.executable, "exec", "--skip-git-repo-check", "--ephemeral",
                   "--ignore-user-config"]
        for entry in self.config:
            command.extend(["-c", entry])
        if not self.config:
            command.extend(["--sandbox", "workspace-write"])
        command.append("-")
        if self.model:
            command[2:2] = ["--model", self.model]
        try:
            process = subprocess.Popen(
                command, cwd=workspace, env=environment, stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError("coding_agent_start_failed") from exc
        try:
            # communicate drains stdin while periodically checking the process.
            # Keep the short brief in memory; never place it in command arguments.
            deadline = time.monotonic() + timeout_seconds
            data = brief.encode("utf-8")
            while True:
                if cancelled and cancelled():
                    self._terminate(process)
                    return CodingAgentResult("cancelled", process.returncode)
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._terminate(process)
                    return CodingAgentResult("timed_out", process.returncode)
                try:
                    process.communicate(input=data, timeout=min(0.2, remaining))
                except subprocess.TimeoutExpired:
                    data = None
                    continue
                break
            # A successful parent may leave processes in its session; stop them
            # before a caller inspects the output or considers publication.
            self._terminate(process)
            if cancelled and cancelled():
                return CodingAgentResult("cancelled", process.returncode)
            return CodingAgentResult("completed" if process.returncode == 0 else "failed", process.returncode)
        except BaseException:
            if process.poll() is None:
                self._terminate(process)
            raise

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        # Do this even if the group leader exited: descendants may still exist.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeError('coding_agent_cleanup_unconfirmed') from exc
        try:
            process.wait(timeout=TERMINATION_GRACE_SECONDS)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            raise RuntimeError('coding_agent_cleanup_unconfirmed') from exc
        process.wait()


ISOLATED_CONFIG = (
    'default_permissions="skill_builder"',
    'permissions.skill_builder.filesystem={":minimal"="read",":workspace_roots"="write","/private/tmp"="deny"}',
    'permissions.skill_builder.network.enabled=false',
    'shell_environment_policy.inherit="none"',
    'shell_environment_policy.set.PYTHONDONTWRITEBYTECODE="1"',
    'approval_policy="never"',
    'web_search="disabled"',
    'features.apps=false',
    'skills.include_instructions=false',
    'skills.bundled.enabled=false',
)


def _runtime_permissions(roots: tuple[str, ...]) -> tuple[str, ...]:
    values = []
    for value in roots:
        path = Path(value)
        if not path.is_absolute() or '..' in path.parts or not path.exists():
            raise ValueError('coding_agent_runtime_root_invalid')
        # Only explicitly selected system-installed runtime trees are eligible;
        # arbitrary household paths are never passed to the model sandbox.
        resolved = path.resolve()
        if str(path) != '/opt/homebrew/opt' and not resolved.is_relative_to('/opt/homebrew/Cellar'):
            raise ValueError('coding_agent_runtime_root_invalid')
        values.append(json.dumps(str(path)) + '="read"')
    return tuple(values)


def _isolated_config(roots: tuple[str, ...]) -> tuple[str, ...]:
    entries = [entry for entry in ISOLATED_CONFIG if not entry.startswith('permissions.skill_builder.filesystem=')]
    filesystem = [ '":minimal"="read"', '":workspace_roots"="write"', '"/private/tmp"="deny"',
                   *_runtime_permissions(roots)]
    entries.append('permissions.skill_builder.filesystem={' + ','.join(filesystem) + '}')
    runtime_bins = [str(Path(root) / 'bin') for root in roots if Path(root).resolve().is_relative_to('/opt/homebrew/Cellar')]
    entries.append('shell_environment_policy.set.PATH=' + json.dumps(':'.join([*runtime_bins, '/usr/bin', '/bin'])))
    return tuple(entries)


def _minimal_host_environment() -> dict[str, str]:
    """Do not pass unrelated account credentials to the Codex host process."""
    return {key: os.environ[key] for key in ('PATH', 'HOME', 'LANG', 'LC_ALL', 'TMPDIR', 'USER', 'LOGNAME', 'SHELL')
            if key in os.environ}


class IsolatedCodexSkillProvider:
    """macOS Codex skill-builder with a runtime-checked command sandbox.

    This does not provide a verification environment or install skills. The
    configured Codex CLI must support named permission profiles. Shell network is
    disabled; external facts must be supplied in the authorized brief. The probe
    is necessary, but a disposable live qualification is still required before
    enabling this in a household runtime. Process-group cleanup does not stop
    children that detach into another session, so this is not production-ready.
    """

    def __init__(self, executable: str, auth_file: Path, *, model: str = "", timeout_seconds: float = 600,
                 runtime_roots: tuple[str, ...] = ()):
        self.executable = executable
        self.auth_file = Path(auth_file)
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.config = _isolated_config(runtime_roots)

    def run(self, *, workspace: Path, name: str, brief: str, cancelled: Callable[[], bool]) -> str:
        import sys
        if sys.platform != 'darwin':
            raise RuntimeError('coding_agent_isolation_unsupported_platform')
        if cancelled():
            return 'cancelled'
        if self.auth_file.is_symlink() or not self.auth_file.is_file():
            raise RuntimeError('coding_agent_auth_unavailable')
        with tempfile.TemporaryDirectory(prefix='.codex-isolated-', dir=workspace.parent) as private_dir:
            private = Path(private_dir)
            # The copied credential is outside the workspace, private to this job,
            # and destroyed when the call exits; a host crash needs cleanup.
            auth = private / 'auth.json'
            shutil.copyfile(self.auth_file, auth)
            auth.chmod(0o600)
            fixture = private / 'probe.txt'
            fixture.write_text('fixture', encoding='utf-8')
            env = _minimal_host_environment()
            env['CODEX_HOME'] = str(private)
            env['ALPHONSE_SANDBOX_PROBE_SECRET'] = 'fixture'
            probe_config = [entry for entry in self.config if not entry.startswith('default_permissions=')]
            command = [self.executable, 'sandbox']
            for entry in probe_config:
                command.extend(['-c', entry])
            try:
                with socket.socket() as listener:
                    listener.bind(('127.0.0.1', 0))
                    listener.listen(1)
                    port = str(listener.getsockname()[1])
                    command.extend(['-P', 'skill_builder', '-C', str(workspace), '--', '/bin/sh', '-c',
                                    'if /bin/cat "$1" >/dev/null 2>&1; then exit 11; fi; '
                                    '[ -z "${ALPHONSE_SANDBOX_PROBE_SECRET+x}" ] || exit 12; '
                                    '/usr/bin/nc -h 2>&1 | /usr/bin/grep -qi usage || exit 15; '
                                    'if /usr/bin/nc -z -w 1 127.0.0.1 "$2" >/dev/null 2>&1; then exit 14; fi; '
                                    'printf ok > "$3" || exit 13', 'sh', str(fixture), port, str(workspace / '.sandbox-probe')])
                    probe = subprocess.run(command, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15, check=False)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise RuntimeError('coding_agent_isolation_probe_failed') from exc
            marker = workspace / '.sandbox-probe'
            if probe.returncode != 0 or not marker.is_file() or marker.read_text() != 'ok':
                raise RuntimeError('coding_agent_isolation_probe_failed')
            marker.unlink()
            if cancelled():
                return 'cancelled'
            prompt = (f'Create a standard Agent Skills package in ./{name}/ with SKILL.md and any necessary '
                      'scripts or references. Keep personal data, credentials and environment files out of the package. '
                      'Include reproducible behavior checks using synthetic data. Do not modify files outside the package.\n\n'
                      'Task specification:\n' + brief)
            result = CodexCodingAgent(self.executable, model=self.model, config=self.config).run(
                workspace=workspace, brief=prompt, timeout_seconds=self.timeout_seconds,
                cancelled=cancelled, environment=env)
            return result.status


class CodexSandboxVerifier:
    """Run explicit behavior checks in a separate no-network Codex sandbox."""

    def __init__(self, executable: str, *, timeout_seconds: float = 30, runtime_roots: tuple[str, ...] = ()):
        if not 0 < timeout_seconds <= 120:
            raise ValueError('coding_skill_check_timeout_invalid')
        self.executable, self.timeout_seconds = executable, timeout_seconds
        self.config = _isolated_config(runtime_roots)

    def verify(self, *, package: Path, checks: list[str], cancelled: Callable[[], bool]) -> bool:
        import sys
        if not checks or len(checks) > 20 or any(not isinstance(c, str) or not c.strip() or len(c) > 4096 for c in checks):
            raise ValueError('coding_skill_checks_required')
        if sys.platform != 'darwin':
            raise RuntimeError('coding_agent_isolation_unsupported_platform')
        if cancelled():
            return False
        with tempfile.TemporaryDirectory(prefix='.codex-verify-', dir=package.parent.parent) as private_dir:
            private = Path(private_dir)
            env = _minimal_host_environment()
            env['CODEX_HOME'] = str(private)
            env['ALPHONSE_SANDBOX_PROBE_SECRET'] = 'fixture'
            fixture = private / 'probe.txt'
            fixture.write_text('fixture', encoding='utf-8')
            marker = package / '.sandbox-probe'
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                listener.listen(1)
                port = str(listener.getsockname()[1])
                command = self._command(package, '/bin/sh', '-c',
                    'if /bin/cat "$1" >/dev/null 2>&1; then exit 11; fi; '
                    '[ -z "${ALPHONSE_SANDBOX_PROBE_SECRET+x}" ] || exit 12; '
                    '/usr/bin/nc -h 2>&1 | /usr/bin/grep -qi usage || exit 15; '
                    'if /usr/bin/nc -z -w 1 127.0.0.1 "$2" >/dev/null 2>&1; then exit 14; fi; '
                    'printf ok > "$3" || exit 13', 'sh', str(fixture), port, str(marker))
                if not self._run(command, env, cancelled, 15) or not marker.is_file() or marker.read_text() != 'ok':
                    raise RuntimeError('coding_agent_isolation_probe_failed')
            marker.unlink()
            for check in checks:
                if cancelled() or not self._run(self._command(package, '/bin/sh', '-c', check), env, cancelled, self.timeout_seconds):
                    return False
            return not cancelled()

    def _command(self, package, *args):
        command = [self.executable, 'sandbox']
        for entry in self.config:
            if not entry.startswith('default_permissions='):
                command.extend(['-c', entry])
        return [*command, '-P', 'skill_builder', '-C', str(package), '--', *args]

    @staticmethod
    def _run(command, env, cancelled, timeout):
        try:
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        except OSError as exc:
            raise RuntimeError('coding_agent_start_failed') from exc
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if cancelled() or time.monotonic() >= deadline:
                    CodexCodingAgent._terminate(process)
                    return False
                time.sleep(.05)
            code = process.returncode
            CodexCodingAgent._terminate(process)
            return code == 0 and not cancelled()
        except BaseException:
            CodexCodingAgent._terminate(process)
            raise
