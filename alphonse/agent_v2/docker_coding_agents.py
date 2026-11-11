"""Experimental Codex and verification adapters backed by job containers."""
from __future__ import annotations

from pathlib import Path
import shlex

from alphonse.agent_v2.docker_coding_sandbox import DockerCodingSandbox


_CONFIG = (
    'permissions.skill_builder.filesystem={":minimal"="read",":workspace_roots"="write",'
    '"/tmp"="deny","/run"="deny","/codex-home"="read",'
    '"/codex-home/auth.json"="deny","/usr/local"="read"}',
    'permissions.skill_builder.network.enabled=false',
    'shell_environment_policy.inherit="none"',
    'shell_environment_policy.set.PATH="/usr/local/bin:/usr/bin:/bin"',
    'shell_environment_policy.set.PYTHONDONTWRITEBYTECODE="1"',
    'approval_policy="never"', 'web_search="disabled"', 'features.apps=false',
    'skills.include_instructions=false', 'skills.bundled.enabled=false',
)


def _flags() -> list[str]:
    return [item for config in _CONFIG for item in ('-c', config)]


def _probe() -> str:
    # This process runs outside the inner sandbox but inside the job container.
    # It starts a local listener, then proves the sandboxed process cannot reach
    # it or read either copy of the temporary credential.
    sandbox = shlex.join(['codex', 'sandbox', *_flags(), '-P', 'skill_builder', '-C', '/workspace', '--',
        '/bin/sh', '-c',
        'for f in /run/codex-auth.json /codex-home/auth.json '
        '/proc/1/root/run/codex-auth.json /proc/1/root/codex-home/auth.json; do '
        'if /bin/cat "$f" >/dev/null 2>&1; then exit 11; fi; done; '
        '[ -z "${ALPHONSE_SANDBOX_PROBE_SECRET+x}" ] || exit 12; '
        'if /usr/bin/nc -z -w 1 127.0.0.1 39027 >/dev/null 2>&1; then exit 13; fi; '
        'printf ok > /workspace/.sandbox-probe'])
    return ('export ALPHONSE_SANDBOX_PROBE_SECRET=fixture; '
            '/usr/bin/nc -l -k -p 39027 >/dev/null 2>&1 & listener=$!; '
            '/bin/sleep 0.2; kill -0 "$listener" || exit 14; '
            '/usr/bin/nc -z -w 1 127.0.0.1 39027 >/dev/null 2>&1 || exit 15; '
            + sandbox + '; result=$?; kill "$listener" 2>/dev/null || :; '
            '[ "$result" -eq 0 ] && [ "$(/bin/cat /workspace/.sandbox-probe 2>/dev/null)" = ok ] '
            '|| exit 16; /bin/rm /workspace/.sandbox-probe || exit 17; ')


class DockerCodexSkillProvider:
    def __init__(self, runner: DockerCodingSandbox, auth_file: Path, *, model: str = '', timeout_seconds: float = 600):
        self.runner, self.auth_file, self.model, self.timeout_seconds = runner, auth_file, model, timeout_seconds

    def run(self, *, workspace: Path, name: str, brief: str, cancelled) -> str:
        # Codex protects these paths in the workspace. On Docker Desktop's
        # bind mounts they must exist before its sandbox creates read-only masks.
        for protected in ('.agents', '.codex'):
            (workspace / protected).mkdir(mode=0o700)
        prompt = (f'Create a standard Agent Skills package in ./{name}/; its directory and frontmatter name '
                  f'must both be {name}. Include SKILL.md, necessary scripts and synthetic behavior tests. '
                  'Do not include household records or credentials in the package.\n\nTask specification:\n' + brief)
        args = ['codex', 'exec', '--skip-git-repo-check', '--ephemeral', '--ignore-user-config',
                '-c', 'default_permissions="skill_builder"', *_flags()]
        if self.model:
            args.extend(['--model', self.model])
        args.append('-')
        entry = ('cp /run/codex-auth.json /codex-home/auth.json && chmod 600 /codex-home/auth.json || exit 18; '
                 + _probe() + 'exec "$@"')
        return self.runner.run(job_key=workspace.name, kind='build', workspace=workspace,
            auth_file=self.auth_file, command=['/bin/sh', '-c', entry, 'sh', *args],
            stdin=prompt.encode(), timeout_seconds=self.timeout_seconds, cancelled=cancelled)

    def stop(self, job_key: str) -> bool:
        return self.runner.destroy(job_key=job_key, kind='build')


class DockerCodexVerifier:
    def __init__(self, runner: DockerCodingSandbox, *, timeout_seconds: float = 120):
        self.runner, self.timeout_seconds = runner, timeout_seconds

    def verify(self, *, package: Path, checks: list[str], cancelled) -> bool:
        if not checks or len(checks) > 20 or any(not isinstance(c, str) or not c.strip() or len(c) > 4096 for c in checks):
            raise ValueError('coding_skill_checks_required')
        script = '\n'.join('(\n' + check + '\n) || exit 1' for check in checks) + '\n'
        command = ['codex', 'sandbox', *_flags(), '-P', 'skill_builder', '-C', '/workspace', '--', '/bin/sh', '-s']
        status = self.runner.run(job_key=package.parent.name, kind='verify', workspace=package,
            command=command, stdin=script.encode(), timeout_seconds=self.timeout_seconds, cancelled=cancelled)
        return status == 'completed'

    def stop(self, job_key: str) -> bool:
        return self.runner.destroy(job_key=job_key, kind='verify')
