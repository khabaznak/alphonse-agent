"""Exercise an offline macOS sandbox boundary with disposable synthetic data.

This intentionally does not launch a model session or contact a provider. It
checks specific file and loopback operations, not all possible filesystem or
network escapes. It is not authorization or a live-pilot isolation guarantee.
"""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import threading
from urllib.request import ProxyHandler, build_opener


def _policy(*, protected: Path, workspace: Path, allowed_port: int) -> str:
    if not protected.is_absolute() or not workspace.is_absolute():
        raise ValueError('absolute_paths_required')
    if type(allowed_port) is not int or not 1 <= allowed_port <= 65535:
        raise ValueError('invalid_port')
    # SBPL strings are quoted, including non-ASCII and embedded punctuation.
    # This denies access to one directory, not every possible private location.
    protected_text = json.dumps(str(protected), ensure_ascii=True)
    workspace_text = json.dumps(str(workspace), ensure_ascii=True)
    return ('(version 1) (allow default) '
            f'(deny file-read-data (subpath {protected_text})) '
            f'(deny file-write*) (allow file-write* (subpath {workspace_text})) '
            '(deny network*) '
            f'(allow network-outbound (remote ip "localhost:{allowed_port}"))')


def probe() -> dict:
    """Run real local probes; no prompts, secrets or provider calls are used.

    A caller needs permission to create local listeners and launch sandbox-exec.
    The result deliberately does not qualify a pilot or prove agent attribution.
    """
    sandbox = Path('/usr/bin/sandbox-exec')
    cat = Path('/bin/cat')
    curl = Path('/usr/bin/curl')
    codex = Path('/opt/homebrew/bin/codex')
    if not all(path.is_file() for path in (sandbox, cat, curl, codex)):
        return {'probe': 'unavailable', 'live_ready': False, 'reason': 'required_executable_not_found'}
    with TemporaryDirectory(prefix='alphonse-isolation-probe-') as temporary:
        root = Path(temporary).resolve()
        workspace, protected = root / 'workspace', root / 'private'
        workspace.mkdir()
        protected.mkdir()
        (workspace / 'allowed.txt').write_text('public-synthetic', encoding='utf-8')
        (protected / 'denied.txt').write_text('private-synthetic', encoding='utf-8')

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                data = b'fixture-ok'
                self.send_response(200)
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        servers = []
        threads = []
        try:
            for _ in range(2):
                try:
                    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
                except OSError:
                    return {'probe': 'unavailable', 'live_ready': False,
                            'reason': 'local_listener_unavailable'}
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                servers.append(server)
                threads.append(thread)
            allowed_port, denied_port = (server.server_port for server in servers)
            opener = build_opener(ProxyHandler({}))
            # Establish that both listeners are reachable before testing whether
            # the child policy blocks one. An absent server is not isolation.
            try:
                reachable = all(opener.open(f'http://127.0.0.1:{port}/', timeout=2).read() == b'fixture-ok'
                                for port in (allowed_port, denied_port))
            except OSError:
                reachable = False
            profile = _policy(protected=protected, workspace=workspace, allowed_port=allowed_port)
            # HOME is a genuine, disposable subprocess home, not the user's
            # inherited home. No API key, profile, proxy or Codex setting is
            # forwarded. The subprocess does not receive fixture secrets.
            env = {'HOME': str(workspace), 'TMPDIR': str(workspace),
                   'PATH': '/opt/homebrew/bin:/usr/bin:/bin', 'LANG': 'C.UTF-8'}

            def run(argv):
                try:
                    return subprocess.run([str(sandbox), '-p', profile, *map(str, argv)],
                        cwd=workspace, env=env, capture_output=True, text=True,
                        timeout=10, check=False)
                except (OSError, subprocess.SubprocessError):
                    return None

            visible = run([cat, workspace / 'allowed.txt'])
            hidden = run([cat, protected / 'denied.txt'])
            descendant = run(['/bin/sh', '-c', 'exec /bin/cat "$1"', 'probe', protected / 'denied.txt'])
            allowed = run([curl, '--noproxy', '*', '--fail', '--silent', '--show-error',
                           '--max-time', '2', f'http://127.0.0.1:{allowed_port}/'])
            denied = run([curl, '--noproxy', '*', '--fail', '--silent', '--show-error',
                          '--max-time', '2', f'http://127.0.0.1:{denied_port}/'])
            version = run([codex, '--version'])
            checks = {
                'both_servers_reachable_from_parent': reachable,
                'workspace_read': visible is not None and visible.returncode == 0 and visible.stdout == 'public-synthetic',
                'protected_read_denied': hidden is not None and hidden.returncode != 0 and hidden.stdout == '',
                'descendant_protected_read_denied': (descendant is not None and descendant.returncode != 0
                                                      and descendant.stdout == ''),
                'fixture_port_reachable': allowed is not None and allowed.returncode == 0 and allowed.stdout == 'fixture-ok',
                'other_port_denied': denied is not None and denied.returncode != 0 and denied.stdout == '',
                'codex_version_starts': version is not None and version.returncode == 0 and version.stdout.startswith('codex-cli '),
            }
            return {'probe': 'passed' if all(checks.values()) else 'failed_or_unavailable',
                    'checks': checks, 'live_ready': False,
                    'limitations': ['Only the selected temporary private path and two loopback ports were tested.',
                                    'The policy allows other paths by default and is not a complete confidentiality boundary.',
                                    'No provider network, credentials, model calls, CLI session or agent attribution was tested.',
                                    'An isolated, metered live configuration and explicit approval remain required.']}
        finally:
            for server in servers:
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join(timeout=3)


def main() -> int:
    result = probe()
    print(json.dumps(result, indent=2))
    return 0 if result.get('probe') == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
