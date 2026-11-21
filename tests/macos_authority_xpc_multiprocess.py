"""Disposable macOS XPC rejection smoke test; no owner data or real credentials.

Compiles and ad-hoc-signs separate processes in a temporary directory, registers
a temporary per-user launchd job, and always attempts to remove it. Requires a
logged-in macOS GUI session and permission to bootstrap a per-user launchd job.
This does NOT exercise an Apple-developer-signed positive peer or a protected
service identity.
"""
from __future__ import annotations

import os
from pathlib import Path
import plistlib
import shutil
import subprocess
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def command(*args, timeout=30, check=True):
    return subprocess.run(args, text=True, capture_output=True, timeout=timeout, check=check)


def main():
    if os.uname().sysname != 'Darwin':
        raise SystemExit('macOS required')
    domain = 'gui/' + str(os.getuid())
    label = 'local.alphonse.authority-test-' + uuid.uuid4().hex
    with tempfile.TemporaryDirectory(prefix='alphonse-xpc-fixture-', dir='/private/tmp') as directory:
        root = Path(directory)
        cache = root / 'module-cache'
        cache.mkdir()
        server = root / 'server'
        desktop = root / 'desktop'
        worker = root / 'worker'
        command('xcrun', 'swiftc', '-parse-as-library', '-module-cache-path', str(cache),
                str(ROOT / 'alphonse/agent_v2/authority_service/macos/PeerPolicy.swift'),
                str(ROOT / 'tests/macos_authority_xpc_service.swift'), '-o', str(server), timeout=120)
        command('xcrun', 'swiftc', '-parse-as-library', '-module-cache-path', str(cache),
                str(ROOT / 'tests/macos_authority_xpc_client.swift'), '-o', str(desktop), timeout=120)
        shutil.copy2(desktop, worker)
        for binary, identifier in ((server, 'local.alphonse.fixture.server'),
                                   (desktop, 'local.alphonse.fixture.desktop'),
                                   (worker, 'local.alphonse.fixture.worker')):
            command('codesign', '--force', '--sign', '-', '--identifier', identifier, str(binary))
            command('codesign', '--verify', '--strict', str(binary))
        config = root / 'fixture.plist'
        config.write_bytes(plistlib.dumps({
            'Label': label, 'ProgramArguments': [str(server), label],
            'MachServices': {label + suffix: True for suffix in
                             ('.open', '.desktop', '.worker', '.mock-desktop', '.mock-worker')},
            'StandardOutPath': str(root / 'service.out'), 'StandardErrorPath': str(root / 'service.err'),
        }))
        registered = False
        try:
            bootstrap = command('launchctl', 'bootstrap', domain, str(config), check=False)
            if bootstrap.returncode:
                raise RuntimeError('fixture_launchd_bootstrap_failed: ' + bootstrap.stderr.strip())
            registered = True
            # Baselines establish each independently signed client can reach
            # this same service; role checks then must reject both clients.
            observations = {}
            for client_name, client in (('desktop', desktop), ('worker', worker)):
                for listener in ('open', 'desktop', 'worker', 'mock-desktop', 'mock-worker'):
                    result = command(str(client), label + '.' + listener, timeout=12, check=False)
                    if result.returncode:
                        raise RuntimeError('fixture_client_failed:' + client_name + ':' + listener + ':' + result.stdout.strip())
                    outcome = result.stdout.strip()
                    observations[client_name + '_' + listener] = outcome
                    should_deliver = listener == 'open' or listener == 'mock-' + client_name
                    expected = outcome == 'delivered' if should_deliver else outcome in ('rejected', 'invalidated')
                    if not expected:
                        raise RuntimeError('fixture_unexpected_result:' + client_name + ':' + listener + ':' + outcome)
            print('cross-process ad-hoc baselines and rejection checks:', observations, flush=True)

            # The endpoint remains the open listener. The client alone pins a
            # service identity. Identifier-only pinning is deliberately weak
            # and is used solely to establish the cross-process mechanics.
            valid = command(str(desktop), label + '.open',
                            'identifier "local.alphonse.fixture.server"', timeout=12, check=False)
            if valid.returncode or valid.stdout.strip() != 'delivered':
                raise RuntimeError('fixture_client_service_positive_failed')
            output = root / 'service.out'
            def ping_count():
                return output.read_text().count('synthetic-ping\n') if output.exists() else 0
            before = ping_count()
            invalid = command(str(desktop), label + '.open',
                              'anchor apple generic and identifier "local.alphonse.authority" and '
                              'certificate leaf[subject.OU] = "ABCD123456"', timeout=12, check=False)
            if invalid.returncode or invalid.stdout.strip() not in ('rejected', 'invalidated'):
                raise RuntimeError('fixture_client_service_mismatch_not_rejected')
            # A short observation window can establish a received ping, not
            # prove that no request was delivered if no line is observed.
            time.sleep(.25)
            received = ping_count() > before
            print('client service-signature mismatch reply:', invalid.stdout.strip(),
                  '; first harmless request observed by mismatched service:', received, flush=True)
        finally:
            if registered:
                removed = command('launchctl', 'bootout', domain + '/' + label, check=False)
                if removed.returncode:
                    raise RuntimeError('fixture_launchd_cleanup_failed: ' + removed.stderr.strip())
                remaining = command('launchctl', 'print', domain + '/' + label, check=False)
                if remaining.returncode == 0:
                    raise RuntimeError('fixture_launchd_cleanup_unconfirmed')
                print('temporary launchd registration removed', flush=True)


if __name__ == '__main__':
    main()
