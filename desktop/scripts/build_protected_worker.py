"""Prepare a signing candidate; this does not install or authorize a worker.

The output must be provisioned in a protected directory and validated as a
running peer before use. It must not be substituted for the normal sidecar.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MACHO_MAGIC = {bytes.fromhex(value) for value in (
    'feedface', 'cefaedfe', 'feedfacf', 'cffaedfe',
    'cafebabe', 'bebafeca', 'cafebabf', 'bfbafeca',
)}
PROHIBITED_ENTITLEMENTS = {
    'com.apple.security.get-task-allow', 'com.apple.security.cs.debugger',
    'com.apple.security.cs.allow-jit', 'com.apple.security.cs.allow-unsigned-executable-memory',
    'com.apple.security.cs.disable-executable-page-protection',
    'com.apple.security.cs.disable-library-validation',
    'com.apple.security.cs.allow-dyld-environment-variables',
}


class BuildError(Exception):
    pass


def _run(args, **kwargs):
    result = subprocess.run(args, text=True, capture_output=True, check=False, **kwargs)
    if result.returncode:
        raise BuildError(f'command_failed: {args[0]}')
    return result.stdout + result.stderr


def _identity_available(identity, listing):
    # Require a certificate fingerprint, never a name, private key or ad-hoc "-".
    return (bool(re.fullmatch(r'[0-9A-F]{40}', identity))
            and any(re.fullmatch(r'\s*\d+\)\s+' + identity + r'\s+"[^"]+"\s*', line)
                    for line in listing.splitlines()))


def _macho_files(bundle):
    root = bundle.resolve(strict=True)
    for current, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(current) / name
            if path.is_symlink():
                try:
                    path.resolve(strict=True).relative_to(root)
                except (OSError, ValueError):
                    raise BuildError('bundle_symlink_outside_or_missing') from None
        for name in files:
            path = Path(current) / name
            if path.is_symlink():
                continue
            with path.open('rb') as stream:
                if stream.read(4) in MACHO_MAGIC:
                    yield path


def _entitlements(binary):
    result = subprocess.run(['/usr/bin/codesign', '-d', '--entitlements', ':-', str(binary)],
                            capture_output=True, check=False)
    if result.returncode:
        raise BuildError('worker_entitlements_unavailable')
    if not result.stdout:
        return {}
    try:
        value = plistlib.loads(result.stdout)
    except (ValueError, TypeError, plistlib.InvalidFileException):
        raise BuildError('worker_entitlements_invalid') from None
    if not isinstance(value, dict):
        raise BuildError('worker_entitlements_invalid')
    return value


def _verify_bundle(bundle, team_id):
    executable = bundle / 'alphonse-protected-worker'
    binaries = list(_macho_files(bundle))
    if executable not in binaries or len(binaries) < 2:
        raise BuildError('bundle_missing_worker_or_native_dependencies')
    for binary in binaries:
        _run(['/usr/bin/codesign', '--verify', '--strict', str(binary)])
        info = _run(['/usr/bin/codesign', '-dvv', str(binary)])
        if f'TeamIdentifier={team_id}' not in info.splitlines():
            raise BuildError('bundle_team_identity_mismatch')
        if binary == executable and not re.search(r'flags=.*\bruntime\b', info):
            raise BuildError('worker_hardened_runtime_missing')
        if binary == executable and PROHIBITED_ENTITLEMENTS.intersection(_entitlements(binary)):
            raise BuildError('worker_entitlement_prohibited')


def prepare(identity, team_id, output, python=sys.executable):
    if sys.platform != 'darwin':
        raise BuildError('macos_required')
    if not re.fullmatch(r'[0-9A-F]{40}', identity) or not re.fullmatch(r'[A-Z0-9]{10}', team_id):
        raise BuildError('signing_identity_or_team_invalid')
    output = Path(output).absolute()
    if output.exists() or output.is_symlink() or not output.parent.is_dir():
        raise BuildError('output_must_not_exist_and_parent_must_exist')
    listing = _run(['/usr/bin/security', 'find-identity', '-v', '-p', 'codesigning'])
    if not _identity_available(identity, listing):
        raise BuildError('signing_identity_unavailable')
    # All intermediate files are disposable. An existing output is never replaced.
    with tempfile.TemporaryDirectory(prefix='alphonse-worker-build-', dir=output.parent) as temp:
        temp = Path(temp)
        _run([python, '-m', 'PyInstaller', '--onedir', '--noconfirm',
              '--codesign-identity', identity, '--collect-data', 'alphonse.agent_v2',
              '--name', 'alphonse-protected-worker', '--distpath', str(temp / 'dist'),
              '--workpath', str(temp / 'work'), '--specpath', str(temp),
              str(ROOT / 'alphonse/agent_v2/daemon.py')], cwd=ROOT)
        bundle = temp / 'dist' / 'alphonse-protected-worker'
        _verify_bundle(bundle, team_id)
        if output.exists() or output.is_symlink():
            raise BuildError('output_must_not_exist_and_parent_must_exist')
        shutil.move(str(bundle), str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--identity', required=True, help='40-character uppercase certificate SHA-1 fingerprint')
    parser.add_argument('--team-id', required=True, help='expected 10-character signing Team ID')
    parser.add_argument('--output', required=True, help='new candidate directory; not an installation')
    args = parser.parse_args()
    try:
        prepare(args.identity, args.team_id, args.output)
    except (BuildError, OSError, subprocess.SubprocessError) as exc:
        print(f'Worker candidate not prepared: {exc}', file=sys.stderr)
        return 1
    print('Worker signing candidate prepared. It is not installed or authorized.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
