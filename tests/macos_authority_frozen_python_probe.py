"""Opt-in, disposable probe of ad-hoc hardened PyInstaller compatibility.

This does not build or install Alphonse, change keychain trust, or establish a
worker security boundary. It records whether a tiny frozen interpreter runs
before and after the same executable receives hardened-runtime signing. The
result is an observation for this interpreter, operating system and toolchain.

Run with the project's Python: ``.venv/bin/python tests/macos_authority_frozen_python_probe.py``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def _run(args, *, env, timeout=90):
    return subprocess.run(args, env=env, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=timeout, check=False)


def main():
    if sys.platform != 'darwin':
        raise SystemExit('macOS required')
    with tempfile.TemporaryDirectory(prefix='alphonse-frozen-worker-', dir='/private/tmp') as directory:
        root = Path(directory)
        source = root / 'worker_probe.py'
        source.write_text('import ctypes, json, sqlite3, ssl, sys\n'
                          'print(json.dumps({"frozen":bool(getattr(sys,"frozen",False)),'
                          '"sqlite":sqlite3.sqlite_version,"ssl":bool(ssl.OPENSSL_VERSION),'
                          '"ctypes":bool(ctypes.CDLL(None))}))\n', encoding='utf-8')
        env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin:/usr/sbin:/sbin'),
               'HOME': str(root), 'TMPDIR': str(root),
               'PYINSTALLER_CONFIG_DIR': str(root / 'cache')}
        built = _run([sys.executable, '-m', 'PyInstaller', '--onedir', '--noconfirm',
                      '--name', 'worker-probe', '--distpath', str(root / 'dist'),
                      '--workpath', str(root / 'work'), '--specpath', str(root), str(source)], env=env)
        if built.returncode:
            print(json.dumps({'status': 'build_failed', 'exit_code': built.returncode}))
            return 1
        executable = root / 'dist' / 'worker-probe' / 'worker-probe'
        baseline = _run([str(executable)], env=env, timeout=30)
        signed = _run(['/usr/bin/codesign', '--force', '--sign', '-', '--options', 'runtime',
                       str(executable)], env=env, timeout=30)
        if signed.returncode:
            print(json.dumps({'status': 'signing_failed', 'baseline_exit': baseline.returncode,
                              'signing_exit': signed.returncode}))
            return 1
        hardened = _run([str(executable)], env=env, timeout=30)
        diagnostic = hardened.stdout + '\n' + hardened.stderr
        print(json.dumps({'status': 'observed', 'baseline_exit': baseline.returncode,
                          'hardened_exit': hardened.returncode,
                          'reported_team_id_library_mismatch': 'different Team IDs' in diagnostic,
                          'frozen_confirmed': '"frozen": true' in baseline.stdout,
                          'scope': 'temporary single-machine probe; no OS trust or installed app changed'},
                         sort_keys=True))
        return 0 if baseline.returncode == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
