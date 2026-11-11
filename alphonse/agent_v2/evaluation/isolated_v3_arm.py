"""Offline flag-off/on V3 fixture arms, executed in separate processes.

These arms use the pre-existing scripted V3 read fixture. The targeted variant
enables the existing experimental context-projection flag, not a new engine.
This is plumbing evidence only: mocked Jev/providers and a single scripted
read do not qualify this variant or establish a performance improvement.
"""
from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys

from .simplified_loop import AdapterUnavailable, Run

_FLAG = 'ALPHONSE_V3_CONTEXT_PROJECTIONS'
_WORKSPACE = Path(__file__).resolve().parents[3]


class IsolatedV3ReadArm:
    offline = True

    def __init__(self, *, projection_enabled: bool):
        if type(projection_enabled) is not bool:
            raise ValueError('projection_enabled_must_be_boolean')
        self.projection_enabled = projection_enabled
        self.revision = ('working-tree:hierarchical-v3:scripted-read:projection-'
                         + ('on' if projection_enabled else 'off'))

    def __call__(self, case, fixture_root, emit):
        # A separate interpreter avoids process-global flag contamination when
        # compared with other arms, including concurrent comparisons. Do not
        # forward credentials or provider configuration into the child.
        env = {k: v for k, v in os.environ.items() if k in {'PATH', 'LANG', 'LC_ALL', 'TMPDIR', 'SYSTEMROOT'}}
        env[_FLAG] = '1' if self.projection_enabled else '0'
        completed = subprocess.run([sys.executable, '-m', __name__, '--worker'],
            cwd=str(_WORKSPACE), env=env, input=json.dumps({'case': case, 'root': str(fixture_root)}),
            capture_output=True, text=True, timeout=30, check=False)
        if completed.returncode != 0:
            raise RuntimeError('isolated_v3_worker_failed')
        try:
            payload = json.loads(completed.stdout)
            if payload['projection_enabled'] is not self.projection_enabled:
                raise ValueError('isolated_v3_flag_mismatch')
            for event in payload['events']:
                emit(event)
            if payload.get('unavailable'):
                raise AdapterUnavailable(payload['unavailable'])
            return Run(**payload['run'])
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError('isolated_v3_worker_invalid') from exc


def _worker():
    from .candidate_comparison import V3ReadFixtureAdapter
    from alphonse.agent_v2.core.intelligence.v3.projections import enabled
    request = json.load(sys.stdin)
    events = []
    try:
        run = V3ReadFixtureAdapter()(request['case'], Path(request['root']), events.append)
        payload = {'run': asdict(run), 'events': events, 'projection_enabled': enabled()}
    except AdapterUnavailable as exc:
        payload = {'unavailable': str(exc), 'events': events, 'projection_enabled': enabled()}
    except Exception as exc:
        # Preserve observations emitted before a failure; exception messages can
        # include sensitive arguments, so report only the exception type.
        payload = {'run': asdict(Run(status='error', error=type(exc).__name__)),
                   'events': events, 'projection_enabled': enabled()}
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == '__main__' and sys.argv[1:] == ['--worker']:
    _worker()
