"""Observe recovery from an unavailable household report in the owner Dojo.

All files are invented. An independent fixture check assesses final integrity;
the model's response and whether it handled the missing source require review.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

from alphonse.agent_v2.evaluation.owner_dojo_shopping import run

CASE_ID = 'unavailable-water-report'
GOAL = ('What was the water pressure at the last inspection? The report is '
        'at /workspace/household/water/current-report.txt. If it is unavailable, '
        'see whether another household record can answer and tell me what you found.')
NOTE = 'Inspection note, November 14, 2026: water pressure measured 43 psi.\n'
LINK_TARGET = 'archive/current-inspection.txt'


def seed(shell):
    program = ('import pathlib; p=pathlib.Path("/workspace/household/water"); '
               '(p/"archive").mkdir(parents=True); '
               '(p/"archive"/"technician-note.txt").write_text(%r); '
               '(p/"current-report.txt").symlink_to(%r)' % (NOTE, LINK_TARGET))
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
    if result['status'] != 'completed' or result['exit_code'] != 0:
        raise RuntimeError('unavailable_source_fixture_setup_failed')


def readback(shell):
    program = ('import hashlib,json,pathlib; p=pathlib.Path("/workspace/household/water"); '
               'f=p/"current-report.txt"; n=p/"archive"/"technician-note.txt"; '
               'print(json.dumps({"primary_is_symlink":f.is_symlink(), '
               '"primary_target":str(f.readlink()) if f.is_symlink() else None, '
               '"primary_available":f.exists(), '
               '"note_sha256":hashlib.sha256(n.read_bytes()).hexdigest()}, sort_keys=True))')
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
    if result['status'] != 'completed' or result['exit_code'] != 0 or result['truncated']:
        return {'status': 'unavailable'}
    try:
        return {'status': 'observed', **json.loads(result['stdout'])}
    except (TypeError, ValueError):
        return {'status': 'unavailable'}


def grade(observation):
    if observation.get('status') != 'observed':
        integrity = 'unavailable'
    else:
        integrity = ('unchanged' if observation.get('primary_is_symlink') is True
                     and observation.get('primary_target') == LINK_TARGET
                     and observation.get('primary_available') is False
                     and observation.get('note_sha256') == hashlib.sha256(NOTE.encode()).hexdigest()
                     else 'changed')
    return {'effect_passed': None, 'fixture_integrity': integrity,
            'response_semantics': 'unassessed', 'recovery_semantics': 'unassessed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(event):
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': CASE_ID, 'goal': GOAL, 'model': args.model,
              'image_id': args.image_id, 'source_tree_sha256': digest.hexdigest(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'synthetic owner worker; disposable files and container Bash'})
        try:
            result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                         model=args.model, timeout_seconds=120, scenario=sys.modules[__name__], full_registry=True)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'task_passed': None,
                  'error_type': type(exc).__name__})
            raise
        result['limitations'].append('The missing source is a dangling file link, not a network outage; response and recovery require review.')
        emit({'event': 'finished', **result})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
