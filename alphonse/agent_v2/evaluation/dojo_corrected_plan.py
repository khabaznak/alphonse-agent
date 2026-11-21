"""Check a corrected household plan through the normal synthetic owner worker.

The household notes are invented and disposable. The result records fixture
integrity separately from the manual assessment of the agent's answer.
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

CASE_ID = 'corrected-gardener-plan'
GOAL = ('Review the household gardener notes in /workspace/household/garden '
        'and tell me the date of the upcoming gardener visit.')
NOTES = {
    'visit-original.txt': 'November 8, 2026: The gardener is scheduled for Tuesday, December 1, 2026.\n',
    'visit-update.txt': ('November 12, 2026: The gardener rescheduled the visit to Thursday, '
                         'December 3, 2026. The December 1 date is no longer valid.\n'),
}


def seed(shell):
    program = ('import pathlib; p=pathlib.Path("/workspace/household/garden"); '
               'p.mkdir(parents=True); notes=%r; '
               '[(p / name).write_text(content) for name, content in notes.items()]' % NOTES)
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
    if result['status'] != 'completed':
        raise RuntimeError('corrected_plan_fixture_setup_failed')


def readback(shell):
    program = ('import hashlib,json,pathlib; p=pathlib.Path("/workspace/household/garden"); '
               'names=%r; print(json.dumps({n:hashlib.sha256((p/n).read_bytes()).hexdigest() '
               'for n in names}, sort_keys=True))' % list(NOTES))
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
    if result['status'] != 'completed' or result['truncated']:
        return {'status': 'unavailable'}
    try:
        return {'status': 'observed', 'note_sha256': json.loads(result['stdout'])}
    except (TypeError, ValueError):
        return {'status': 'unavailable'}


def grade(readback_result):
    if readback_result.get('status') != 'observed':
        integrity = 'unavailable'
    else:
        expected = {name: hashlib.sha256(content.encode()).hexdigest() for name, content in NOTES.items()}
        integrity = 'unchanged' if readback_result.get('note_sha256') == expected else 'changed'
    return {'effect_passed': None, 'fixture_integrity': integrity, 'response_semantics': 'unassessed'}


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
        def emit(row):
            stream.write(json.dumps(row, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': CASE_ID, 'inference': 'openai_codex', 'model': args.model,
              'image_id': args.image_id, 'source_tree_sha256': digest.hexdigest(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'synthetic household notes; disposable owner daemon and Bash container'})
        fd = os.open(args.journal.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        try:
            result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                         model=args.model, timeout_seconds=120, scenario=sys.modules[__name__], full_registry=True)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'task_passed': None,
                  'error_type': type(exc).__name__})
            raise
        result['limitations'].extend([
            'Invented notes are static; this does not test live scheduling or real household data.',
            'Default registry with synthetic settings; the owner processor filters advertised tools.',
            'Fixture integrity is independent of answer correctness; response and source grounding require review.',
        ])
        emit({'event': 'finished', **result})
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
