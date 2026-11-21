"""First turn of a synthetic front-door Dojo, through the owner worker.

No device or network is accessed. This is a static simulated device record,
not a Wyze integration or a live door sensor.
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

CASE_ID = 'front-door-observation'
GOAL = 'I just installed a new Wyze front-door lock. Can you check whether the front door is closed?'
# Invented fixture values. These are not a real household device or a Wyze API.
RECORD = {'name': 'front door', 'brand': 'Wyze', 'door_contact': 'closed', 'lock_state': 'locked',
          'device_id': 'synthetic-front-door', 'ip_address': '192.0.2.10', 'mac_address': '02:00:00:00:00:01',
          'source': 'synthetic local device snapshot'}
RECORD_TEXT = json.dumps(RECORD, sort_keys=True, separators=(',', ':')) + '\n'


def seed(shell):
    program = ('import pathlib; p=pathlib.Path("/workspace/household/devices"); '
               'p.mkdir(parents=True); p.joinpath("front-door.json").write_text(%r)' % RECORD_TEXT)
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
    if result['status'] != 'completed':
        raise RuntimeError('front_door_fixture_setup_failed')


def readback(shell):
    # This checks fixture integrity, not whether the model understood its result.
    program = ('import hashlib,pathlib; p=pathlib.Path("/workspace/household/devices/front-door.json"); '
               'print(hashlib.sha256(p.read_bytes()).hexdigest())')
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
    if result['status'] != 'completed' or result['truncated']:
        return {'status': 'unavailable'}
    return {'status': 'observed', 'record_sha256': result['stdout'].strip()}


def grade(readback_result):
    if readback_result.get('status') != 'observed':
        return {'effect_passed': None, 'fixture_integrity': 'unavailable', 'response_semantics': 'unassessed'}
    unchanged = readback_result.get('record_sha256') == hashlib.sha256(RECORD_TEXT.encode()).hexdigest()
    return {'effect_passed': None, 'fixture_integrity': 'unchanged' if unchanged else 'changed',
            'response_semantics': 'unassessed'}


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
              'scope': 'synthetic static device record; disposable owner daemon and Bash container'})
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
            'Static synthetic device snapshot; no device discovery, network or real Wyze service.',
            'Default registry with synthetic settings; the owner processor filters which tools are advertised.',
            'Only the first turn is exercised; memory, code use, events and notifications are unassessed.',
            'Door-state answer semantics and source grounding require independent review.',
        ])
        emit({'event': 'finished', **result})
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
