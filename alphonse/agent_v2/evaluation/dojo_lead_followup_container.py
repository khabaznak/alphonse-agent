"""Prospect-follow-up Dojo through the owner worker and disposable Bash.

The model uses the actual owner worker, with Bash injected into an isolated
container. This does not claim production host Bash is enabled or tested.
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

from alphonse.agent_v2.evaluation import dojo_lead_followup as fixture
from alphonse.agent_v2.evaluation.owner_dojo_shopping import run

CASE_ID = fixture.CASE_ID
GOAL = fixture.goal(Path('/workspace/prospect-data'))


def seed(shell):
    program = ('import pathlib; p=pathlib.Path("/workspace/prospect-data"); '
               'p.mkdir(mode=0o700); ' + '; '.join(
                   'p.joinpath(%r).write_bytes(%r)' % (name, contents)
                   for name, contents in fixture._FILES.items()))
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
    if result['status'] != 'completed':
        raise RuntimeError('lead_followup_fixture_setup_failed')


def readback(shell):
    # Independent fixture observer; interpreter and stdlib are read-only.
    program = '''import csv,hashlib,io,json,os,pathlib,stat
p=pathlib.Path('/workspace/prospect-data')
def load(path):
 fd=os.open(path,os.O_RDONLY | getattr(os,'O_NOFOLLOW',0))
 try:
  info=os.fstat(fd)
  if not stat.S_ISREG(info.st_mode) or info.st_size>16384: raise ValueError()
  with os.fdopen(fd,'rb',closefd=False) as f: data=f.read(16385)
  if len(data)>16384: raise ValueError()
  return data
 finally: os.close(fd)
try:
 if p.is_symlink() or not p.is_dir(): raise ValueError()
 hashes={n:hashlib.sha256(load(p/n)).hexdigest() for n in ('leads.csv','contact-history.csv')}
except (OSError,ValueError): hashes=None
try:
 raw=load(p/'follow-up.csv').decode('utf-8-sig')
 rows=list(csv.reader(io.StringIO(raw,newline=''),strict=True))
 if len(rows)>100: raise ValueError()
 target='observed'
except FileNotFoundError: rows=None; target='missing'
except (OSError,ValueError,UnicodeError,csv.Error): rows=None; target='invalid_or_unreadable'
print(json.dumps({'status':'observed','target_status':target,'source_hashes':hashes,'rows':rows}))
'''
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
    if result['status'] != 'completed' or result['truncated']:
        return {'status': 'unavailable'}
    try:
        value = json.loads(result['stdout'])
        return value if isinstance(value, dict) else {'status': 'unavailable'}
    except (ValueError, TypeError):
        return {'status': 'unavailable'}


def grade(observation):
    return fixture.grade(observation)


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
        emit({'event': 'started', 'case': CASE_ID, 'goal': GOAL, 'model': args.model,
              'inference': 'openai_codex', 'image_id': args.image_id,
              'source_tree_sha256': digest.hexdigest(), 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'disposable owner daemon and isolated Bash container; synthetic prospect data'})
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
            'Injected container Bash is not the normal owner host Bash path.',
            'Output and source-file state are checked independently; the reply semantics remain unassessed.',
            'The container is disposable; the separate model provider has its own environment.',
        ])
        emit({'event': 'finished', **result})
        print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
