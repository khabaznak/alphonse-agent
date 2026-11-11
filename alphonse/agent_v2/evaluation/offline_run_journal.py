"""Durable receipts for offline development comparisons, not a live benchmark.

Create a fresh JSONL journal before any attempt; fsync each start and terminal
record. Inspect partial journals without resuming or silently rerunning work.
The journal retains synthetic traces and is not an OS/network sandbox, a
complete provenance attestation, or evidence of delivery or real-model quality.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import monotonic
from typing import Mapping
from uuid import uuid4

from .candidate_comparison import SingleLoopFixtureAdapter
from .isolated_v3_arm import IsolatedV3ReadArm
from .simplified_loop import Adapter, _run, load_frozen


def _timestamp():
    return datetime.now(timezone.utc).isoformat()


def _append(stream, record):
    line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n'
    stream.write(line)
    stream.flush()
    os.fsync(stream.fileno())


def _valid_outcome(outcome):
    """Validate structural consistency, not provenance or outcome truth."""
    if (not isinstance(outcome, dict) or
            not {'availability', 'passed', 'metrics', 'oracle', 'events'} <= set(outcome)):
        return False
    availability, passed = outcome.get('availability'), outcome.get('passed')
    if availability == 'available':
        oracle = outcome.get('oracle')
        if (type(passed) is not bool or not isinstance(outcome.get('status'), str) or not outcome['status'].strip()
                or not isinstance(outcome.get('metrics'), dict)
                or not isinstance(outcome.get('events'), list)
                or not isinstance(oracle, dict) or type(oracle.get('passed')) is not bool):
            return False
        checks = oracle.get('checks')
        return (isinstance(checks, dict) and bool(checks)
                and all(type(value) is bool or value is None for value in checks.values())
                and oracle['passed'] is passed and passed is all(checks.values()))
    if availability in {'unavailable', 'error'}:
        return (passed is None and isinstance(outcome.get('reason'), str)
                and bool(outcome['reason'].strip()) and outcome.get('metrics') is None
                and outcome.get('oracle') is None
                and (outcome.get('events') == [] if availability == 'unavailable'
                     else outcome.get('events') is None))
    return False


def _fsync_parent(path: Path):
    """Persist the newly created directory entry or fail before any attempt."""
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def run_journal(corpus_path: str | Path, output: str | Path, *,
                arms: Mapping[str, Adapter | None] | None = None) -> dict:
    """Execute once into an exclusive, durable journal; never resume or overwrite.

    A started record without a terminal record has an unknown outcome; it must
    not be retried automatically. Even a terminal record cannot prove a private
    user received a response. Partial tool traces inside a killed attempt may be
    lost. The selected source hashes describe only these harness modules, not
    all transitive inputs or the executing environment.
    """
    if arms is None:
        arms = {'current_v3': IsolatedV3ReadArm(projection_enabled=False),
                'targeted_v3_projection': IsolatedV3ReadArm(projection_enabled=True),
                'single_loop': SingleLoopFixtureAdapter()}
    if not arms or any(not isinstance(name, str) or not name.strip() for name in arms):
        raise ValueError('arm_names_required')
    corpus, digest = load_frozen(corpus_path)
    schedule = [{'case_id': case['id'], 'arm': name} for case in corpus['cases'] for name in arms]
    run_id = str(uuid4())
    source_dir = Path(__file__).resolve().parent
    source_names = ('offline_run_journal.py', 'offline_three_arm.py', 'simplified_loop.py',
                    'candidate_comparison.py', 'isolated_v3_arm.py')
    sources = {name: hashlib.sha256((source_dir / name).read_bytes()).hexdigest() for name in source_names}
    header = {'event': 'run_started', 'schema_version': 1, 'run_id': run_id,
              'experiment_kind': 'offline_scripted_plumbing', 'recorded_at': _timestamp(), 'corpus_sha256': digest,
              'split': corpus['split'], 'schedule': schedule,
              'arms': {name: str(getattr(arm, 'revision', 'unavailable' if arm is None else 'unknown'))
                       for name, arm in arms.items()},
              'python': sys.version, 'platform': platform.platform(), 'selected_source_sha256': sources,
              'limitations': ['Not a live model benchmark or OS/network sandbox.',
                              'Selected source hashes do not cover all runtime inputs.',
                              'Partial traces in interrupted attempts are unknown.',
                              'This journal does not establish delivery or independently anchored provenance.']}
    # Opening exclusively happens before any adapter is invoked. A prior partial
    # journal blocks retries at the same path; there is no implicit resume.
    output_path = Path(output)
    with output_path.open('x', encoding='utf-8') as stream:
        _append(stream, header)
        # A synced file alone does not guarantee that its new name is durable.
        # Fail closed if the directory cannot be synced; no attempt has started.
        _fsync_parent(output_path)
        for case in corpus['cases']:
            for name, arm in arms.items():
                attempt_id = json.dumps([case['id'], name], ensure_ascii=False, separators=(',', ':'))
                _append(stream, {'event': 'attempt_started', 'run_id': run_id, 'recorded_at': _timestamp(),
                                 'attempt_id': attempt_id, 'case_id': case['id'], 'arm': name})
                started = monotonic()
                try:
                    outcome = _run(case, arm)
                    # Check serializability before emitting terminal record. A
                    # malformed payload is infrastructure failure, not success.
                    json.dumps(outcome, allow_nan=False)
                    if not _valid_outcome(outcome):
                        raise ValueError('invalid_outcome')
                except Exception as exc:
                    outcome = {'availability': 'error', 'passed': None,
                               'reason': 'runner_error:' + type(exc).__name__,
                               'metrics': None, 'oracle': None, 'events': None}
                _append(stream, {'event': 'attempt_finished', 'run_id': run_id, 'recorded_at': _timestamp(),
                                 'elapsed_seconds': monotonic() - started,
                                 'attempt_id': attempt_id, 'outcome': outcome})
        _append(stream, {'event': 'run_finished', 'run_id': run_id, 'recorded_at': _timestamp(),
                         'attempt_count': len(schedule)})
    return inspect_journal(output)


def inspect_journal(path: str | Path) -> dict:
    """Inspect a complete, interrupted, or damaged journal; never execute work."""
    records = []
    damage = None
    with Path(path).open('rb') as stream:
        for index, line in enumerate(stream, 1):
            if not line.endswith(b'\n'):
                damage = f'truncated_line:{index}'
                break
            try:
                record = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                damage = f'invalid_line:{index}'
                break
            if not isinstance(record, dict):
                damage = f'invalid_record:{index}'
                break
            records.append(record)
    if not records or records[0].get('event') != 'run_started':
        return {'complete': False, 'valid': False, 'damage': damage or 'missing_header',
                'started': [], 'finished': [], 'unknown': [], 'not_started': []}
    header = records[0]
    run_id, schedule = header.get('run_id'), header.get('schedule')
    if not isinstance(run_id, str) or not run_id or not isinstance(schedule, list) or not schedule or any(
            not isinstance(x, dict) or set(x) != {'case_id', 'arm'} or
            not all(isinstance(x[k], str) and bool(x[k].strip()) for k in ('case_id', 'arm')) for x in schedule):
        return {'complete': False, 'valid': False, 'damage': 'invalid_header',
                'started': [], 'finished': [], 'unknown': [], 'not_started': []}
    expected = [json.dumps([x['case_id'], x['arm']], ensure_ascii=False, separators=(',', ':')) for x in schedule]
    if len(expected) != len(set(expected)):
        return {'complete': False, 'valid': False, 'damage': 'duplicate_schedule',
                'started': [], 'finished': [], 'unknown': [], 'not_started': []}
    started, finished, outcomes, final = [], [], {}, False
    for row in records[1:]:
        event, identity = row.get('event'), row.get('attempt_id')
        if row.get('run_id') != run_id or final:
            damage = damage or 'invalid_sequence'
            break
        if event == 'attempt_started':
            i = len(started)
            if (i >= len(expected) or identity != expected[i] or len(finished) != i or
                    row.get('case_id') != schedule[i]['case_id'] or row.get('arm') != schedule[i]['arm']):
                damage = damage or 'invalid_sequence'
                break
            started.append(identity)
        elif event == 'attempt_finished':
            if (not started or len(started) != len(finished) + 1 or identity != started[-1]
                    or not _valid_outcome(row.get('outcome'))):
                damage = damage or 'invalid_sequence'
                break
            finished.append(identity)
            outcomes[identity] = row['outcome']
        elif event == 'run_finished':
            if len(finished) != len(expected) or row.get('attempt_count') != len(expected):
                damage = damage or 'invalid_sequence'
                break
            final = True
        else:
            damage = damage or 'invalid_sequence'
            break
    return {'run_id': run_id, 'corpus_sha256': header.get('corpus_sha256'),
            'complete': final and damage is None, 'valid': damage is None,
            'qualification': 'unassessed',
            'damage': damage, 'started': started, 'finished': finished,
            'unknown': [x for x in started if x not in finished],
            'not_started': expected[len(started):], 'outcomes': outcomes,
            'limitations': header.get('limitations', [])}


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('corpus', type=Path)
    run.add_argument('output', type=Path)
    inspect = commands.add_parser('inspect')
    inspect.add_argument('journal', type=Path)
    args = parser.parse_args(argv)
    result = run_journal(args.corpus, args.output) if args.command == 'run' else inspect_journal(args.journal)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return int(not result['complete'] or any(x.get('passed') is not True for x in result.get('outcomes', {}).values()))


if __name__ == '__main__':
    raise SystemExit(main())
