"""Summarize an existing offline synthetic-suite journal without rerunning it.

The journal and its fixture grades are caller-controlled and unaudited. This is
not an external-agent benchmark, delivery evidence, or release qualification.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

from .benchmark_summary import summarize
from .offline_run_journal import inspect_journal
from .synthetic_suite_journal import normalize_suite_result


def summarize_synthetic_journal(path: str | Path) -> dict:
    """Account for finished, interrupted, and unstarted scheduled fixtures.

    This reads and validates the journal once. It does not inspect or rerun
    inputs, verify the reported fixture grade, or infer real delivery or other
    optional measurements. Invalid or damaged journals fail closed; a valid
    interrupted journal remains a usable, explicitly incomplete snapshot.
    """
    inspected = inspect_journal(path, include_details=True)
    if not inspected.get('valid'):
        raise ValueError('invalid_synthetic_journal')
    if inspected.get('experiment_kind') != 'offline_synthetic_suite_plumbing':
        raise ValueError('synthetic_journal_kind_required')
    schema_version = inspected.get('header_schema_version')
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError('invalid_synthetic_journal_schema')
    digest = inspected.get('corpus_sha256')
    if not isinstance(digest, str) or re.fullmatch(r'[0-9a-f]{64}', digest) is None:
        raise ValueError('invalid_synthetic_journal_digest')
    if inspected['complete'] and type(inspected.get('terminal_attempt_count')) is not int:
        raise ValueError('invalid_synthetic_journal_attempt_count')
    schedule, attempts = [], []
    for pair in inspected['schedule']:
        identity = json.dumps([pair['case_id'], pair['arm']], ensure_ascii=False, separators=(',', ':'))
        item = {'attempt_id': identity, 'case_id': pair['case_id'], 'arm': pair['arm']}
        schedule.append(item)
        if identity in inspected['not_started']:
            continue
        if identity in inspected['unknown']:
            attempts.append({**item, 'state': 'interrupted', 'grade': 'unknown', 'elapsed_ms': None})
            continue
        outcome = inspected['outcomes'][identity]
        seconds = inspected['elapsed_seconds'][identity]
        if seconds is not None:
            try:
                invalid = (isinstance(seconds, bool) or not isinstance(seconds, (int, float))
                           or not math.isfinite(seconds) or seconds < 0 or not math.isfinite(seconds * 1000))
            except OverflowError:
                invalid = True
            if invalid:
                raise ValueError('invalid_synthetic_journal_elapsed')
        elapsed = None if seconds is None else seconds * 1000
        if outcome['availability'] == 'available':
            try:
                normalized = normalize_suite_result(outcome.get('details'))
            except (ValueError, TypeError, KeyError):
                raise ValueError('synthetic_fixture_grade_required') from None
            if normalized != outcome or outcome['oracle']['checks'] != {'fixture_only': outcome['passed']}:
                raise ValueError('synthetic_fixture_grade_required')
            state, grade = 'finished', 'pass' if outcome['passed'] else 'fail'
        elif outcome['availability'] == 'unavailable':
            state, grade = 'unavailable', 'unknown'
        else:
            state, grade = 'finished', 'unknown'
        attempts.append({**item, 'state': state, 'grade': grade, 'elapsed_ms': elapsed})
    report = summarize(schedule, attempts)
    report['journal'] = {'run_id': inspected['run_id'], 'complete': inspected['complete'],
                         'kind': inspected['experiment_kind'], 'input_sha256': digest,
                         'input_digest_verification': 'unverified_header'}
    report['limitations'].extend([
        'Fixture grades are caller-supplied journal results and have not been independently audited.',
        'The source journal does not support repeated identical case/arm pairs.',
        'Input digest syntax is checked; header provenance and input bytes are not independently verified.',
        'This synthetic journal is not an external baseline or evidence of actual delivery.',
    ])
    return report


def main(argv: list[str] | None = None) -> int:
    """Print a validated fixture-only summary of an existing journal."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path, help='existing synthetic-suite JSONL journal')
    args = parser.parse_args(argv)
    try:
        report = summarize_synthetic_journal(args.journal)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'synthetic journal unavailable: {type(exc).__name__}\n')
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
