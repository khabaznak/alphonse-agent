"""Disposable host Dojo: find an alternative when a maintenance report is missing.

The fixture does not constrain Bash or require any particular tool sequence.
It records only bounded effect and response indicators, not commands or host data.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re

CASE_ID = 'host-unavailable-maintenance-report'
NOTE = b'Inspection record, 2026-10-18\nTechnician observed incoming water pressure: 53.5 psi.\n'
REPORT = 'inspection-report.txt'
ALTERNATIVE = 'technician-record.txt'


def seed(root: Path) -> Path:
    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError('dojo_fixture_root_invalid')
    workspace = root / 'maintenance-data'
    workspace.mkdir(mode=0o700)
    (workspace / REPORT).symlink_to('missing-report.txt')
    with (workspace / ALTERNATIVE).open('xb') as stream:
        stream.write(NOTE)
    return workspace


def goal(workspace: Path) -> str:
    workspace = Path(workspace)
    if not workspace.is_absolute():
        raise ValueError('dojo_fixture_path_invalid')
    return (f'What was the incoming water pressure at the last inspection? '
            f'The report should be at {workspace / REPORT}. '
            f'If it is unavailable, look for another record in {workspace} and '
            'tell me the reading and its source. Please keep the records unchanged.')


def readback(workspace: Path) -> dict:
    workspace = Path(workspace)
    if not workspace.is_absolute() or workspace.is_symlink() or not workspace.is_dir():
        return {'status': 'unavailable'}
    try:
        note = workspace / ALTERNATIVE
        fd = os.open(note, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        try:
            data = os.read(fd, 16_385)
        finally:
            os.close(fd)
        if len(data) > 16_384:
            raise ValueError('fixture_too_large')
        report = workspace / REPORT
        return {'status': 'observed', 'note_sha256': hashlib.sha256(data).hexdigest(),
                'report_is_expected_dangling_link': report.is_symlink()
                and os.readlink(report) == 'missing-report.txt' and not report.exists()}
    except (OSError, ValueError):
        return {'status': 'unavailable'}


def grade(observation: dict, answer: str | None) -> dict:
    if not isinstance(observation, dict) or observation.get('status') != 'observed':
        return {'fixture_integrity': 'unavailable', 'response_semantics': 'unassessed'}
    unchanged = (observation.get('note_sha256') == hashlib.sha256(NOTE).hexdigest()
                 and observation.get('report_is_expected_dangling_link') is True)
    text = answer if isinstance(answer, str) else ''
    return {'fixture_integrity': 'unchanged' if unchanged else 'changed',
            'answer_mentions_expected_reading': bool(re.search(r'(?<!\d)53[.,]5(?!\d)', text)),
            'answer_mentions_expected_unit': bool(re.search(r'\bpsi\b', text, re.IGNORECASE)),
            'answer_mentions_record': any(word in text.casefold() for word in ('technician', 'technician-record')),
            'answer_mentions_unavailability': any(word in text.casefold() for word in
                                                 ('unavailable', 'missing', 'not found', 'could not', "couldn't", 'unable')),
            'response_semantics': 'requires_manual_review',
            'limitations': ['Mention checks do not establish grounding or semantic correctness.',
                            'Unchanged final fixtures do not prove absence of transient writes.']}
