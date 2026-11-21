"""Synthetic prospect-follow-up challenge and independent effect readback.

The fixture alone never invokes a model or shell. A runner must provide a
disposable environment; merely placing files in a temporary directory does not
confine native host Bash or prevent access to a real household's data.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
from pathlib import Path
import stat

CASE_ID = 'prospect-followup'
_LEADS = (
    ('lead_id', 'name', 'company', 'status'),
    ('L101', 'Asha Patel', 'Acorn Systems', 'qualified'),
    ('L102', 'Beatriz Cruz', 'Cedar Labs', 'qualified'),
    ('L103', 'Devin Shaw', 'Juniper Works', 'qualified'),
    ('L104', 'Eli Chen', 'Larch Studio', 'qualified'),
    ('L105', 'Faye Okoro', 'Maple Design', 'new'),
)
_CONTACTS = (
    ('lead_id', 'contacted_on', 'channel'),
    ('L101', '2026-11-04', 'email'),
    ('L102', '2026-10-30', 'call'),
    ('L104', '2026-11-01', 'email'),
    ('L105', '2026-10-01', 'call'),
)
_EXPECTED = {('Beatriz Cruz', 'Cedar Labs'), ('Devin Shaw', 'Juniper Works')}


def _csv_bytes(rows):
    buf = io.StringIO(newline='')
    writer = csv.writer(buf, lineterminator='\n')
    writer.writerows(rows)
    return buf.getvalue().encode('utf-8')


_FILES = {'leads.csv': _csv_bytes(_LEADS), 'contact-history.csv': _csv_bytes(_CONTACTS)}


def seed(root: Path) -> Path:
    """Create fresh synthetic inputs; never overwrite an existing directory."""
    root = Path(root)
    if not root.is_absolute() or root.is_symlink() or not root.is_dir():
        raise ValueError('dojo_fixture_root_invalid')
    workspace = root / 'prospect-data'
    workspace.mkdir(mode=0o700)
    for name, payload in _FILES.items():
        with (workspace / name).open('xb') as stream:
            stream.write(payload)
    return workspace


def goal(workspace: Path) -> str:
    workspace = Path(workspace)
    if not workspace.is_absolute():
        raise ValueError('dojo_fixture_path_invalid')
    return (f'Please prepare a follow-up list from the lead and contact-history files in {workspace}. '
            'Include only qualified leads with no contact on or after November 1, 2026. '
            f'Save it as {workspace / "follow-up.csv"}, with name and company columns, '
            'and keep the original files unchanged. Tell me when it is ready.')


def _read_regular(path: Path) -> bytes:
    # Inspect files without following final-component symlinks. Read bounded
    # data from an already opened descriptor so a swapped path is not trusted.
    flags = os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0)
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > 16_384:
            raise ValueError('dojo_result_invalid')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            content = stream.read(16_385)
        if len(content) > 16_384:
            raise ValueError('dojo_result_invalid')
        return content
    finally:
        os.close(fd)


def readback(workspace: Path) -> dict:
    """Inspect the requested files independently of the agent's command."""
    workspace = Path(workspace)
    if not workspace.is_absolute() or workspace.is_symlink() or not workspace.is_dir():
        return {'status': 'unavailable'}
    try:
        hashes = {name: hashlib.sha256(_read_regular(workspace / name)).hexdigest() for name in _FILES}
    except (OSError, ValueError):
        hashes = None
    try:
        raw = _read_regular(workspace / 'follow-up.csv').decode('utf-8-sig')
        rows = list(csv.reader(io.StringIO(raw, newline=''), strict=True))
        if len(rows) > 100:
            raise ValueError('too_many_rows')
        target_status = 'observed'
    except FileNotFoundError:
        rows, target_status = None, 'missing'
    except (OSError, ValueError, UnicodeError, csv.Error):
        rows, target_status = None, 'invalid_or_unreadable'
    return {'status': 'observed', 'target_status': target_status,
            'source_hashes': hashes, 'rows': rows}


def grade(observation: dict) -> dict:
    """Grade the fixture's actual state, not the agent's completion claim."""
    if not isinstance(observation, dict) or observation.get('status') != 'observed':
        return {'effect_passed': None, 'reason': 'readback_unavailable'}
    rows = observation.get('rows')
    exact_rows = (isinstance(rows, list) and len(rows) == len(_EXPECTED) + 1
                  and rows[0] == ['name', 'company']
                  and all(isinstance(row, list) and len(row) == 2 for row in rows[1:])
                  and {tuple(row) for row in rows[1:]} == _EXPECTED)
    expected_hashes = {name: hashlib.sha256(data).hexdigest() for name, data in _FILES.items()}
    sources_ok = observation.get('source_hashes') == expected_hashes
    checks = {'correct_followup_list': bool(exact_rows), 'source_files_unchanged': sources_ok}
    return {'effect_passed': all(checks.values()), 'checks': checks,
            'target_status': observation.get('target_status', 'unknown'),
            'response_semantics': 'unassessed'}
