"""Experimental, unwired append-only attempt ledger for owner tasks.

Callers own the SQLite transaction and must authenticate the owner and task.
Beginning an attempt is not proof it was dispatched; a missing outcome remains
unknown. Entries alone cannot prove completeness: production must atomically
integrate *every* action path and provider-side operation before claiming it.
The hash chain detects accidental corruption, not a malicious database owner
or an undetectable truncated suffix; it is not an authentication boundary.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any


_GENESIS = '0' * 64
_OUTCOMES = {'success', 'failed', 'not_dispatched'}


class AttemptLedgerError(ValueError):
    pass


def _name(value: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise AttemptLedgerError('attempt_ledger_invalid_name')
    return value


def _canonical(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
    except (TypeError, ValueError):
        raise AttemptLedgerError('attempt_ledger_invalid_payload') from None


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def install(conn: sqlite3.Connection) -> None:
    """Install the ledger in the caller's transaction; never commits it."""
    conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_attempt_ledger (
        owner_id TEXT NOT NULL, task_id TEXT NOT NULL, sequence INTEGER NOT NULL,
        event_json TEXT NOT NULL, previous_hash TEXT NOT NULL, event_hash TEXT NOT NULL,
        PRIMARY KEY(owner_id, task_id, sequence)
    )''')
    conn.execute('''CREATE TRIGGER IF NOT EXISTS v2_owner_attempt_ledger_no_update
        BEFORE UPDATE ON v2_owner_attempt_ledger BEGIN
        SELECT RAISE(ABORT, 'attempt_ledger_append_only'); END''')
    conn.execute('''CREATE TRIGGER IF NOT EXISTS v2_owner_attempt_ledger_no_delete
        BEFORE DELETE ON v2_owner_attempt_ledger BEGIN
        SELECT RAISE(ABORT, 'attempt_ledger_append_only'); END''')


def _digest(owner_id: str, task_id: str, sequence: int, event: dict, previous_hash: str) -> str:
    return _hash(_canonical({'owner_id': owner_id, 'task_id': task_id, 'sequence': sequence,
                             'event': event, 'previous_hash': previous_hash}))


def read_verified(conn: sqlite3.Connection, *, owner_id: str, task_id: str) -> list[dict[str, Any]]:
    """Return verified events or fail closed on malformed and inconsistent rows."""
    _name(owner_id)
    _name(task_id)
    rows = conn.execute('''SELECT sequence, event_json, previous_hash, event_hash
        FROM v2_owner_attempt_ledger WHERE owner_id=? AND task_id=? ORDER BY sequence''',
        (owner_id, task_id)).fetchall()
    previous = _GENESIS
    started: set[str] = set()
    resolved: set[str] = set()
    events = []
    for sequence, serialized, stored_previous, digest in rows:
        try:
            event = json.loads(serialized)
            if not isinstance(event, dict) or _canonical(event) != serialized:
                raise AttemptLedgerError('attempt_ledger_corrupt')
            attempt_id = _name(event.get('attempt_id'))
            if event.get('type') == 'start':
                if (set(event) != {'type', 'attempt_id', 'tool_id', 'arguments_sha256'}
                        or attempt_id in started):
                    raise AttemptLedgerError('attempt_ledger_corrupt')
                _name(event['tool_id'])
                argument_hash = event['arguments_sha256']
                if (not isinstance(argument_hash, str) or len(argument_hash) != 64
                        or any(x not in '0123456789abcdef' for x in argument_hash)):
                    raise AttemptLedgerError('attempt_ledger_corrupt')
                started.add(attempt_id)
            elif event.get('type') == 'outcome':
                if (set(event) != {'type', 'attempt_id', 'outcome'} or attempt_id not in started
                        or attempt_id in resolved or event.get('outcome') not in _OUTCOMES):
                    raise AttemptLedgerError('attempt_ledger_corrupt')
                resolved.add(attempt_id)
            else:
                raise AttemptLedgerError('attempt_ledger_corrupt')
            if (sequence != len(events) + 1 or stored_previous != previous
                    or digest != _digest(owner_id, task_id, sequence, event, previous)):
                raise AttemptLedgerError('attempt_ledger_corrupt')
        except (TypeError, KeyError, json.JSONDecodeError):
            raise AttemptLedgerError('attempt_ledger_corrupt') from None
        previous = digest
        events.append(event)
    return events


def _append(conn: sqlite3.Connection, owner_id: str, task_id: str, event: dict) -> None:
    events = read_verified(conn, owner_id=owner_id, task_id=task_id)
    rows = conn.execute('''SELECT event_hash FROM v2_owner_attempt_ledger
        WHERE owner_id=? AND task_id=? ORDER BY sequence DESC LIMIT 1''', (owner_id, task_id)).fetchone()
    previous = rows[0] if rows else _GENESIS
    sequence = len(events) + 1
    conn.execute('''INSERT INTO v2_owner_attempt_ledger
        (owner_id, task_id, sequence, event_json, previous_hash, event_hash) VALUES (?,?,?,?,?,?)''',
        (owner_id, task_id, sequence, _canonical(event), previous,
         _digest(owner_id, task_id, sequence, event, previous)))


def begin(conn: sqlite3.Connection, *, owner_id: str, task_id: str, attempt_id: str,
          tool_id: str, arguments: dict[str, Any]) -> None:
    """Append a durable pre-dispatch marker; caller commits before dispatch."""
    if not conn.in_transaction:
        raise AttemptLedgerError('attempt_ledger_transaction_required')
    _name(owner_id)
    _name(task_id)
    _name(attempt_id)
    _name(tool_id)
    if not isinstance(arguments, dict):
        raise AttemptLedgerError('attempt_ledger_invalid_arguments')
    if any(event['attempt_id'] == attempt_id for event in read_verified(conn, owner_id=owner_id, task_id=task_id)):
        raise AttemptLedgerError('attempt_ledger_duplicate_attempt')
    _append(conn, owner_id, task_id, {'type': 'start', 'attempt_id': attempt_id,
                                     'tool_id': tool_id, 'arguments_sha256': _hash(_canonical(arguments))})


def finish(conn: sqlite3.Connection, *, owner_id: str, task_id: str, attempt_id: str,
           outcome: str) -> None:
    """Record a caller-verified outcome once; unconfirmed effects stay unresolved.

    The host must independently prove ``not_dispatched`` before recording it.
    """
    if not conn.in_transaction:
        raise AttemptLedgerError('attempt_ledger_transaction_required')
    _name(attempt_id)
    if outcome not in _OUTCOMES:
        raise AttemptLedgerError('attempt_ledger_invalid_outcome')
    events = read_verified(conn, owner_id=owner_id, task_id=task_id)
    matching = [event for event in events if event['attempt_id'] == attempt_id]
    if len(matching) != 1 or matching[0]['type'] != 'start':
        raise AttemptLedgerError('attempt_ledger_outcome_unavailable')
    _append(conn, owner_id, task_id, {'type': 'outcome', 'attempt_id': attempt_id, 'outcome': outcome})


def unresolved_attempts(events: list[dict[str, Any]]) -> list[str]:
    """Derive unknown outcomes from a previously verified event sequence."""
    started = [event['attempt_id'] for event in events if event['type'] == 'start']
    finished = {event['attempt_id'] for event in events if event['type'] == 'outcome'}
    return [attempt for attempt in started if attempt not in finished]
