"""Read-only, owner-scoped counts for work the exclusive owner loop pauses.

This is a point-in-time diagnostic, not a recovery decision or effect receipt.
It returns no request contents, row identifiers, or other members' records.
The IPC boundary must independently authenticate the local OS peer and bind
all runtime stores to this same database.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any


_QUEUE_STATUSES = ('pending', 'processing', 'retry_wait', 'failed')
_CHECKPOINT_STATUSES = ('queued', 'running', 'waiting_user', 'done', 'failed', 'cancelled')
_OUTBOX_STATUSES = ('pending', 'claimed', 'retry_wait', 'delivered', 'failed', 'unconfirmed')
_SCHEDULE_STATUSES = ('active', 'paused', 'completed', 'cancelled', 'failed')
_OCCURRENCE_STATUSES = ('pending', 'claimed', 'enqueued', 'processing', 'response_pending', 'retry_wait',
                        'delivered', 'failed', 'delivery_failed', 'unconfirmed', 'cancelled', 'completed')
_ENGINES = ('tactical_v2', 'hierarchical_v3', 'single_loop_experimental')
_MAX_VM_CALLBACKS = 20_000  # With a 100-op callback interval: 2 million steps.
_MAX_QUERY_SECONDS = 0.75


def _counts(rows: list[sqlite3.Row], known: tuple[str, ...], column: str) -> dict[str, int]:
    counts = {value: 0 for value in known}
    counts['other'] = 0
    for row in rows:
        value = row[column]
        counts[value if isinstance(value, str) and value in known else 'other'] += int(row['n'])
    return counts


def _statuses(connection: sqlite3.Connection, sql: str, parameters: tuple[Any, ...],
              known: tuple[str, ...]) -> dict[str, int]:
    # Collapse unknown stored status values before fetching results; a damaged
    # schema cannot create unbounded result categories or expose raw labels.
    placeholders = ','.join('?' for _ in known)
    rows = connection.execute(f'''WITH source AS ({sql}), classified AS (
        SELECT CASE WHEN status IN ({placeholders}) THEN status ELSE 'other' END AS category FROM source)
        SELECT category, COUNT(*) AS n FROM classified GROUP BY category''', (*parameters, *known)).fetchall()
    return _counts(rows, known, 'category')


def _engines(connection: sqlite3.Connection, sql: str, parameters: tuple[Any, ...], field: str) -> dict[str, int]:
    # The fixed categories avoid returning an arbitrary value from stored JSON.
    # JSON1 validity/type checks distinguish malformed and missing stamps.
    rows = connection.execute(f'''WITH source AS ({sql}), classified AS (
        SELECT CASE WHEN json_valid(body) = 0 THEN 'malformed'
          WHEN json_type(body, ?) IS NULL OR json_type(body, ?) = 'null' THEN 'missing'
          WHEN json_type(body, ?) != 'text' THEN 'other'
          WHEN json_extract(body, ?) = '' THEN 'missing'
          WHEN json_extract(body, ?) IN ('tactical_v2','hierarchical_v3','single_loop_experimental')
            THEN json_extract(body, ?)
          ELSE 'other' END AS engine FROM source)
        SELECT engine,COUNT(*) AS n FROM classified GROUP BY engine''',
        (*parameters, *(field for _ in range(6)))).fetchall()
    return _counts(rows, (*_ENGINES, 'malformed', 'missing'), 'engine')


def inspect_legacy_state(db_path: str | Path, *, expected_owner: str | None = None,
                         include_schedules: bool = False) -> dict[str, Any] | None:
    """Inspect a file-backed database without changing queue or delivery state.

    The canonical active owner is determined and checked in the same database
    snapshot. `expected_owner`, when supplied by trusted host code, must match.
    `None` means unauthorized, missing schema, invalid input, or unavailable;
    unknown state is never reported as an empty inventory. This function does
    not authenticate an OS peer and must not be exposed as an unauthenticated
    endpoint. An immediate transaction reserves the same database while owner
    authorization and counts are read; query-only mode disallows subsequent
    SQL writes. Busy wait and query work have cooperative budgets; an individual
    SQLite or OS operation can exceed the time target. Contention or budget
    exhaustion returns unavailable. This does not modify application state or
    run schema migrations, and the reservation is released before IPC response.
    The trusted caller may request schedules only after verifying the runtime
    schedule store uses this database. Missing schedule schema is unavailable.
    """
    if expected_owner is not None and (not isinstance(expected_owner, str) or not expected_owner.strip()):
        return None
    if not isinstance(db_path, (str, Path)) or str(db_path) in ('', ':memory:'):
        return None
    try:
        path = Path(db_path).expanduser().resolve(strict=True)
        if not path.is_file():
            return None
        connection = sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, timeout=1)
        connection.row_factory = sqlite3.Row
        try:
            connection.execute('BEGIN IMMEDIATE')
            connection.execute('PRAGMA query_only=ON')
            deadline = time.monotonic() + _MAX_QUERY_SECONDS
            callbacks = 0
            def progress() -> int:
                nonlocal callbacks
                callbacks += 1
                return int(callbacks > _MAX_VM_CALLBACKS or time.monotonic() >= deadline)
            connection.set_progress_handler(progress, 100)
            owner_rows = connection.execute('''SELECT u.user_id FROM v2_users u
                JOIN v2_user_settings s ON s.key='admin_user_id' AND s.value=u.user_id
                WHERE u.role IN ('owner','admin') AND u.is_active=1''').fetchall()
            if len(owner_rows) != 1:
                return None
            owner = owner_rows[0]['user_id']
            if not isinstance(owner, str) or not owner.strip() or (expected_owner is not None and owner != expected_owner):
                return None
            origin_exists = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_trusted_owner_ingress'").fetchone() is not None
            # A matching origin row is not proof of trust: it can be malformed
            # or have invalid peer, digest or history binding. Report both
            # marker-present and other categories so neither silently vanishes.
            queue_mark = ("EXISTS (SELECT 1 FROM v2_trusted_owner_ingress o "
                          "WHERE o.message_id=q.message_id AND o.owner_id=q.user_id)" if origin_exists else '0')
            checkpoint_mark = ("EXISTS (SELECT 1 FROM v2_trusted_owner_ingress o "
                               "WHERE o.message_id=c.task_id AND o.owner_id=c.owner_id)" if origin_exists else '0')
            def grouped(sql: str, known: tuple[str, ...]) -> dict[str, dict[str, int]]:
                return {'statuses': _statuses(connection, sql, (owner,), known),
                        'engine_stamps': _engines(connection, sql, (owner,), '$.intelligence_engine')}
            queue_source = 'SELECT q.status,q.metadata_json AS body FROM v2_inbound_messages q WHERE q.user_id=?'
            checkpoint_source = 'SELECT c.status,c.task_state_json AS body FROM v2_task_checkpoints c WHERE c.owner_id=?'
            queue_other = grouped(f'{queue_source} AND NOT ({queue_mark})', _QUEUE_STATUSES)
            queue_marked = grouped(f'{queue_source} AND ({queue_mark})', _QUEUE_STATUSES)
            checkpoints_other = grouped(f'{checkpoint_source} AND NOT ({checkpoint_mark})', _CHECKPOINT_STATUSES)
            checkpoints_marked = grouped(f'{checkpoint_source} AND ({checkpoint_mark})', _CHECKPOINT_STATUSES)
            outbox = _statuses(connection, 'SELECT status FROM v2_outbox WHERE audience_user_id=?',
                               (owner,), _OUTBOX_STATUSES)
            counter = connection.execute('''SELECT CASE
                WHEN typeof(attempt_count)!='integer' OR attempt_count<0 THEN 'invalid'
                WHEN attempt_count=0 THEN 'zero' ELSE 'positive' END AS counter, COUNT(*) AS n
                FROM v2_outbox WHERE audience_user_id=? GROUP BY counter''', (owner,)).fetchall()
            scheduled: dict[str, Any] = {'status': 'unavailable'}
            if include_schedules is True:
                try:
                    tasks = _statuses(connection, 'SELECT status FROM v2_scheduled_tasks WHERE owner_user_id=?',
                                      (owner,), _SCHEDULE_STATUSES)
                    occurrences = _statuses(connection, '''SELECT e.status FROM v2_scheduled_task_executions e
                        JOIN v2_scheduled_tasks t ON t.scheduled_task_id=e.scheduled_task_id
                        WHERE t.owner_user_id=?''', (owner,), _OCCURRENCE_STATUSES)
                    scheduled = {'status': 'available', 'tasks': tasks, 'occurrences': occurrences}
                except sqlite3.Error:
                    # Never present an absent or incompatible schedule schema as zero.
                    scheduled = {'status': 'unavailable'}
            return {
                'queue_other_ingress': queue_other,
                'queue_owner_origin_marked': queue_marked,
                'checkpoints_other_ingress': checkpoints_other,
                'checkpoints_owner_origin_marked': checkpoints_marked,
                'owner_addressed_outbox': {'statuses': outbox,
                    'attempt_counter': _counts(counter, ('zero', 'positive', 'invalid'), 'counter')},
                'owner_scheduled_work': scheduled,
            }
        finally:
            connection.close()
    except (sqlite3.Error, OSError, ValueError, TypeError, OverflowError):
        return None
