"""Unregistered prototype: preserve a narrow legacy pending queue row.

No daemon, IPC or UI installs or invokes this module. Callers must authenticate
the active local owner, establish same-file store identity and exclude old
writers. A hold never cancels an external effect or proves none occurred.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import sqlite3
import time
from typing import Callable

_TABLE = '''CREATE TABLE IF NOT EXISTS v2_legacy_pending_holds (
    message_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, created_at TEXT NOT NULL
) STRICT'''
_TRIGGERS = {
    operation: f'''CREATE TRIGGER IF NOT EXISTS v2_legacy_pending_hold_prevent_{operation.lower()}
    BEFORE {operation} ON v2_inbound_messages
    WHEN EXISTS (SELECT 1 FROM v2_legacy_pending_holds WHERE message_id={row}.message_id)
    BEGIN SELECT RAISE(ABORT, 'legacy_pending_hold_requires_reconciliation'); END'''
    for operation, row in [('INSERT', 'NEW'), ('UPDATE', 'OLD'), ('DELETE', 'OLD')]
}


def _normalized(value: str) -> str:
    return ' '.join(value.replace('IF NOT EXISTS ', '').split()).casefold()


def _unique_object(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError('legacy_hold_unavailable')
        result[name] = value
    return result


def _objects(conn: sqlite3.Connection, *, install: bool) -> None:
    for kind, name, sql in [('table', 'v2_legacy_pending_holds', _TABLE), *[
            ('trigger', f'v2_legacy_pending_hold_prevent_{operation.lower()}', definition)
            for operation, definition in _TRIGGERS.items()]]:
        existing = conn.execute('SELECT type,sql FROM main.sqlite_master WHERE name=?', (name,)).fetchone()
        if existing is None and install:
            conn.execute(sql)
        elif existing is None or existing[0] != kind or not isinstance(existing[1], str) or _normalized(existing[1]) != _normalized(sql):
            raise ValueError('legacy_hold_schema_unavailable')


def _begin(conn: sqlite3.Connection) -> None:
    if not isinstance(conn, sqlite3.Connection) or conn.in_transaction:
        raise ValueError('legacy_hold_transaction_unavailable')
    databases = conn.execute('PRAGMA database_list').fetchall()
    if len(databases) != 1 or databases[0][1] != 'main' or not databases[0][2]:
        raise ValueError('legacy_hold_database_unavailable')
    conn.execute('BEGIN IMMEDIATE')
    deadline = time.monotonic() + .75
    calls = 0
    def progress():
        nonlocal calls
        calls += 1
        return int(calls > 20000 or time.monotonic() >= deadline)
    conn.set_progress_handler(progress, 100)


def install_legacy_hold_schema(conn: sqlite3.Connection) -> None:
    """Explicit disposable installer; callers must drain incompatible writers."""
    if not isinstance(conn, sqlite3.Connection) or conn.in_transaction:
        raise ValueError('legacy_hold_transaction_unavailable')
    try:
        _begin(conn)
        columns = {row[1] for row in conn.execute('PRAGMA main.table_info(v2_inbound_messages)')}
        if not {'message_id', 'user_id', 'status', 'claim_generation'} <= columns:
            raise ValueError('legacy_hold_schema_unavailable')
        _objects(conn, install=True)
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.set_progress_handler(None, 0)
            conn.rollback()
        raise
    finally:
        if isinstance(conn, sqlite3.Connection):
            conn.set_progress_handler(None, 0)


def _candidate(conn: sqlite3.Connection, owner: str, message_id: str) -> tuple:
    if (not isinstance(owner, str) or not owner.strip() or not isinstance(message_id, str)
            or not message_id or len(message_id.encode('utf-8')) > 256 or not message_id.isprintable()):
        raise ValueError('legacy_hold_unavailable')
    if conn.execute('''SELECT 1 FROM main.v2_users u JOIN main.v2_user_settings s
        ON s.key='admin_user_id' AND s.value=u.user_id
        WHERE u.user_id=? AND u.role IN ('owner','admin') AND u.is_active=1''', (owner,)).fetchone() is None:
        raise ValueError('legacy_hold_unavailable')
    # Refuse unfamiliar layouts and oversized text before fetching private data.
    names = [item[1] for item in conn.execute('PRAGMA main.table_info(v2_inbound_messages)')]
    expected_columns = {'message_id', 'sequence', 'timestamp', 'prompt', 'user_id', 'project_id', 'tag',
        'correlation_id', 'memory_session_id', 'metadata_json', 'queued_at', 'status', 'processing_at',
        'attempt_count', 'lease_owner', 'lease_expires_at', 'next_attempt_at', 'last_error', 'claim_generation'}
    if set(names) != expected_columns:
        raise ValueError('legacy_hold_schema_unavailable')
    text_columns = expected_columns - {'sequence', 'attempt_count'}
    bounds = ' AND '.join(f'length(CAST("{column}" AS BLOB))<=65536' for column in sorted(text_columns))
    row = conn.execute(f'''SELECT * FROM main.v2_inbound_messages WHERE message_id=? AND user_id=?
        AND {bounds} AND length(CAST(metadata_json AS BLOB))<=8192''', (message_id, owner)).fetchone()
    if row is None:
        raise ValueError('legacy_hold_unavailable')
    data = dict(zip(names, row))
    try:
        metadata = json.loads(data['metadata_json'], object_pairs_hook=_unique_object)
        engine = metadata['intelligence_engine']
        schema = metadata['intelligence_schema_version']
        if (set(metadata) != {'intelligence_engine', 'intelligence_schema_version'}
                or engine not in ('tactical_v2', 'hierarchical_v3')
                or type(schema) is not int or schema != {'tactical_v2': 2, 'hierarchical_v3': 3}[engine]
                or data['status'] != 'pending' or type(data['attempt_count']) is not int or data['attempt_count'] != 0
                or any(data[key] != '' for key in ('claim_generation', 'processing_at', 'lease_owner',
                    'lease_expires_at', 'next_attempt_at', 'last_error', 'correlation_id'))):
            raise ValueError()
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        raise ValueError('legacy_hold_unavailable') from None
    # Fail closed if a required schema is absent. All associations remain
    # unverified; absence is not proof that an external effect never occurred.
    checks = [
        ('v2_task_checkpoints', 'task_id'), ('v2_questions', 'task_id'),
        ('v2_outbox', 'task_id'), ('v2_scheduled_handoffs', 'message_id'),
        ('v2_scheduled_task_executions', 'queued_message_id'),
        ('v2_automation_executions', 'queued_message_id'),
        ('v2_trusted_owner_ingress', 'message_id'), ('v2_inbound_receipts', 'message_id'),
        ('v2_legacy_pending_holds', 'message_id'),
    ]
    for table, column in checks:
        if conn.execute(f'SELECT 1 FROM main.{table} WHERE {column}=? LIMIT 1', (message_id,)).fetchone():
            raise ValueError('legacy_hold_unavailable')
    # Older processors could save a distinct task_id while retaining the
    # inbound identifier inside checkpoint JSON. Unknown schema, duplicate
    # keys, excessive size or excessive row count make the proof unavailable.
    checkpoints = conn.execute('''SELECT CASE WHEN length(CAST(task_state_json AS BLOB))<=65536
        THEN task_state_json ELSE NULL END FROM main.v2_task_checkpoints LIMIT 1001''')
    for count, checkpoint in enumerate(checkpoints, 1):
        if count > 1000 or not isinstance(checkpoint[0], str):
            raise ValueError('legacy_hold_unavailable')
        try:
            saved = json.loads(checkpoint[0], object_pairs_hook=_unique_object)
            if (not isinstance(saved, dict) or
                    (saved.get('message_id') is not None and not isinstance(saved['message_id'], str)) or
                    saved.get('message_id') == message_id):
                raise ValueError('legacy_hold_unavailable')
        except (ValueError, TypeError, RecursionError):
            raise ValueError('legacy_hold_unavailable') from None
    if conn.execute('''SELECT 1 FROM main.v2_outbox WHERE correlation_id=? LIMIT 1''', (message_id,)).fetchone():
        raise ValueError('legacy_hold_unavailable')
    if conn.execute('''SELECT 1 FROM main.v2_task_dependencies
        WHERE parent_task_id=? OR child_task_id=? OR question_id=? LIMIT 1''', (message_id, message_id, message_id)).fetchone():
        raise ValueError('legacy_hold_unavailable')
    return tuple(row)


def _token(key: bytes, owner: str, message_id: str, row: tuple) -> str:
    if type(key) is not bytes or len(key) != 32:
        raise ValueError('legacy_hold_key_invalid')
    body = json.dumps([owner, message_id, row], ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    return hmac.new(key, b'alphonse-legacy-hold-v1\0' + body, hashlib.sha256).hexdigest()


def preview_legacy_hold(conn: sqlite3.Connection, *, owner: str, message_id: str, key: bytes) -> str:
    """Return an opaque, session-bound token; it is not a permission grant."""
    if not isinstance(conn, sqlite3.Connection) or conn.in_transaction:
        raise ValueError('legacy_hold_transaction_unavailable')
    try:
        _begin(conn)
        _objects(conn, install=False)
        token = _token(key, owner, message_id, _candidate(conn, owner, message_id))
        conn.rollback()
        return token
    except BaseException:
        if conn.in_transaction:
            conn.set_progress_handler(None, 0)
            conn.rollback()
        raise
    finally:
        if isinstance(conn, sqlite3.Connection):
            conn.set_progress_handler(None, 0)


def hold_legacy_pending(conn: sqlite3.Connection, *, owner: str, message_id: str, key: bytes,
                        token: str, clock: Callable[[], datetime]) -> str:
    """Protect this row against queue writes without retry, delete or completion."""
    if not isinstance(conn, sqlite3.Connection) or conn.in_transaction:
        raise ValueError('legacy_hold_transaction_unavailable')
    try:
        _begin(conn)
        _objects(conn, install=False)
        expected = _token(key, owner, message_id, _candidate(conn, owner, message_id))
        if not isinstance(token, str) or not hmac.compare_digest(expected, token):
            raise ValueError('legacy_hold_stale')
        stamp = clock()
        if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
            raise ValueError('legacy_hold_clock_invalid')
        conn.execute('INSERT INTO main.v2_legacy_pending_holds VALUES (?,?,?)',
                     (message_id, owner, stamp.astimezone(timezone.utc).isoformat()))
        conn.commit()
        return 'held'
    except BaseException:
        if conn.in_transaction:
            conn.set_progress_handler(None, 0)
            conn.rollback()
        raise
    finally:
        if isinstance(conn, sqlite3.Connection):
            conn.set_progress_handler(None, 0)
