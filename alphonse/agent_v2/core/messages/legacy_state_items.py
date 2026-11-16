"""Bounded, owner-scoped metadata for preserved work; never a recovery action."""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path

_LIMIT = 50
_ENGINES = {'tactical_v2', 'hierarchical_v3', 'single_loop_experimental'}
_STATUSES = {
    'queue': {'pending', 'processing', 'retry_wait', 'failed'},
    'checkpoint': {'queued', 'running', 'waiting_user', 'done', 'failed', 'cancelled'},
}
_OUTBOX_STATUSES = {'pending', 'claimed', 'retry_wait', 'delivered', 'failed', 'unconfirmed'}


def inspect_legacy_items(db_path: str | Path, *, expected_owner: str) -> dict | None:
    """Read bounded identifiers/statuses under the canonical owner transaction.

    This does not authenticate the caller or prove an effect, delivery, origin
    marker, or permission to replay. Callers must authenticate the local peer
    and require the same authoritative database as the running queue and stores.
    Request text and checkpoint/effect contents are never returned.
    """
    if (not isinstance(expected_owner, str) or not expected_owner.strip()
            or not isinstance(db_path, (str, Path)) or str(db_path) in ('', ':memory:')):
        return None
    try:
        path = Path(db_path).expanduser().resolve(strict=True)
        if not path.is_file():
            return None
        conn = sqlite3.connect(path.as_uri() + '?mode=rw', uri=True, timeout=1)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute('BEGIN IMMEDIATE')
            conn.execute('PRAGMA query_only=ON')
            deadline = time.monotonic() + .75
            calls = 0
            def progress():
                nonlocal calls
                calls += 1
                return int(calls > 20000 or time.monotonic() >= deadline)
            conn.set_progress_handler(progress, 100)
            owners = conn.execute('''SELECT u.user_id FROM v2_users u
                JOIN v2_user_settings s ON s.key='admin_user_id' AND s.value=u.user_id
                WHERE u.role IN ('owner','admin') AND u.is_active=1''').fetchall()
            if len(owners) != 1 or owners[0][0] != expected_owner:
                return None
            has_origins = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_trusted_owner_ingress'").fetchone() is not None
            def group(kind, table, alias, identifier, owner, payload):
                marker = (f"CASE WHEN EXISTS (SELECT 1 FROM v2_trusted_owner_ingress o WHERE o.message_id={alias}.{identifier} AND o.owner_id={alias}.{owner}) THEN 'present' ELSE 'absent' END"
                          if has_origins else "'unavailable'")
                # Only a fixed engine category leaves SQL; bound the payload
                # before JSON parsing and never fetch it into application code.
                engine = f"""CASE WHEN length(CAST({alias}.{payload} AS BLOB)) > 65536 OR NOT json_valid({alias}.{payload}) THEN 'malformed'
                    WHEN json_type({alias}.{payload}, '$.intelligence_engine') IS NULL OR json_type({alias}.{payload}, '$.intelligence_engine') = 'null' THEN 'missing'
                    WHEN json_extract({alias}.{payload}, '$.intelligence_engine') IN ('tactical_v2','hierarchical_v3','single_loop_experimental') THEN json_extract({alias}.{payload}, '$.intelligence_engine')
                    ELSE 'other' END"""
                rows = conn.execute(f"""SELECT CASE WHEN length(CAST({alias}.{identifier} AS BLOB)) <= 256 THEN {alias}.{identifier} END AS identifier,
                    {alias}.status AS status, {engine} AS engine, {marker} AS marker
                    FROM {table} {alias} WHERE {alias}.{owner}=? ORDER BY {alias}.{identifier} LIMIT ?""",
                    (expected_owner, _LIMIT + 1)).fetchall()
                items = []
                for row in rows[:_LIMIT]:
                    identifier_value = row['identifier']
                    if not isinstance(identifier_value, str) or not identifier_value or not identifier_value.isprintable():
                        raise ValueError('legacy_item_identifier_invalid')
                    status = row['status'] if row['status'] in _STATUSES[kind] else 'other'
                    category = row['engine'] if row['engine'] in _ENGINES | {'missing', 'malformed'} else 'other'
                    items.append({'id': identifier_value, 'status': status, 'engine': category, 'origin_marker': row['marker']})
                return {'items': items, 'truncated': len(rows) > _LIMIT}
            outbox_rows = conn.execute('''SELECT
                CASE WHEN length(CAST(o.outbox_message_id AS BLOB)) <= 256 THEN o.outbox_message_id END AS identifier,
                o.status, CASE WHEN typeof(o.attempt_count)!='integer' OR o.attempt_count<0 THEN 'invalid'
                    WHEN o.attempt_count=0 THEN 'zero' ELSE 'positive' END AS attempts,
                CASE WHEN length(CAST(o.task_id AS BLOB)) <= 256 AND EXISTS
                    (SELECT 1 FROM v2_task_checkpoints c WHERE c.task_id=o.task_id AND c.owner_id=?)
                    THEN o.task_id ELSE NULL END AS owner_task
                FROM v2_outbox o WHERE o.audience_user_id=? ORDER BY o.outbox_message_id LIMIT ?''',
                (expected_owner, expected_owner, _LIMIT + 1)).fetchall()
            outbox = []
            for row in outbox_rows[:_LIMIT]:
                identity, task = row['identifier'], row['owner_task']
                if not isinstance(identity, str) or not identity or not identity.isprintable():
                    raise ValueError('legacy_item_identifier_invalid')
                if task is not None and (not isinstance(task, str) or not task or not task.isprintable()):
                    raise ValueError('legacy_item_identifier_invalid')
                outbox.append({'id': identity, 'status': row['status'] if row['status'] in _OUTBOX_STATUSES else 'other',
                               'attempts': row['attempts'], 'owner_task_id': task})
            return {'queue': group('queue', 'v2_inbound_messages', 'q', 'message_id', 'user_id', 'metadata_json'),
                    'checkpoints': group('checkpoint', 'v2_task_checkpoints', 'c', 'task_id', 'owner_id', 'task_state_json'),
                    'outbox': {'items': outbox, 'truncated': len(outbox_rows) > _LIMIT}}
        finally:
            conn.close()
    except (sqlite3.Error, OSError, ValueError, TypeError, OverflowError):
        return None
