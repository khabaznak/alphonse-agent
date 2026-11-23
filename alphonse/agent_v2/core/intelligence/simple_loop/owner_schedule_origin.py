"""Provenance for owner-created schedules, distinct from local-owner ingress.

These helpers use the caller's transaction. They do not dispatch work or grant
authority to an occurrence; an occurrence needs a separate claim and fence.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from datetime import datetime, timezone
from uuid import uuid4


_FIELDS = ('scheduled_task_id', 'owner_user_id', 'project_id', 'name', 'description', 'prompt',
           'schedule_json', 'origin_channel_json', 'timezone', 'delivery_mode', 'idempotency_key',
           'job_kind', 'job_version', 'payload_json', 'created_at')


def _digest(row):
    content = [row[field] for field in _FIELDS]
    return hashlib.sha256(json.dumps(content, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def ensure_schedule_origin_schema(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_schedule_origins (
        scheduled_task_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
        request_message_id TEXT NOT NULL UNIQUE, request_payload_sha256 TEXT NOT NULL,
        schedule_sha256 TEXT NOT NULL) STRICT''')
    conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_schedule_occurrences (
        schedule_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
        message_id TEXT NOT NULL UNIQUE, due_at TEXT NOT NULL,
        payload_sha256 TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('queued','completed','cancelled','failed'))) STRICT''')
    conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_schedule_telegram_origins (
        scheduled_task_id TEXT PRIMARY KEY, request_message_id TEXT NOT NULL UNIQUE) STRICT''')


def create_schedule_origin_locked(conn, task, schedule_record):
    """Bind an exact newly inserted schedule to the current owner request.

    Caller must already hold the verified owner claim transaction. All writes
    roll back with it. The function never authenticates a standalone caller.
    """
    if not conn.in_transaction:
        raise ValueError('owner_schedule_origin_transaction_required')
    ensure_schedule_origin_schema(conn)
    schedule_id = getattr(schedule_record, 'scheduled_task_id', None)
    source_id = getattr(task, 'message_id', None)
    owner = getattr(task, 'user', None)
    row = conn.execute('SELECT * FROM v2_scheduled_tasks WHERE scheduled_task_id=?', (schedule_id,)).fetchone()
    local_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_trusted_owner_ingress'").fetchone()
    source = conn.execute('''SELECT o.owner_id,o.peer_uid,o.payload_sha256,q.prompt
        FROM v2_trusted_owner_ingress o JOIN v2_inbound_messages q ON q.message_id=o.message_id
        JOIN v2_user_settings s ON s.key='admin_user_id' AND s.value=o.owner_id
        JOIN v2_users u ON u.user_id=o.owner_id AND u.role IN ('admin','owner') AND u.is_active=1
        WHERE o.message_id=?''', (source_id,)).fetchone() if local_table else None
    telegram = None
    if source is None:
        from .owner_telegram_ingress import verified_telegram_origin_locked
        telegram = verified_telegram_origin_locked(conn, source_id)
        if telegram is not None:
            source = conn.execute('''SELECT o.owner_id,o.payload_sha256,q.prompt
                FROM v2_owner_telegram_ingress o JOIN v2_inbound_messages q ON q.message_id=o.message_id
                WHERE o.message_id=?''', (source_id,)).fetchone()
    if (row is None or source is None or (telegram is None and source['peer_uid'] != os.getuid())
            or not isinstance(owner, str) or source['owner_id'] != owner or row['owner_user_id'] != owner
            or row['idempotency_key'] != 'owner-schedule:' + source_id
            or row['prompt'] != source['prompt'] or row['prompt'] != getattr(task, 'goal', None)
            or row['status'] != 'active' or row['job_kind'] != 'prompt' or row['job_version'] != 1
            or row['delivery_mode'] != 'pdca' or json.loads(row['schedule_json']).get('kind') != 'once'):
        raise ValueError('owner_schedule_origin_invalid')
    conn.execute('''INSERT INTO v2_owner_schedule_origins
        (scheduled_task_id,owner_id,request_message_id,request_payload_sha256,schedule_sha256)
        VALUES (?,?,?,?,?)''', (schedule_id, owner, source_id, source['payload_sha256'], _digest(row)))
    if telegram is not None:
        conn.execute('INSERT INTO v2_owner_schedule_telegram_origins VALUES (?,?)', (schedule_id, source_id))


def verified_schedule_origin_locked(conn, schedule_id):
    """Return the current owner-created schedule if provenance remains intact.

    This is a read under the caller's transaction, not authorization to run it.
    """
    telegram_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_schedule_telegram_origins'").fetchone()
    telegram = conn.execute('SELECT request_message_id FROM v2_owner_schedule_telegram_origins WHERE scheduled_task_id=?',
                            (schedule_id,)).fetchone() if telegram_table else None
    if telegram is not None:
        row = conn.execute('''SELECT t.*,o.owner_id AS origin_owner,o.request_message_id,
            o.request_payload_sha256,o.schedule_sha256,i.owner_id AS request_owner,
            i.payload_sha256 AS current_request_sha256
            FROM v2_owner_schedule_origins o JOIN v2_scheduled_tasks t ON t.scheduled_task_id=o.scheduled_task_id
            JOIN v2_owner_telegram_ingress i ON i.message_id=o.request_message_id
            WHERE o.scheduled_task_id=?''', (schedule_id,)).fetchone()
        if (row is None or telegram['request_message_id'] != row['request_message_id']
                or row['origin_owner'] != row['owner_user_id'] or row['request_owner'] != row['origin_owner']
                or row['request_payload_sha256'] != row['current_request_sha256'] or row['schedule_sha256'] != _digest(row)):
            return None
        from .owner_telegram_ingress import verified_telegram_origin_locked
        if verified_telegram_origin_locked(conn, row['request_message_id']) is None:
            from .owner_telegram_handoff import verify_finalized_telegram_reply_locked
            if not verify_finalized_telegram_reply_locked(conn, row['request_message_id']):
                return None
        return row
    local_table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_trusted_owner_ingress'").fetchone()
    if not local_table:
        return None
    row = conn.execute('''SELECT t.*,o.owner_id AS origin_owner,
        o.request_message_id,o.request_payload_sha256,o.schedule_sha256,
        i.owner_id AS request_owner,i.peer_uid,i.payload_sha256 AS current_request_sha256
        FROM v2_owner_schedule_origins o JOIN v2_scheduled_tasks t ON t.scheduled_task_id=o.scheduled_task_id
        JOIN v2_trusted_owner_ingress i ON i.message_id=o.request_message_id
        JOIN v2_user_settings s ON s.key='admin_user_id' AND s.value=o.owner_id
        JOIN v2_users u ON u.user_id=o.owner_id AND u.role IN ('admin','owner') AND u.is_active=1
        WHERE o.scheduled_task_id=?''', (schedule_id,)).fetchone()
    if (row is None or row['origin_owner'] != row['owner_user_id'] or row['request_owner'] != row['origin_owner']
            or row['peer_uid'] != os.getuid() or row['request_payload_sha256'] != row['current_request_sha256']
            or row['schedule_sha256'] != _digest(row)):
        return None
    return row


def verified_telegram_schedule_destination_locked(conn, schedule_id):
    """Resolve a currently authorized Telegram schedule's original private chat."""
    table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_schedule_telegram_origins'").fetchone()
    if not table:
        return None
    bound = conn.execute('SELECT request_message_id FROM v2_owner_schedule_telegram_origins WHERE scheduled_task_id=?',
                         (schedule_id,)).fetchone()
    if bound is None or verified_schedule_origin_locked(conn, schedule_id) is None:
        return None
    row = conn.execute('''SELECT owner_id,integration_id,provider_user_id,chat_id,provider_message_id
        FROM v2_owner_telegram_ingress WHERE message_id=?''', (bound['request_message_id'],)).fetchone()
    return ({**dict(row), 'request_message_id': bound['request_message_id']} if row is not None else None)


def _queue_digest(row):
    fields = ('message_id', 'timestamp', 'prompt', 'user_id', 'project_id', 'tag',
              'correlation_id', 'memory_session_id', 'metadata_json')
    return hashlib.sha256(json.dumps([row[key] for key in fields], ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


def stage_due_occurrence_locked(conn, schedule_id, *, now):
    """Atomically queue a due provenance-bound one-time schedule, at most once.

    A queue row is not a local-owner ingress and no worker may execute it until
    it has its own scheduled-origin claim fence. Caller owns the transaction.
    """
    if not conn.in_transaction:
        raise ValueError('owner_schedule_origin_transaction_required')
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ValueError('owner_schedule_clock_invalid')
    ensure_schedule_origin_schema(conn)
    row = verified_schedule_origin_locked(conn, schedule_id)
    if row is None or row['status'] != 'active' or not row['next_run_at']:
        raise ValueError('owner_schedule_origin_unavailable')
    schedule = json.loads(row['schedule_json'])
    if schedule.get('kind') != 'once' or schedule.get('run_at') != row['next_run_at']:
        raise ValueError('owner_schedule_origin_unavailable')
    due = datetime.fromisoformat(row['next_run_at'])
    if due.tzinfo is None or due.utcoffset() is None or now.astimezone(timezone.utc) < due.astimezone(timezone.utc):
        raise ValueError('owner_schedule_not_due')
    prior = conn.execute('SELECT * FROM v2_owner_schedule_occurrences WHERE schedule_id=?', (schedule_id,)).fetchone()
    if prior is not None:
        return prior['message_id']
    message_id = uuid4().hex
    timestamp = now.astimezone(timezone.utc).isoformat()
    prompt = ('This scheduled task is due now. Carry out the original request now; '
              'do not schedule it again. Your response is the task result intended for delivery; '
              'delivery is handled separately and is not guaranteed. If the request was to remind '
              'or notify the owner, express the requested reminder or notification in your response now. '
              'For other requests, do the available work and report the outcome or what remains uncertain.\n'
              'Original request:\n' + row['prompt'])
    metadata = json.dumps({'intelligence_engine': 'single_loop_experimental',
        'intelligence_schema_version': 1, 'routing_disposition': 'pdca_task'}, sort_keys=True)
    conn.execute('''INSERT INTO v2_inbound_messages
        (message_id,sequence,timestamp,prompt,user_id,project_id,tag,correlation_id,memory_session_id,
         metadata_json,queued_at,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,'pending')''',
        (message_id, time.time_ns(), timestamp, prompt, row['owner_user_id'], '', '', '', '', metadata, timestamp))
    queued = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()
    conn.execute('''INSERT INTO v2_owner_schedule_occurrences
        (schedule_id,owner_id,message_id,due_at,payload_sha256,state) VALUES (?,?,?,?,?,'queued')''',
        (schedule_id, row['owner_user_id'], message_id, row['next_run_at'], _queue_digest(queued)))
    return message_id


def verified_occurrence_locked(conn, message_id, *, now):
    """Verify scheduled provenance and queued payload; caller must fence a claim."""
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        return False
    occurrence = conn.execute('''SELECT * FROM v2_owner_schedule_occurrences WHERE message_id=?''', (message_id,)).fetchone()
    if occurrence is None or occurrence['state'] != 'queued':
        return False
    row = verified_schedule_origin_locked(conn, occurrence['schedule_id'])
    if (row is None or row['status'] != 'active' or row['owner_user_id'] != occurrence['owner_id']
            or row['next_run_at'] != occurrence['due_at']):
        return False
    try:
        due = datetime.fromisoformat(occurrence['due_at'])
        if due.tzinfo is None or due.utcoffset() is None or now.astimezone(timezone.utc) < due.astimezone(timezone.utc):
            return False
    except (ValueError, TypeError):
        return False
    queued = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()
    return bool(queued is not None and queued['user_id'] == occurrence['owner_id']
                and _queue_digest(queued) == occurrence['payload_sha256'])
