"""Owner-scoped visibility of scheduled work needing attention.

This never retries a task or asserts whether an interrupted effect occurred.
"""
from __future__ import annotations

from datetime import datetime, timezone


def record_uncertain_schedule(queue, receipt, task):
    """Persist attention while the scheduled-origin claim is still valid.

    Failure to record is safe: an expired or missing queue claim is still
    reported as uncertain by the read-only observer. A stale worker cannot
    change the occurrence or mark another actor's work.
    """
    with queue.trusted_scheduled_owner_claim_transaction(receipt, task) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_schedule_attention (
            message_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL,
            recorded_at TEXT NOT NULL) STRICT''')
        conn.execute('''INSERT OR IGNORE INTO v2_owner_schedule_attention
            (message_id,owner_id,recorded_at) VALUES (?,?,?)''',
            (receipt.message_id, receipt.user_id, datetime.now(timezone.utc).isoformat()))


def list_schedule_attention(queue, owner_id, *, now=None):
    """Return owner-created schedule states without modifying or retrying them.

    A dead worker's lease is not evidence that a side effect did or did not
    happen. Missing, expired or failed queue records are therefore uncertain.
    Callers must authenticate the requesting owner independently.
    """
    if not isinstance(owner_id, str) or not owner_id:
        raise ValueError('owner_schedule_attention_unavailable')
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None or current.utcoffset() is None:
        raise ValueError('owner_schedule_attention_unavailable')
    current = current.astimezone(timezone.utc)
    with queue._connect() as conn:
        table = conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_owner_schedule_origins'").fetchone()
        if table is None:
            return {}
        attention = conn.execute("SELECT 1 FROM sqlite_master WHERE name='v2_owner_schedule_attention'").fetchone()
        marker = ("EXISTS (SELECT 1 FROM v2_owner_schedule_attention a WHERE a.message_id=x.message_id "
                  "AND a.owner_id=o.owner_id)" if attention else '0')
        rows = conn.execute(f'''SELECT o.scheduled_task_id, t.status AS schedule_status,
            x.state AS occurrence_status, q.status AS queue_status, q.lease_expires_at,
            {marker} AS attention
            FROM v2_owner_schedule_origins o
            JOIN v2_scheduled_tasks t ON t.scheduled_task_id=o.scheduled_task_id AND t.owner_user_id=o.owner_id
            LEFT JOIN v2_owner_schedule_occurrences x ON x.schedule_id=o.scheduled_task_id AND x.owner_id=o.owner_id
            LEFT JOIN v2_inbound_messages q ON q.message_id=x.message_id AND q.user_id=o.owner_id
            WHERE o.owner_id=?''', (owner_id,)).fetchall()
    result = {}
    for row in rows:
        state = row['occurrence_status']
        if state in {'completed', 'failed', 'cancelled'}:
            result[row['scheduled_task_id']] = state
        elif state is None:
            result[row['scheduled_task_id']] = row['schedule_status']
        elif row['attention'] or row['queue_status'] not in {'pending', 'processing'}:
            result[row['scheduled_task_id']] = 'uncertain'
        elif row['queue_status'] == 'pending':
            result[row['scheduled_task_id']] = 'queued'
        else:
            try:
                expiry = datetime.fromisoformat(row['lease_expires_at'])
                if expiry.tzinfo is None or expiry.utcoffset() is None:
                    raise ValueError
                result[row['scheduled_task_id']] = 'processing' if expiry.astimezone(timezone.utc) > current else 'uncertain'
            except (TypeError, ValueError, OverflowError):
                result[row['scheduled_task_id']] = 'uncertain'
    return result
