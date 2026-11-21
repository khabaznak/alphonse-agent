"""Durable owner Desktop presentation of a recorded scheduled response.

Listing is repeatable and never re-runs work. A client acknowledgement means
the result was presented by that client, not that a person read it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import re

from alphonse.agent_v2.database import connect_database
from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders


def ensure_table(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_scheduled_deliveries (
        message_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, event_id TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL CHECK(kind IN ('answer','question')),
        state TEXT NOT NULL CHECK(state IN ('pending','acknowledged')),
        created_at TEXT NOT NULL, acknowledged_at TEXT NOT NULL DEFAULT '') STRICT''')


def enqueue_locked(conn, *, owner_id, message_id, event_id, kind):
    """Queue a verified conversation event in its creator's transaction."""
    if (not conn.in_transaction or not OwnerDesktopReminders._owner(conn, owner_id)
            or not isinstance(message_id, str) or re.fullmatch(r'[0-9a-f]{32}', message_id) is None
            or not isinstance(event_id, str) or not event_id or kind not in {'answer', 'question'}):
        raise ValueError('owner_scheduled_delivery_invalid')
    event = conn.execute('''SELECT 1 FROM v2_conversation_events WHERE event_id=? AND owner_user_id=?
        AND source='scheduled' AND role='assistant' AND source_message_id=?''',
        (event_id, owner_id, 'outbound:scheduled:' + message_id)).fetchone()
    occurrence = conn.execute('''SELECT 1 FROM v2_owner_schedule_occurrences
        WHERE message_id=? AND owner_id=? AND state='queued' ''', (message_id, owner_id)).fetchone()
    if event is None or occurrence is None:
        raise ValueError('owner_scheduled_delivery_invalid')
    ensure_table(conn)
    conn.execute('''INSERT INTO v2_owner_scheduled_deliveries
        (message_id,owner_id,event_id,kind,state,created_at) VALUES (?,?,?,?,'pending',?)''',
        (message_id, owner_id, event_id, kind, datetime.now(timezone.utc).isoformat()))


class OwnerScheduledDelivery:
    def __init__(self, path):
        if not path or path == ':memory:':
            raise ValueError('owner_scheduled_delivery_unavailable')
        self.path = str(Path(path).expanduser())
        with connect_database(self.path) as conn:
            ensure_table(conn)

    def list_pending(self, owner):
        with connect_database(self.path) as conn:
            conn.execute('BEGIN')
            if not OwnerDesktopReminders._owner(conn, owner):
                raise PermissionError('owner_scheduled_delivery_unavailable')
            rows = conn.execute('''SELECT d.message_id,d.kind,e.content,e.created_at,e.sequence
                FROM v2_owner_scheduled_deliveries d
                JOIN v2_conversation_events e ON e.event_id=d.event_id AND e.owner_user_id=d.owner_id
                WHERE d.owner_id=? AND d.state='pending' AND e.role='assistant' AND e.source='scheduled'
                  AND e.source_message_id='outbound:scheduled:' || d.message_id
                ORDER BY e.sequence LIMIT 100''', (owner,)).fetchall()
            return [{'id': row['message_id'], 'message': row['content'], 'kind': row['kind'],
                     'created_at': row['created_at'], 'sequence': row['sequence']} for row in rows]

    def acknowledge(self, owner, message_id):
        if not isinstance(message_id, str) or re.fullmatch(r'[0-9a-f]{32}', message_id) is None:
            raise ValueError('owner_scheduled_delivery_invalid')
        with connect_database(self.path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            if not OwnerDesktopReminders._owner(conn, owner):
                raise PermissionError('owner_scheduled_delivery_unavailable')
            changed = conn.execute('''UPDATE v2_owner_scheduled_deliveries SET state='acknowledged',
                acknowledged_at=CASE WHEN state='pending' THEN ? ELSE acknowledged_at END
                WHERE message_id=? AND owner_id=? AND state IN ('pending','acknowledged')''',
                (datetime.now(timezone.utc).isoformat(), message_id, owner))
            if changed.rowcount != 1:
                raise ValueError('owner_scheduled_delivery_invalid')
            return {'status': 'client_acknowledged'}
