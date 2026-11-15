"""New owner-authorized, one-time Desktop reminders; no external delivery."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from alphonse.agent_v2.database import connect_database


class OwnerDesktopReminders:
    """A local poll claims each reminder at most once, even across restart.

    A crash after claim and before acknowledgement is intentionally reported as
    uncertain. It is never automatically replayed or counted as human receipt.
    """

    def __init__(self, path: str):
        if not path or path == ':memory:':
            raise ValueError('owner_reminders_persistent_store_required')
        self.path = str(Path(path).expanduser())
        with connect_database(self.path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS owner_desktop_reminders (
                id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, request_id TEXT NOT NULL,
                message TEXT NOT NULL, due_at TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('pending','offered','acknowledged','cancelled')),
                token_hash TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL,
                offered_at TEXT NOT NULL DEFAULT '', acknowledged_at TEXT NOT NULL DEFAULT '',
                UNIQUE(owner_id, request_id)) STRICT''')

    @staticmethod
    def _owner(conn, owner):
        return bool(owner and conn.execute('''SELECT 1 FROM v2_users u
            JOIN v2_user_settings s ON s.key='admin_user_id' AND s.value=u.user_id
            WHERE u.user_id=? AND u.role IN ('admin','owner') AND u.is_active=1''', (owner,)).fetchone())

    def create(self, owner, message, due_at, request_id, *, now=None):
        if (not isinstance(message, str) or not message.strip() or len(message) > 500
                or not isinstance(request_id, str) or len(request_id) != 32
                or any(c not in '0123456789abcdef' for c in request_id)
                or not isinstance(due_at, str)):
            raise ValueError('owner_reminder_invalid')
        current = now or datetime.now(timezone.utc)
        try:
            due = datetime.fromisoformat(due_at)
            if due.tzinfo is None:
                raise ValueError
            due = due.astimezone(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            raise ValueError('owner_reminder_time_invalid') from None
        if current.tzinfo is None:
            raise ValueError('owner_reminder_time_invalid')
        due_text, text = due.isoformat(), message.strip()
        with connect_database(self.path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            if not self._owner(conn, owner):
                raise PermissionError('owner_reminder_unavailable')
            row = conn.execute('SELECT id,message,due_at,state FROM owner_desktop_reminders WHERE owner_id=? AND request_id=?',
                               (owner, request_id)).fetchone()
            if row is not None:
                if row[1] != text or row[2] != due_text:
                    raise ValueError('owner_reminder_request_conflict')
                return {'id': row[0], 'due_at': due_text, 'state': row[3]}
            if not current < due <= current + timedelta(days=366):
                raise ValueError('owner_reminder_time_invalid')
            reminder_id = uuid4().hex
            conn.execute('''INSERT INTO owner_desktop_reminders
                (id,owner_id,request_id,message,due_at,state,created_at) VALUES (?,?,?,?,?,'pending',?)''',
                (reminder_id, owner, request_id, text, due_text, current.astimezone(timezone.utc).isoformat()))
            return {'id': reminder_id, 'due_at': due_text, 'state': 'pending'}

    def take_due(self, owner, *, now=None):
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        with connect_database(self.path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            if not self._owner(conn, owner):
                raise PermissionError('owner_reminder_unavailable')
            row = conn.execute('''SELECT id,message,due_at FROM owner_desktop_reminders
                WHERE owner_id=? AND state='pending' AND due_at<=? ORDER BY due_at,id LIMIT 1''',
                (owner, current)).fetchone()
            if row is None:
                return {'reminder': None}
            token = secrets.token_hex(32)
            digest = hashlib.sha256(token.encode()).hexdigest()
            changed = conn.execute('''UPDATE owner_desktop_reminders SET state='offered',token_hash=?,offered_at=?
                WHERE id=? AND owner_id=? AND state='pending' ''', (digest, current, row[0], owner))
            if changed.rowcount != 1:
                raise RuntimeError('owner_reminder_unavailable')
            return {'reminder': {'id': row[0], 'message': row[1], 'due_at': row[2], 'acknowledgement_token': token}}

    def acknowledge(self, owner, reminder_id, token, *, now=None):
        if not isinstance(reminder_id, str) or not isinstance(token, str) or len(token) != 64:
            raise ValueError('owner_reminder_ack_invalid')
        digest = hashlib.sha256(token.encode()).hexdigest()
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat()
        with connect_database(self.path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            if not self._owner(conn, owner):
                raise PermissionError('owner_reminder_unavailable')
            row = conn.execute('SELECT state,token_hash FROM owner_desktop_reminders WHERE id=? AND owner_id=?',
                               (reminder_id, owner)).fetchone()
            if row is None or row[0] not in ('offered', 'acknowledged') or not hmac.compare_digest(row[1], digest):
                raise ValueError('owner_reminder_ack_invalid')
            if row[0] == 'offered':
                conn.execute("UPDATE owner_desktop_reminders SET state='acknowledged',acknowledged_at=? WHERE id=? AND owner_id=?",
                             (current, reminder_id, owner))
            return {'status': 'client_acknowledged'}

    def list_open(self, owner):
        with connect_database(self.path) as conn:
            conn.execute('BEGIN')
            if not self._owner(conn, owner):
                raise PermissionError('owner_reminder_unavailable')
            rows = conn.execute("SELECT id,message,due_at,state FROM owner_desktop_reminders WHERE owner_id=? AND state IN ('pending','offered') ORDER BY due_at,id LIMIT 101", (owner,)).fetchall()
            if len(rows) > 100:
                raise ValueError('owner_reminders_too_many')
            return [{'id': row[0], 'message': row[1], 'due_at': row[2], 'state': row[3]} for row in rows]

    def lookup(self, owner, request_id):
        if (not isinstance(request_id, str) or len(request_id) != 32
                or any(c not in '0123456789abcdef' for c in request_id)):
            raise ValueError('owner_reminder_request_invalid')
        with connect_database(self.path) as conn:
            conn.execute('BEGIN')
            if not self._owner(conn, owner):
                raise PermissionError('owner_reminder_unavailable')
            row = conn.execute('SELECT id,message,due_at,state FROM owner_desktop_reminders WHERE owner_id=? AND request_id=?', (owner, request_id)).fetchone()
            return {'reminder': None if row is None else {'id': row[0], 'message': row[1], 'due_at': row[2], 'state': row[3]}}

    def cancel(self, owner, reminder_id):
        if not isinstance(reminder_id, str):
            raise ValueError('owner_reminder_cancel_invalid')
        with connect_database(self.path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            if not self._owner(conn, owner):
                raise PermissionError('owner_reminder_unavailable')
            changed = conn.execute("UPDATE owner_desktop_reminders SET state='cancelled' WHERE id=? AND owner_id=? AND state='pending'", (reminder_id, owner))
            if changed.rowcount != 1:
                raise ValueError('owner_reminder_not_pending')
            return {'status': 'cancelled'}

    def resolve_unconfirmed(self, owner, reminder_id):
        """An authenticated owner explicitly clears a previously offered item."""
        if not isinstance(reminder_id, str):
            raise ValueError('owner_reminder_resolve_invalid')
        current = datetime.now(timezone.utc).isoformat()
        with connect_database(self.path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            if not self._owner(conn, owner):
                raise PermissionError('owner_reminder_unavailable')
            changed = conn.execute("UPDATE owner_desktop_reminders SET state='acknowledged',acknowledged_at=? WHERE id=? AND owner_id=? AND state='offered'", (current, reminder_id, owner))
            if changed.rowcount != 1:
                raise ValueError('owner_reminder_not_unconfirmed')
            return {'status': 'owner_resolved'}
