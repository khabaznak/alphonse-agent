"""Experimental, disconnected reminder-proposal state machine.

The caller must hold a claim-fenced owner transaction for preparation and
consumption and must authenticate a separate explicit Desktop approval for
confirmation. No such Desktop approval transport is currently connected. A
model judgment, ordinary conversation text, an actor argument, or this module
alone cannot authenticate a person or authorize an effect.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders
from .owner_clarification import load_bound_owner_exchange


class ProposalUnavailable(ValueError):
    pass


def _clock(now):
    if not isinstance(now, datetime) or now.tzinfo is None or now.utcoffset() is None:
        raise ProposalUnavailable('owner_reminder_proposal_clock_invalid')
    return now.astimezone(timezone.utc)


def _transaction(conn):
    if not conn.in_transaction:
        raise ProposalUnavailable('owner_reminder_proposal_transaction_required')


def _active_owner(conn, owner):
    if not OwnerDesktopReminders._owner(conn, owner):
        raise ProposalUnavailable('owner_reminder_proposal_unavailable')


def ensure_table(conn):
    _transaction(conn)
    conn.execute('''CREATE TABLE IF NOT EXISTS owner_reminder_proposals (
        id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, request_id TEXT NOT NULL,
        prior_message_id TEXT NOT NULL, exchange_sha256 TEXT NOT NULL, message TEXT NOT NULL, due_at TEXT NOT NULL,
        timezone TEXT NOT NULL, expires_at TEXT NOT NULL, proposal_sha256 TEXT NOT NULL,
        state TEXT NOT NULL CHECK(state IN ('pending','confirmed','consumed','cancelled')),
        reminder_id TEXT NOT NULL DEFAULT '', UNIQUE(owner_id,request_id)) STRICT''')
    conn.execute('''CREATE TABLE IF NOT EXISTS owner_reminder_proposal_consumptions (
        owner_id TEXT NOT NULL, prior_message_id TEXT NOT NULL, proposal_id TEXT NOT NULL UNIQUE,
        reminder_id TEXT NOT NULL, PRIMARY KEY(owner_id, prior_message_id)) STRICT''')


def _digest(row):
    values = [row[k] for k in ('id', 'owner_id', 'request_id', 'prior_message_id', 'exchange_sha256',
                                'message', 'due_at', 'timezone', 'expires_at')]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _load(conn, task, proposal_id, *, now):
    _transaction(conn)
    _active_owner(conn, task.user)
    row = conn.execute('SELECT * FROM owner_reminder_proposals WHERE id=? AND owner_id=? AND request_id=?',
                       (proposal_id, task.user, task.message_id)).fetchone()
    if row is None or not hmac.compare_digest(row['proposal_sha256'], _digest(row)):
        raise ProposalUnavailable('owner_reminder_proposal_unavailable')
    exchange = load_bound_owner_exchange(conn, task)
    if (row['prior_message_id'] != exchange.prior_message_id
            or not hmac.compare_digest(row['exchange_sha256'], exchange.sha256)):
        raise ProposalUnavailable('owner_reminder_proposal_context_changed')
    if row['state'] in {'pending', 'confirmed'} and _clock(now) >= datetime.fromisoformat(row['expires_at']):
        raise ProposalUnavailable('owner_reminder_proposal_expired')
    return row


def prepare_locked(conn, task, *, message, due_at, timezone_name, now):
    """Store exact proposed fields, without confirming or creating a reminder."""
    _transaction(conn)
    _active_owner(conn, task.user)
    current = _clock(now)
    if not isinstance(message, str) or not message.strip() or len(message) > 500:
        raise ProposalUnavailable('owner_reminder_proposal_invalid')
    try:
        zone = ZoneInfo(timezone_name)
        due = datetime.fromisoformat(due_at)
        if due.tzinfo is None or due.utcoffset() is None:
            raise ValueError
        local = due.astimezone(zone)
        if local.replace(tzinfo=None) != due.replace(tzinfo=None) or local.utcoffset() != due.utcoffset():
            raise ValueError
        wall = local.replace(tzinfo=None)
        if wall.replace(tzinfo=zone, fold=0).utcoffset() != wall.replace(tzinfo=zone, fold=1).utcoffset():
            raise ValueError
        due_utc = due.astimezone(timezone.utc)
        if not current < due_utc <= current + timedelta(days=366):
            raise ValueError
    except (ValueError, TypeError, OverflowError, KeyError):
        raise ProposalUnavailable('owner_reminder_proposal_time_invalid') from None
    exchange = load_bound_owner_exchange(conn, task)
    ensure_table(conn)
    proposed = {'id': secrets.token_hex(16), 'owner_id': task.user, 'request_id': task.message_id,
                'prior_message_id': exchange.prior_message_id,
                'exchange_sha256': exchange.sha256, 'message': message.strip(),
                'due_at': due_utc.isoformat(), 'timezone': timezone_name,
                'expires_at': (current + timedelta(minutes=10)).isoformat()}
    proposed['proposal_sha256'] = _digest(proposed)
    previous = conn.execute('SELECT id FROM owner_reminder_proposals WHERE owner_id=? AND request_id=?',
                            (task.user, task.message_id)).fetchone()
    if previous is not None:
        # Do not silently overwrite an earlier proposal or invent authority
        # for a changed interpretation of the same owner reply.
        raise ProposalUnavailable('owner_reminder_proposal_already_exists')
    conn.execute('''INSERT INTO owner_reminder_proposals
        (id,owner_id,request_id,prior_message_id,exchange_sha256,message,due_at,timezone,expires_at,proposal_sha256,state)
        VALUES (:id,:owner_id,:request_id,:prior_message_id,:exchange_sha256,:message,:due_at,:timezone,:expires_at,
                :proposal_sha256,'pending')''', proposed)
    return {k: proposed[k] for k in ('id', 'message', 'due_at', 'timezone', 'expires_at', 'proposal_sha256')}


def confirm_locked(conn, task, *, proposal_id, proposal_sha256, authenticated_owner, now):
    """Record an explicitly authenticated UI approval of the exact proposal.

    The future transport must derive ``authenticated_owner`` from a protected
    owner interaction, not from model output or a caller-controlled payload.
    This function alone does not supply such a transport.
    """
    if authenticated_owner != task.user:
        raise ProposalUnavailable('owner_reminder_proposal_unavailable')
    row = _load(conn, task, proposal_id, now=now)
    if (row['state'] not in {'pending', 'confirmed'} or not isinstance(proposal_sha256, str)
            or not hmac.compare_digest(row['proposal_sha256'], proposal_sha256)):
        raise ProposalUnavailable('owner_reminder_proposal_unavailable')
    conn.execute("UPDATE owner_reminder_proposals SET state='confirmed' WHERE id=? AND state='pending'", (proposal_id,))


def cancel_locked(conn, task, *, proposal_id, authenticated_owner, now):
    if authenticated_owner != task.user:
        raise ProposalUnavailable('owner_reminder_proposal_unavailable')
    row = _load(conn, task, proposal_id, now=now)
    if row['state'] not in {'pending', 'confirmed'}:
        raise ProposalUnavailable('owner_reminder_proposal_unavailable')
    conn.execute("UPDATE owner_reminder_proposals SET state='cancelled' WHERE id=?", (proposal_id,))


def consume_locked(conn, task, *, proposal_id, now, is_cancelled):
    """Create once and consume approval atomically; caller must roll back on error."""
    if not callable(is_cancelled) or is_cancelled() is not False:
        raise ProposalUnavailable('owner_reminder_proposal_unavailable')
    row = _load(conn, task, proposal_id, now=now)
    committed = conn.execute('''SELECT proposal_id,reminder_id FROM owner_reminder_proposal_consumptions
        WHERE owner_id=? AND prior_message_id=?''', (task.user, row['prior_message_id'])).fetchone()
    if row['state'] == 'consumed':
        existing = conn.execute('''SELECT id,message,due_at FROM owner_desktop_reminders
            WHERE id=? AND owner_id=? AND request_id=?''', (row['reminder_id'], task.user, task.message_id)).fetchone()
        if (committed is None or committed['proposal_id'] != proposal_id
                or committed['reminder_id'] != row['reminder_id'] or existing is None
                or existing['message'] != row['message'] or existing['due_at'] != row['due_at']):
            raise ProposalUnavailable('owner_reminder_proposal_effect_unavailable')
        return {'id': existing['id'], 'due_at': existing['due_at'], 'status': 'already_consumed'}
    if committed is not None:
        raise ProposalUnavailable('owner_reminder_prior_intent_already_consumed')
    if row['state'] != 'confirmed' or _clock(now) >= datetime.fromisoformat(row['due_at']):
        raise ProposalUnavailable('owner_reminder_proposal_unavailable')
    OwnerDesktopReminders.ensure_table(conn)
    if conn.execute('SELECT 1 FROM owner_desktop_reminders WHERE owner_id=? AND request_id=?',
                    (task.user, task.message_id)).fetchone() is not None:
        # A different route already recorded an effect for this request.
        # Reconcile it instead of claiming this proposal created it.
        raise ProposalUnavailable('owner_reminder_proposal_effect_unavailable')
    effect = OwnerDesktopReminders.create_locked(conn, task.user, row['message'], row['due_at'], task.message_id, now=now)
    if effect.get('state') != 'pending' or is_cancelled() is not False:
        raise ProposalUnavailable('owner_reminder_proposal_unavailable')
    changed = conn.execute("UPDATE owner_reminder_proposals SET state='consumed',reminder_id=? WHERE id=? AND state='confirmed'",
                           (effect['id'], proposal_id))
    if changed.rowcount != 1:
        raise ProposalUnavailable('owner_reminder_proposal_unavailable')
    conn.execute('''INSERT INTO owner_reminder_proposal_consumptions
                    (owner_id,prior_message_id,proposal_id,reminder_id) VALUES (?,?,?,?)''',
                 (task.user, row['prior_message_id'], proposal_id, effect['id']))
    return {'id': effect['id'], 'due_at': effect['due_at'], 'status': 'created'}
