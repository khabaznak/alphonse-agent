"""Durable one-shot Telegram reply ledger for a verified owner request.

The caller must supply verifiers which check origin, current access and the
selected draft using the same SQLite connection. This module is a building
block, not an authorization mechanism or a running integration.
"""
from __future__ import annotations

import sqlite3
from typing import Callable

from alphonse.agent_v2.database import connect_database


class TelegramReplyUnavailable(RuntimeError):
    pass


_FIELDS = ('message_id', 'owner_id', 'integration_id', 'provider_user_id', 'chat_id', 'provider_message_id', 'text')


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_telegram_replies (
        message_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, integration_id TEXT NOT NULL,
        provider_user_id TEXT NOT NULL, chat_id TEXT NOT NULL, provider_message_id TEXT NOT NULL,
        text TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('pending','started_unconfirmed','provider_accepted','finalized')),
        reply_provider_message_id TEXT NOT NULL DEFAULT '') STRICT''')


def stage_locked(conn: sqlite3.Connection, reply: dict, *, verify: Callable) -> None:
    """Stage exactly one verified draft in the caller's open transaction.

    ``verify(conn, reply)`` must revalidate the authenticated origin, recipient,
    actor and selected answer while the caller's claim and checkpoint are held.
    Identical repeated staging is harmless; changed payload is rejected.
    """
    if (not conn.in_transaction or not isinstance(reply, dict) or set(reply) != set(_FIELDS)
            or any(not isinstance(reply[key], str) or not reply[key].strip() for key in _FIELDS)
            or not callable(verify) or verify(conn, dict(reply)) is not True):
        raise TelegramReplyUnavailable('telegram_reply_unavailable')
    existing = conn.execute('SELECT ' + ','.join(_FIELDS) + ' FROM v2_owner_telegram_replies WHERE message_id=?',
                            (reply['message_id'],)).fetchone()
    if existing is not None:
        if tuple(existing) != tuple(reply[key] for key in _FIELDS):
            raise TelegramReplyUnavailable('telegram_reply_conflict')
        return
    conn.execute('INSERT INTO v2_owner_telegram_replies (' + ','.join(_FIELDS) + ',status) VALUES (?,?,?,?,?,?,?,?)',
                 (*[reply[key] for key in _FIELDS], 'pending'))


def dispatch_once(db_path: str, message_id: str, *, verify: Callable, send: Callable) -> str:
    """Try one verified pending reply; never retry after dispatch begins.

    ``send`` receives the exact integration_id, chat_id and text. A missing or ambiguous
    provider acknowledgement and all exceptions leave the durable row in
    ``started_unconfirmed``. This cannot atomically fence a provider call
    against revocation immediately after the pre-dispatch check.
    """
    if not message_id or not callable(verify) or not callable(send):
        raise TelegramReplyUnavailable('telegram_reply_unavailable')
    with connect_database(db_path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        cursor = conn.execute('SELECT ' + ','.join(_FIELDS) + ',status FROM v2_owner_telegram_replies WHERE message_id=?',
                              (message_id,))
        row = cursor.fetchone()
        if row is None:
            raise TelegramReplyUnavailable('telegram_reply_unavailable')
        values = dict(zip((*_FIELDS, 'status'), tuple(row)))
        if values['status'] != 'pending':
            return values['status']
        reply = {key: values[key] for key in _FIELDS}
        if verify(conn, reply) is not True:
            raise TelegramReplyUnavailable('telegram_reply_unavailable')
        changed = conn.execute("UPDATE v2_owner_telegram_replies SET status='started_unconfirmed' WHERE message_id=? AND status='pending'", (message_id,))
        if changed.rowcount != 1:
            raise TelegramReplyUnavailable('telegram_reply_unavailable')
        if verify(conn, reply) is not True:
            raise TelegramReplyUnavailable('telegram_reply_unavailable')
    try:
        provider_id = send(integration_id=reply['integration_id'], chat_id=reply['chat_id'], text=reply['text'])
    except Exception:
        return 'started_unconfirmed'
    if not isinstance(provider_id, str) or not provider_id.strip():
        return 'started_unconfirmed'
    with connect_database(db_path) as conn:
        result = conn.execute("UPDATE v2_owner_telegram_replies SET status='provider_accepted', reply_provider_message_id=? WHERE message_id=? AND status='started_unconfirmed'",
                              (provider_id.strip(), message_id))
        if result.rowcount != 1:
            raise TelegramReplyUnavailable('telegram_reply_unavailable')
    return 'provider_accepted'
