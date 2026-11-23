"""Incremental Telegram owner polling with durable no-replay cutoff.

This adapter does not send replies. Use only with an authenticated Telegram
Bot API client constructed by trusted code. No caller input is authentication.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import threading

from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import _allowed, enqueue_telegram_owner
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue


def ensure_telegram_poll_schema(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_telegram_poll_state (
        integration_id TEXT PRIMARY KEY, activated_at_epoch INTEGER NOT NULL, next_offset INTEGER) STRICT''')
    conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_telegram_held_updates (
        integration_id TEXT NOT NULL, update_id INTEGER NOT NULL, reason TEXT NOT NULL,
        update_json TEXT NOT NULL, PRIMARY KEY(integration_id,update_id)) STRICT''')


class TelegramOwnerPoller:
    """Consume updates once; preserve pre-activation or unsupported updates.

    Progress is committed only after an update is durably held or enqueued.
    Re-fetch after a crash is harmless because ingress and holds are deduped.
    Telegram's offset acknowledges earlier updates, so never advance it ahead
    of local persistence. Held updates are never processed automatically.
    """

    def __init__(self, queue: SQLiteMessageQueue, integration_id: str, client, *, activated_at: datetime,
                 on_accepted=None):
        if (type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing
                or not isinstance(integration_id, str) or not integration_id.strip()
                or not isinstance(activated_at, datetime) or activated_at.tzinfo is None
                or activated_at.utcoffset() is None or not callable(getattr(client, 'get_updates', None))):
            raise ValueError('telegram_owner_poller_configuration_invalid')
        self.queue, self.integration_id, self.client = queue, integration_id.strip(), client
        self.on_accepted = on_accepted
        self._poll_lock = threading.Lock()
        with queue._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            ensure_telegram_poll_schema(conn)
            conn.execute('''INSERT OR IGNORE INTO v2_owner_telegram_poll_state
                (integration_id,activated_at_epoch,next_offset) VALUES (?,?,NULL)''',
                (self.integration_id, int(activated_at.timestamp())))

    def poll_once(self):
        # One poller instance must be owned by the daemon's single-instance
        # lifecycle. Concurrent callers on that instance cannot reorder updates.
        with self._poll_lock:
            return self._poll_once()

    def _poll_once(self):
        with self.queue._connect() as conn:
            row = conn.execute('SELECT * FROM v2_owner_telegram_poll_state WHERE integration_id=?',
                               (self.integration_id,)).fetchone()
        if row is None:
            raise RuntimeError('telegram_owner_poll_state_unavailable')
        offset, cutoff = row['next_offset'], row['activated_at_epoch']
        updates = self.client.get_updates(offset=offset, timeout=0)
        if not isinstance(updates, list) or any(not isinstance(item, dict) or type(item.get('update_id')) is not int
                                               or item['update_id'] < 0 for item in updates):
            raise ValueError('telegram_owner_updates_invalid')
        ordered = sorted(updates, key=lambda item: item['update_id'])
        queued, held = 0, 0
        for update in ordered:
            update_id = update['update_id']
            if offset is not None and update_id < offset:
                continue
            message = update.get('message')
            timestamp = message.get('date') if isinstance(message, dict) else None
            reason = ''
            # Telegram timestamps have one-second precision. An update in the
            # activation second might precede activation and must be held.
            if type(timestamp) is not int or timestamp <= cutoff:
                reason = 'before_activation_or_unknown_time'
            if not reason:
                try:
                    message_id = enqueue_telegram_owner(self.queue, integration_id=self.integration_id, update=update)
                    if self.on_accepted is not None:
                        try:
                            self.on_accepted(message_id)
                        except Exception:
                            # Feedback must never change ingress or cause replay.
                            pass
                    queued += 1
                except (ValueError, PermissionError):
                    reason = 'unsupported_or_unauthorized'
            if reason:
                # Never retain content from unmapped senders or group chats.
                # Preserve held content only for the currently mapped owner.
                sender = message.get('from') if isinstance(message, dict) else None
                chat = message.get('chat') if isinstance(message, dict) else None
                owner_update = False
                if (isinstance(sender, dict) and type(sender.get('id')) is int
                        and isinstance(chat, dict) and type(chat.get('id')) is int and chat.get('type') == 'private'):
                    with self.queue._connect() as conn:
                        owner_update = _allowed(conn, self.integration_id, str(sender['id']), str(chat['id'])) is not None
                raw = json.dumps(update if owner_update else {}, ensure_ascii=False, sort_keys=True, allow_nan=False)
                if len(raw.encode('utf-8')) > 1048576:
                    raise ValueError('telegram_owner_held_update_too_large')
                with self.queue._connect() as conn:
                    conn.execute('BEGIN IMMEDIATE')
                    prior = conn.execute('''SELECT update_json,reason FROM v2_owner_telegram_held_updates
                        WHERE integration_id=? AND update_id=?''', (self.integration_id, update_id)).fetchone()
                    if prior is not None and (prior['update_json'] != raw or prior['reason'] != reason):
                        raise ValueError('telegram_owner_held_update_conflict')
                    conn.execute('''INSERT OR IGNORE INTO v2_owner_telegram_held_updates
                        VALUES (?,?,?,?)''', (self.integration_id, update_id, reason, raw))
                held += 1
            with self.queue._connect() as conn:
                conn.execute('BEGIN IMMEDIATE')
                # A stale concurrent poll cannot move the cursor backwards.
                conn.execute('''UPDATE v2_owner_telegram_poll_state SET next_offset=MAX(COALESCE(next_offset,0),?)
                    WHERE integration_id=?''', (update_id + 1, self.integration_id))
            offset = update_id + 1
        return {'queued': queued, 'held': held, 'next_offset': offset}
