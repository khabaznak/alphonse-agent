"""Opt-in polling lifecycle for the distinct Telegram owner inbox.

Starting this listener is a consequential integration action. Its caller must
own the daemon single-instance lifecycle and choose when to start it. It does
not dispatch replies, start other integrations, or drain the legacy outbox.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
import math
import threading
import time
from typing import Callable

from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_poller import TelegramOwnerPoller
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.integrations.telegram.runtime import TelegramHttpClient


class TelegramOwnerListener:
    def __init__(self, queue: SQLiteMessageQueue, integration_store, integration_id: str, *,
                 client_factory: Callable = TelegramHttpClient,
                 clock: Callable = lambda: datetime.now(timezone.utc), poll_interval: float = 1.0,
                 activity_interval: float = 4.0):
        if (type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing
                or not isinstance(integration_id, str) or not integration_id.strip()
                or not callable(client_factory) or not callable(clock)
                or isinstance(poll_interval, bool) or not isinstance(poll_interval, (int, float))
                or not .1 <= poll_interval <= 60
                or isinstance(activity_interval, bool) or not isinstance(activity_interval, (int, float))
                or not math.isfinite(activity_interval) or not .1 <= activity_interval <= 60
                or not callable(getattr(integration_store, 'get', None))
                or not callable(getattr(integration_store, 'resolve_credentials', None))):
            raise ValueError('telegram_owner_listener_configuration_invalid')
        try:
            if not os.path.samefile(queue.db_path, integration_store.db_path):
                raise ValueError('telegram_owner_listener_store_mismatch')
        except (OSError, AttributeError):
            raise ValueError('telegram_owner_listener_store_mismatch') from None
        self.queue, self.store, self.integration_id = queue, integration_store, integration_id.strip()
        self.client_factory, self.clock, self.poll_interval = client_factory, clock, float(poll_interval)
        self._stop = threading.Event()
        self._thread = None
        self._lifecycle_lock = threading.Lock()
        self._revision = None
        self._poller = None
        self.last_error = None
        self.activity_interval = activity_interval
        self._receipts = threading.Event()
        self._receipt_stop = None
        self._receipt_rate_lock = threading.Lock()
        self._receipt_next_at = 0.0
        self._activity_lock = threading.Lock()
        self._last_activity = {}

    def _current(self):
        record = self.store.get(self.integration_id)
        if (record is None or record.integration_id != self.integration_id or record.provider_key != 'telegram'
                or record.enabled is not True or record.revocation_pending or record.credential_deletion_pending):
            raise PermissionError('telegram_owner_listener_unavailable')
        return record

    @staticmethod
    def _record_revision(record):
        return (record.updated_at, record.credential_generation, record.credential_ref)

    def start(self):
        with self._lifecycle_lock:
            if self._thread and self._thread.is_alive():
                return
            record = self._current()
            revision = self._record_revision(record)
            credentials = self.store.resolve_credentials(record)
            token = credentials.get('bot_token') if isinstance(credentials, dict) else None
            if not isinstance(token, str) or not token.strip():
                raise ValueError('telegram_owner_listener_credentials_unavailable')
            if self._record_revision(self._current()) != revision:
                raise PermissionError('telegram_owner_listener_configuration_changed')
            client = self.client_factory(bot_token=token)
            now = self.clock()
            self._poller = TelegramOwnerPoller(self.queue, self.integration_id, client, activated_at=now,
                                                on_accepted=self._queue_receipt)
            self._revision = revision
            self.last_error = None
            self._stop.clear()
            # Each generation gets its own stop flag and queue; an old blocked
            # provider call cannot consume work from a restarted listener.
            self._receipts = threading.Event()
            self._receipt_stop = threading.Event()
            self._thread = threading.Thread(target=self._run, name='alphonse-telegram-owner', daemon=True)
            self._thread.start()
            threading.Thread(target=self._receipt_loop, args=(self._receipt_stop, self._receipts, client),
                             name='alphonse-telegram-receipts', daemon=True).start()

    def poll_once(self):
        if self._poller is None or self._revision is None or self._stop.is_set():
            raise RuntimeError('telegram_owner_listener_not_running')
        if self._record_revision(self._current()) != self._revision:
            raise PermissionError('telegram_owner_listener_configuration_changed')
        return self._poller.poll_once()

    def send_reply(self, *, integration_id: str, chat_id: str, text: str) -> str:
        """Send one previously authorized, durably marked reply.

        The caller must bind the exact recipient and commit its one-shot send
        marker before calling. This method provides only a point-in-time
        configuration check; it does not establish delivery or allow retry.
        """
        if (self._poller is None or self._revision is None or self._stop.is_set()
                or integration_id != self.integration_id or not isinstance(chat_id, str) or not chat_id.strip()
                or not isinstance(text, str) or not text.strip()):
            raise PermissionError('telegram_owner_reply_unavailable')
        if self._record_revision(self._current()) != self._revision or self._stop.is_set():
            raise PermissionError('telegram_owner_listener_configuration_changed')
        sender = getattr(self._poller.client, 'send_message', None)
        if not callable(sender):
            raise RuntimeError('telegram_owner_reply_unavailable')
        # Never expose client exception text, which can contain credentials or
        # private payloads. A failure here is an uncertain provider outcome.
        try:
            result = sender(chat_id=chat_id, text=text)
        except Exception:
            raise RuntimeError('telegram_owner_reply_unconfirmed') from None
        if not isinstance(result, str) or not result.strip():
            raise RuntimeError('telegram_owner_reply_unconfirmed')
        return result

    def _origin_locked(self, conn, message_id, *, finalized=False):
        from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import verified_telegram_origin_locked
        origin = verified_telegram_origin_locked(conn, message_id)
        if origin is None and finalized:
            # Only receipt feedback may use the finalized origin. Activity must
            # remain tied to a live inbound task.
            from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_handoff import verify_finalized_telegram_reply_locked
            if verify_finalized_telegram_reply_locked(conn, message_id):
                row = conn.execute('SELECT * FROM v2_owner_telegram_ingress WHERE message_id=?', (message_id,)).fetchone()
                origin = dict(row) if row is not None else None
        return origin if origin and origin['integration_id'] == self.integration_id else None

    def _authorized_origin(self, message_id):
        try:
            if self._stop.is_set() or self._record_revision(self._current()) != self._revision:
                return None
            with self.queue._connect() as conn:
                return self._origin_locked(conn, message_id)
        except Exception:
            return None

    def _queue_receipt(self, message_id):
        """Wake the worker; ingress already committed the pending receipt."""
        self._receipts.set()

    @staticmethod
    def _receipt_schema(conn):
        from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import ensure_telegram_receipt_schema
        ensure_telegram_receipt_schema(conn)

    def _receipt_loop(self, stopped, receipts, client):
        cursor = -1
        while not stopped.is_set():
            try:
                with self.queue._connect() as conn:
                    row = conn.execute("""SELECT r.message_id,i.update_id FROM v2_owner_telegram_receipts r
                        JOIN v2_owner_telegram_ingress i ON i.message_id=r.message_id
                        WHERE r.status='pending' AND i.integration_id=? AND i.update_id>?
                        ORDER BY i.update_id LIMIT 1""", (self.integration_id, cursor)).fetchone()
                if row is None:
                    cursor = -1
                    receipts.wait(.2)
                    receipts.clear()
                    continue
                message_id, cursor = row['message_id'], row['update_id']
            except Exception:
                stopped.wait(.2)
                continue
            if stopped.is_set():
                break
            # Serialize receipt attempts across listener generations. A
            # provider failure, including an opaque 429, triggers a long
            # cooldown without blocking polling or the owner worker.
            while not stopped.is_set():
                with self._receipt_rate_lock:
                    now = time.monotonic()
                    delay = self._receipt_next_at - now
                    if delay <= 0:
                        self._receipt_next_at = now + 1.0
                        break
                if stopped.wait(delay):
                    break
            if stopped.is_set():
                break
            if not stopped.is_set():
                try:
                    current = not self._stop.is_set() and self._record_revision(self._current()) == self._revision
                except Exception:
                    current = False
                if not current:
                    continue
                try:
                    with self.queue._connect() as conn:
                        conn.execute('BEGIN IMMEDIATE')
                        origin = self._origin_locked(conn, message_id, finalized=True)
                        claimed = conn.execute("UPDATE v2_owner_telegram_receipts SET status=? WHERE message_id=? AND status='pending'", ('attempted' if origin is not None else 'discarded', message_id)).rowcount
                except Exception:
                    continue  # Leave pending for a later scan.
                if not claimed or origin is None:
                    continue
                try:
                    if stopped.is_set() or self._stop.is_set() or self._record_revision(self._current()) != self._revision:
                        continue
                except Exception:
                    continue
                try:
                    # This reaction records receipt only, not progress or
                    # completion. It remains truthful after any outcome.
                    client.set_message_reaction(chat_id=origin['chat_id'],
                                                message_id=origin['provider_message_id'], emoji='👀')
                except Exception:
                    with self._receipt_rate_lock:
                        self._receipt_next_at = max(self._receipt_next_at, time.monotonic() + 60.0)

    def _send_typing(self, client, chat_id):
        # Limit typing refreshes per private chat.
        with self._activity_lock:
            now = time.monotonic()
            last = self._last_activity.get(chat_id)
            if last is not None and now - last < self.activity_interval:
                return
            self._last_activity[chat_id] = now
        client.send_chat_action(chat_id=chat_id, action='typing')

    def begin_activity(self, message_id: str, *, cancelled=lambda: False):
        """Refresh transient typing while work runs, without blocking it."""
        if self._authorized_origin(message_id) is None:
            return None
        stop = threading.Event()
        generation_stop, client = self._receipt_stop, self._poller.client
        def feedback():
            while not stop.is_set() and not generation_stop.is_set() and not cancelled():
                origin = self._authorized_origin(message_id)
                if origin is None or stop.is_set() or generation_stop.is_set() or cancelled():
                    break
                try:
                    self._send_typing(client, origin['chat_id'])
                except Exception:
                    # Includes 429 and uncertain provider outcomes. Do not
                    # repeatedly hammer a provider that rejected feedback.
                    break
                if stop.wait(self.activity_interval):
                    break
        threading.Thread(target=feedback, name='alphonse-telegram-activity', daemon=True).start()
        return stop

    def _run(self):
        while not self._stop.is_set():
            try:
                self.poll_once()
                self.last_error = None
            except PermissionError:
                self.last_error = 'configuration_changed'
                self._stop.set()
                break
            except Exception:
                # Never expose provider exception text: it can contain token or
                # private message material. Keep failures visible to the caller.
                self.last_error = 'poll_failed'
            self._stop.wait(self.poll_interval)

    def stop(self, *, timeout: float = 12.0) -> bool:
        """Request stop and report whether the worker actually exited."""
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 <= timeout <= 60:
            raise ValueError('telegram_owner_listener_timeout_invalid')
        self._stop.set()
        if self._receipt_stop is not None:
            self._receipt_stop.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=timeout)
        return not bool(thread and thread.is_alive())
