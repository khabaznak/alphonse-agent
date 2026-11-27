"""Telegram owner origin, distinct from a locally authenticated Desktop peer.

Only a trusted Telegram polling adapter may call ``enqueue_telegram_owner``;
its argument is not proof that an update came from Telegram. No public IPC
method should expose this function. This module does not authorize tools or
external effects, execute a message, or deliver a reply.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import hmac
import json
import sqlite3
import time
from uuid import uuid4

from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimReceipt, fenced_claim_transaction
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue, _claim_mode_conflict


def ensure_telegram_origin_schema(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_telegram_ingress (
        message_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, integration_id TEXT NOT NULL,
        provider_user_id TEXT NOT NULL, chat_id TEXT NOT NULL, provider_message_id TEXT NOT NULL,
        update_id INTEGER NOT NULL, address_id TEXT NOT NULL, address_updated_at TEXT NOT NULL,
        integration_updated_at TEXT NOT NULL, credential_generation INTEGER NOT NULL,
        history_event_id TEXT NOT NULL, payload_sha256 TEXT NOT NULL, update_sha256 TEXT NOT NULL,
        UNIQUE(integration_id, update_id), UNIQUE(integration_id, chat_id, provider_message_id)
    ) STRICT''')


def ensure_telegram_receipt_schema(conn):
    conn.execute("CREATE TABLE IF NOT EXISTS v2_owner_telegram_receipts (message_id TEXT PRIMARY KEY, status TEXT NOT NULL DEFAULT 'attempted') STRICT")
    if 'status' not in {row['name'] for row in conn.execute('PRAGMA table_info(v2_owner_telegram_receipts)')}:
        conn.execute("ALTER TABLE v2_owner_telegram_receipts ADD COLUMN status TEXT NOT NULL DEFAULT 'attempted'")


def _digest(values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def _allowed(conn, integration_id, provider_user_id, chat_id):
    # Restrict initial implementation to an active, explicitly mapped owner
    # in a private chat. A group chat must never inherit this authority.
    if provider_user_id != chat_id:
        return None
    row = conn.execute('''SELECT a.address_id,a.user_id,a.updated_at AS address_updated_at,
        i.updated_at AS integration_updated_at,i.config_json,i.credential_generation
        FROM v2_user_addresses a JOIN v2_users u ON u.user_id=a.user_id
        JOIN v2_user_settings s ON s.key='admin_user_id' AND s.value=u.user_id
        JOIN v2_integrations i ON i.integration_id=a.integration_id
        WHERE a.integration_id=? AND a.provider_user_id=? AND a.channel_target=?
          AND a.provider_key='telegram' AND a.is_active=1 AND u.is_active=1
          AND u.role IN ('owner','admin') AND i.provider_key='telegram' AND i.enabled=1
          AND i.revocation_pending=0 AND i.credential_deletion_pending=0''',
        (integration_id, provider_user_id, chat_id)).fetchone()
    if row is None:
        return None
    try:
        config = json.loads(row['config_json'])
        allowed = config.get('allowed_chat_ids', [])
        if not isinstance(allowed, list) or any(not isinstance(x, (str, int)) for x in allowed):
            return None
        if allowed and chat_id not in {str(x).strip() for x in allowed}:
            return None
    except (ValueError, TypeError, AttributeError):
        return None
    return row


def enqueue_telegram_owner(queue: SQLiteMessageQueue, *, integration_id: str, update: dict) -> str:
    """Persist one authenticated-by-caller text update, without dispatching it.

    Identity, integration, inbox and history are checked and written in a
    single transaction. Duplicate updates resolve to the original message;
    edited and non-private messages are not accepted. A caller must establish
    freshness/backlog policy before invoking this function.
    """
    if type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing or queue.db_path == ':memory:':
        raise ValueError('telegram_owner_ingress_unavailable')
    if not isinstance(integration_id, str) or not integration_id.strip() or not isinstance(update, dict):
        raise ValueError('telegram_owner_update_invalid')
    update_id, msg = update.get('update_id'), update.get('message')
    if type(update_id) is not int or update_id < 0 or not isinstance(msg, dict):
        raise ValueError('telegram_owner_update_invalid')
    sender, chat = msg.get('from'), msg.get('chat')
    if (not isinstance(sender, dict) or sender.get('is_bot') is True or not isinstance(chat, dict)
            or chat.get('type') != 'private' or type(sender.get('id')) is not int
            or type(chat.get('id')) is not int or type(msg.get('message_id')) is not int
            or not isinstance(msg.get('text'), str) or not msg['text'].strip()):
        raise ValueError('telegram_owner_update_invalid')
    provider_user_id, chat_id, provider_message_id = str(sender['id']), str(chat['id']), str(msg['message_id'])
    prompt, integration_id = msg['text'].strip(), integration_id.strip()
    update_digest = _digest([integration_id, update_id, provider_user_id, chat_id, provider_message_id, prompt])
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if _claim_mode_conflict(conn, fenced=True):
            raise RuntimeError('owner_worker_claim_mode_conflict')
        ensure_telegram_origin_schema(conn)
        ensure_telegram_receipt_schema(conn)
        owner = _allowed(conn, integration_id, provider_user_id, chat_id)
        if owner is None:
            raise PermissionError('telegram_owner_identity_denied')
        previous = conn.execute('''SELECT o.*,q.prompt FROM v2_owner_telegram_ingress o
            LEFT JOIN v2_inbound_messages q ON q.message_id=o.message_id
            WHERE o.integration_id=? AND o.update_id=?''', (integration_id, update_id)).fetchone()
        if previous is not None:
            if (previous['provider_user_id'], previous['chat_id'], previous['provider_message_id']) != (
                    provider_user_id, chat_id, provider_message_id) or not hmac.compare_digest(previous['update_sha256'], update_digest):
                raise ValueError('telegram_owner_duplicate_conflict')
            # Never re-enqueue an update, including after its queue row is gone.
            return previous['message_id']
        message_id, event_id = uuid4().hex, str(uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        metadata = json.dumps({'intelligence_engine': 'single_loop_experimental',
            'intelligence_schema_version': 1, 'routing_disposition': 'pdca_task'}, sort_keys=True)
        payload = [message_id, timestamp, prompt, owner['user_id'], '', '', '', '', metadata]
        conn.execute('''INSERT INTO v2_inbound_messages
            (message_id,sequence,timestamp,prompt,user_id,project_id,tag,correlation_id,memory_session_id,
             metadata_json,queued_at,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,'pending')''',
            (message_id, time.time_ns(), timestamp, prompt, owner['user_id'], '', '', '', '', metadata, timestamp))
        from alphonse.agent_v2.conversations import SQLiteConversationStore
        event = SQLiteConversationStore._record(conn, event_id=event_id, owner=owner['user_id'],
            project='', session='', role='user', message=prompt, source='telegram',
            source_id=f'inbound:{message_id}', timestamp=timestamp)
        if event is None:
            raise ValueError('telegram_owner_history_unavailable')
        digest = _digest([payload, integration_id, provider_user_id, chat_id, provider_message_id,
                          update_id, owner['address_id'], owner['address_updated_at'],
                          owner['integration_updated_at'], owner['credential_generation'], event_id])
        conn.execute('''INSERT INTO v2_owner_telegram_ingress VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (message_id, owner['user_id'], integration_id, provider_user_id, chat_id, provider_message_id,
             update_id, owner['address_id'], owner['address_updated_at'], owner['integration_updated_at'],
             owner['credential_generation'], event_id, digest, update_digest))
        conn.execute("INSERT INTO v2_owner_telegram_receipts (message_id,status) VALUES (?,'pending')", (message_id,))
        return message_id


def verified_telegram_origin_locked(conn, message_id):
    """Revalidate origin, mapping and queued event on the caller's connection."""
    try:
        origin = conn.execute('SELECT * FROM v2_owner_telegram_ingress WHERE message_id=?', (message_id,)).fetchone()
        queued = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()
        if origin is None or queued is None:
            return None
        owner = _allowed(conn, origin['integration_id'], origin['provider_user_id'], origin['chat_id'])
        if (owner is None or owner['user_id'] != origin['owner_id'] or owner['address_id'] != origin['address_id']
                or owner['address_updated_at'] != origin['address_updated_at']
                or owner['integration_updated_at'] != origin['integration_updated_at']
                or owner['credential_generation'] != origin['credential_generation'] or queued['user_id'] != origin['owner_id']):
            return None
        payload = [queued[key] for key in ('message_id', 'timestamp', 'prompt', 'user_id', 'project_id', 'tag',
                                           'correlation_id', 'memory_session_id', 'metadata_json')]
        expected = _digest([payload, origin['integration_id'], origin['provider_user_id'], origin['chat_id'],
                            origin['provider_message_id'], origin['update_id'], origin['address_id'],
                            origin['address_updated_at'], origin['integration_updated_at'],
                            origin['credential_generation'], origin['history_event_id']])
        update_digest = _digest([origin['integration_id'], origin['update_id'], origin['provider_user_id'],
                                 origin['chat_id'], origin['provider_message_id'], queued['prompt']])
        if (not hmac.compare_digest(origin['payload_sha256'], expected)
                or not hmac.compare_digest(origin['update_sha256'], update_digest)):
            return None
        event = conn.execute('SELECT * FROM v2_conversation_events WHERE event_id=?', (origin['history_event_id'],)).fetchone()
        if (event is None or event['owner_user_id'] != origin['owner_id'] or event['role'] != 'user'
                or event['source'] != 'telegram' or event['source_message_id'] != f'inbound:{message_id}'
                or event['content'] != queued['prompt'] or event['project_id'] != queued['project_id']
                or event['memory_session_id'] != queued['memory_session_id']
                or event['created_at'] != queued['timestamp'] or event['observed_at'] != queued['timestamp']
                or event['speaker_id'] != origin['owner_id'] or event['schema_version'] != 1
                or event['shared_with'] != '[]' or event['provenance'] != 'recorded'
                or event['visibility'] != 'private'):
            return None
        return {key: origin[key] for key in ('message_id', 'owner_id', 'integration_id', 'provider_user_id',
                                               'chat_id', 'provider_message_id', 'update_id')}
    except (ValueError, TypeError, KeyError, IndexError, sqlite3.Error):
        return None


@contextmanager
def telegram_owner_claim_transaction(queue, receipt, task, *, clock=None, _finalize=None):
    """Fence a Telegram-origin claim; it is not a Desktop owner claim."""
    if (type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing
            or not isinstance(receipt, ClaimReceipt) or task.task_id != receipt.message_id
            or task.message_id != receipt.message_id or task.user != receipt.user_id
            or task.intelligence_engine != 'single_loop_experimental'
            or type(task.intelligence_schema_version) is not int or task.intelligence_schema_version != 1):
        raise ClaimFenceUnavailable('telegram_owner_claim_unavailable')
    def check(conn):
        origin = verified_telegram_origin_locked(conn, receipt.message_id)
        row = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (receipt.message_id,)).fetchone()
        if (origin is None or origin['owner_id'] != receipt.user_id or row is None or task.goal != row['prompt']
                or any(getattr(task, key, None) != row[key] for key in ('project_id', 'tag', 'correlation_id', 'memory_session_id'))
                or json.dumps(task.metadata, sort_keys=True) != row['metadata_json']):
            raise ClaimFenceUnavailable('telegram_owner_claim_unavailable')
    with queue._connect() as conn:
        with fenced_claim_transaction(conn, receipt, clock=clock or (lambda: datetime.now(timezone.utc)), _finalize=_finalize):
            check(conn)
            yield conn
            check(conn)


def verify_telegram_owner_claim(queue, receipt, task, *, clock=None):
    """Point-in-time check; later actions still need their own claim fence."""
    try:
        with telegram_owner_claim_transaction(queue, receipt, task, clock=clock):
            return True
    except Exception:
        return False
