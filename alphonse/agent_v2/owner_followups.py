"""Durable, owner-scoped holding area for follow-up drafts.

Held follow-ups are deliberately not execution requests. A future release path
must bind predecessor history and create trusted ingress atomically; a restart,
failed predecessor, or uncertain delivery never dispatches these rows.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from datetime import datetime, timezone
from uuid import uuid4

from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue

_ID = re.compile(r'[0-9a-f]{32}\Z')


def _id(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError('owner_followup_invalid')
    return value


class OwnerFollowups:
    def __init__(self, queue, owner, peer_uid):
        if (type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing
                or queue.db_path == ':memory:' or type(peer_uid) is not int or peer_uid != os.getuid()
                or not isinstance(owner, str) or not owner):
            raise PermissionError('owner_followup_unavailable')
        self.queue, self.owner, self.peer_uid = queue, owner, peer_uid

    def _authorize(self, conn):
        if conn.execute('''SELECT 1 FROM v2_users u JOIN v2_user_settings s
                ON s.key='admin_user_id' AND s.value=u.user_id
                WHERE u.user_id=? AND u.role IN ('admin','owner') AND u.is_active=1''', (self.owner,)).fetchone() is None:
            raise PermissionError('owner_followup_unavailable')
        conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_held_followups (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
            owner_id TEXT NOT NULL, peer_uid INTEGER NOT NULL, active_message_id TEXT NOT NULL,
            prompt TEXT NOT NULL, prompt_digest TEXT NOT NULL, created_at TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('queued','removed','released'))) STRICT''')

    def add(self, active_message_id, prompt, request_id):
        active_message_id, request_id = _id(active_message_id), _id(request_id)
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt.encode('utf-8')) > 16384:
            raise ValueError('owner_followup_invalid')
        prompt = prompt.strip()
        digest = hashlib.sha256((request_id + '\0' + prompt).encode('utf-8')).hexdigest()
        with self.queue._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._authorize(conn)
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_steering'").fetchone() and conn.execute('SELECT 1 FROM v2_owner_steering WHERE id=?', (request_id,)).fetchone():
                raise ValueError('owner_followup_invalid')
            existing = conn.execute('SELECT * FROM v2_owner_held_followups WHERE id=?', (request_id,)).fetchone()
            if existing is not None:
                if (existing['owner_id'] != self.owner or existing['peer_uid'] != self.peer_uid
                        or existing['active_message_id'] != active_message_id
                        or not hmac.compare_digest(existing['prompt_digest'], digest)):
                    raise ValueError('owner_followup_invalid')
                return {'id': request_id, 'status': existing['status']}
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_composer_tombstones'").fetchone() and conn.execute('SELECT 1 FROM v2_owner_composer_tombstones WHERE id=?', (request_id,)).fetchone():
                raise ValueError('owner_followup_closed')
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_composer_drafts'").fetchone():
                draft = conn.execute('SELECT * FROM v2_owner_composer_drafts WHERE id=?', (request_id,)).fetchone()
                if draft:
                    if draft['resolution'] == 'not_submitted':
                        raise ValueError('owner_followup_closed')
                    if (draft['owner_id'], draft['peer_uid'], draft['active_message_id'], draft['prompt']) != (self.owner, self.peer_uid, active_message_id, prompt):
                        raise ValueError('owner_followup_invalid')
            origin = conn.execute('''SELECT 1 FROM v2_trusted_owner_ingress o
                JOIN v2_inbound_messages q ON q.message_id=o.message_id AND q.user_id=o.owner_id
                WHERE o.message_id=? AND o.owner_id=? AND o.peer_uid=? AND q.status IN ('pending','processing')''',
                (active_message_id, self.owner, self.peer_uid)).fetchone()
            if origin is None:
                return {'id': request_id, 'status': 'late'}
            count = conn.execute("SELECT count(*) FROM v2_owner_held_followups WHERE owner_id=? AND status='queued'", (self.owner,)).fetchone()[0]
            if count >= 100:
                raise ValueError('owner_followup_limit')
            conn.execute('''INSERT INTO v2_owner_held_followups
                (id,owner_id,peer_uid,active_message_id,prompt,prompt_digest,created_at,status)
                VALUES (?,?,?,?,?,?,?,'queued')''',
                (request_id, self.owner, self.peer_uid, active_message_id, prompt, digest, datetime.now(timezone.utc).isoformat()))
            return {'id': request_id, 'status': 'queued'}

    def list(self):
        with self.queue._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._authorize(conn)
            rows = conn.execute('''SELECT f.id,f.active_message_id,
                CASE WHEN f.status='released' THEN COALESCE(e.content,'') ELSE f.prompt END AS prompt,
                f.created_at,f.status FROM v2_owner_held_followups f
                LEFT JOIN v2_conversation_events e ON e.owner_user_id=f.owner_id
                  AND e.source_message_id=('inbound:' || f.id)
                WHERE f.owner_id=? AND f.peer_uid=? AND f.status IN ('queued','released')
                ORDER BY f.sequence''', (self.owner, self.peer_uid)).fetchall()
            items = [dict(row) for row in rows]
            from alphonse.agent_v2.core.intelligence.simple_loop.owner_clarification import bind_owner_followup
            for item in items:
                item['can_remove'] = item['status'] == 'queued'
                if item['status'] == 'queued':
                    prior = conn.execute('''SELECT id FROM v2_owner_held_followups
                        WHERE owner_id=? AND peer_uid=? AND active_message_id=? AND status='released'
                          AND sequence < (SELECT sequence FROM v2_owner_held_followups WHERE id=?)
                        ORDER BY sequence DESC LIMIT 1''',
                        (self.owner, self.peer_uid, item['active_message_id'], item['id'])).fetchone()
                    prior_id = prior['id'] if prior else item['active_message_id']
                    item['predecessor_message_id'] = prior_id
                    item['predecessor_active'] = conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (prior_id,)).fetchone() is not None
                    if not item['predecessor_active']:
                        try:
                            bind_owner_followup(conn, self.owner, prior_id)
                        except ValueError:
                            item['status'] = 'needs_attention'
                    continue
                pending = conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (item['id'],)).fetchone()
                if pending is not None:
                    continue
                try:
                    bind_owner_followup(conn, self.owner, item['id'])
                    item['status'] = 'completed'
                except ValueError:
                    item['status'] = 'needs_attention'
            return {'items': items}

    def remove(self, followup_id):
        followup_id = _id(followup_id)
        with self.queue._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._authorize(conn)
            row = conn.execute('SELECT status FROM v2_owner_held_followups WHERE id=? AND owner_id=? AND peer_uid=?',
                               (followup_id, self.owner, self.peer_uid)).fetchone()
            if row is None:
                raise ValueError('owner_followup_unavailable')
            if row['status'] == 'released':
                raise ValueError('owner_followup_already_released')
            conn.execute("UPDATE v2_owner_held_followups SET status='removed',prompt='' WHERE id=?", (followup_id,))
            return {'id': followup_id, 'status': 'removed'}

    def release_next(self):
        """Atomically stage only the oldest draft after a verified prior ACK.

        Failure, cancellation, uncertain delivery, an active ingress, or
        missing/revoked history keeps the draft held. This doesn't run a task.
        """
        from alphonse.agent_v2.core.intelligence.simple_loop.owner_clarification import bind_owner_followup
        from alphonse.agent_v2.core.messages.sqlite_queue import _claim_mode_conflict, _ensure_owner_history_columns, _owner_ingress_digest
        from alphonse.agent_v2.conversations import SQLiteConversationStore

        with self.queue._connect() as conn:
            conn.execute('BEGIN IMMEDIATE')
            self._authorize(conn)
            if _claim_mode_conflict(conn, fenced=True):
                return None
            row = conn.execute("SELECT * FROM v2_owner_held_followups WHERE owner_id=? AND peer_uid=? AND status='queued' ORDER BY sequence LIMIT 1",
                               (self.owner, self.peer_uid)).fetchone()
            if row is None:
                return None
            if conn.execute('''SELECT 1 FROM v2_inbound_messages q JOIN v2_trusted_owner_ingress o
                ON o.message_id=q.message_id WHERE o.owner_id=? AND q.status IN ('pending','processing') LIMIT 1''', (self.owner,)).fetchone():
                return None
            prior = conn.execute('''SELECT id FROM v2_owner_held_followups
                WHERE owner_id=? AND peer_uid=? AND active_message_id=? AND status='released' AND sequence<?
                ORDER BY sequence DESC LIMIT 1''', (self.owner, self.peer_uid, row['active_message_id'], row['sequence'])).fetchone()
            prior_id = prior['id'] if prior else row['active_message_id']
            try:
                binding = bind_owner_followup(conn, self.owner, prior_id)
            except ValueError:
                return None
            message_id, prompt = row['id'], row['prompt']
            if (not prompt or not hmac.compare_digest(row['prompt_digest'],
                    hashlib.sha256((message_id + '\0' + prompt).encode('utf-8')).hexdigest())):
                return None
            timestamp = datetime.now(timezone.utc).isoformat()
            metadata = json.dumps({'intelligence_engine': 'single_loop_experimental',
                'intelligence_schema_version': 1, 'routing_disposition': 'pdca_task',
                'owner_followup': binding}, sort_keys=True)
            payload = [message_id, timestamp, prompt, self.owner, '', '', '', '', metadata]
            _ensure_owner_history_columns(conn)
            conn.execute('''INSERT INTO v2_inbound_messages
                (message_id,sequence,timestamp,prompt,user_id,project_id,tag,correlation_id,memory_session_id,
                 metadata_json,queued_at,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,'pending')''',
                (message_id, time.time_ns(), timestamp, prompt, self.owner, '', '', '', '', metadata, timestamp))
            event = SQLiteConversationStore._record(conn, event_id=str(uuid4()), owner=self.owner,
                project='', session='', role='user', message=prompt, source='desktop',
                source_id='inbound:' + message_id, timestamp=timestamp)
            if event is None:
                raise ValueError('owner_followup_history_unavailable')
            digest = _owner_ingress_digest(payload, 1, event.event_id)
            conn.execute('''INSERT INTO v2_trusted_owner_ingress
                (message_id,owner_id,peer_uid,payload_sha256,created_at,history_version,history_event_id)
                VALUES (?,?,?,?,?,?,?)''', (message_id, self.owner, self.peer_uid, digest, timestamp, 1, event.event_id))
            conn.execute("UPDATE v2_owner_held_followups SET status='released',prompt='' WHERE id=? AND status='queued'", (message_id,))
            return message_id
