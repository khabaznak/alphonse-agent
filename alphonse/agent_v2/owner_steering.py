"""Bounded steering accepted only against a live model reservation.

Draft foundation: callers must not expose this until settlement and projection
are wired atomically and verified across cancellation and recovery.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
from datetime import datetime, timezone

from alphonse.agent_v2.core.messages.claim_fence import ClaimReceipt, _read_claim
from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue, _owner_ingress_matches
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import STATE_KEY
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_candidate import CandidateState
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import TaskBudget


def ensure_table(conn):
    conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_steering (
        sequence INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL,
        owner_id TEXT NOT NULL, peer_uid INTEGER NOT NULL, message_id TEXT NOT NULL,
        claim_digest TEXT NOT NULL, reservation_id TEXT NOT NULL, content TEXT NOT NULL,
        content_digest TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('accepted','included_in_planning','removed'))
    ) STRICT''')


def _has_dependents(conn, owner, message_id):
    if (conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_held_followups'").fetchone()
            and conn.execute("SELECT 1 FROM v2_owner_held_followups WHERE owner_id=? AND active_message_id=? AND status IN ('queued','released') LIMIT 1",
                             (owner, message_id)).fetchone()):
        return True
    rows = conn.execute('SELECT metadata_json FROM v2_inbound_messages WHERE user_id=? AND message_id!=?',
                        (owner, message_id)).fetchall()
    for row in rows:
        try:
            metadata = json.loads(row[0])
            if any(isinstance(metadata.get(key), dict) and metadata[key].get('message_id') == message_id
                   for key in ('owner_followup', 'owner_clarification')):
                return True
        except (ValueError, TypeError, AttributeError):
            return True
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_task_checkpoints'").fetchone():
        rows = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE owner_id=? AND task_id!=?',
                            (owner, message_id)).fetchall()
        for row in rows:
            try:
                metadata = json.loads(row[0])['metadata']
                if any(isinstance(metadata.get(key), dict) and metadata[key].get('message_id') == message_id
                       for key in ('owner_followup', 'owner_clarification')):
                    return True
            except (ValueError, TypeError, KeyError, AttributeError):
                return True
    return False


def submit(queue, *, owner, peer_uid, daemon_id, message_id, request_id, content):
    """Accept only while this daemon's validated model reservation is live."""
    if (type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing
            or queue.db_path == ':memory:' or type(peer_uid) is not int or peer_uid != os.getuid()
            or not isinstance(owner, str) or not owner or not isinstance(daemon_id, str) or not daemon_id
            or not isinstance(message_id, str) or re.fullmatch(r'[0-9a-f]{32}', message_id) is None
            or not isinstance(request_id, str) or re.fullmatch(r'[0-9a-f]{32}', request_id) is None
            or not isinstance(content, str) or not content.strip() or len(content.encode('utf-8')) > 4096):
        raise ValueError('owner_steer_invalid')
    content = content.strip()
    digest = hashlib.sha256((request_id + '\0' + content).encode()).hexdigest()
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        admin = conn.execute('''SELECT 1 FROM v2_users u JOIN v2_user_settings s
            ON s.key='admin_user_id' AND s.value=u.user_id WHERE u.user_id=?
            AND u.is_active=1 AND u.role IN ('admin','owner')''', (owner,)).fetchone()
        if admin is None:
            raise PermissionError('owner_steer_unavailable')
        ensure_table(conn)
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_held_followups'").fetchone() and conn.execute('SELECT 1 FROM v2_owner_held_followups WHERE id=?', (request_id,)).fetchone():
            raise ValueError('owner_steer_invalid')
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_composer_tombstones'").fetchone() and conn.execute('SELECT 1 FROM v2_owner_composer_tombstones WHERE id=?', (request_id,)).fetchone():
            raise ValueError('owner_draft_closed')
        if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_composer_drafts'").fetchone():
            draft = conn.execute('SELECT * FROM v2_owner_composer_drafts WHERE id=?', (request_id,)).fetchone()
            if draft:
                if draft['resolution'] == 'not_submitted':
                    raise ValueError('owner_draft_closed')
                if (draft['owner_id'], draft['peer_uid'], draft['active_message_id'], draft['prompt']) != (owner, peer_uid, message_id, content):
                    raise ValueError('owner_draft_invalid')
        existing = conn.execute('SELECT * FROM v2_owner_steering WHERE id=?', (request_id,)).fetchone()
        if existing is not None:
            if (existing['owner_id'] != owner or existing['peer_uid'] != peer_uid
                    or existing['message_id'] != message_id
                    or not hmac.compare_digest(existing['content_digest'], digest)):
                raise ValueError('owner_steer_invalid')
            return {'id': request_id, 'status': existing['status']}
        row = conn.execute('''SELECT q.*,o.owner_id,o.peer_uid,o.payload_sha256,o.history_version,o.history_event_id,
            c.owner_id AS checkpoint_owner,c.task_state_json FROM v2_inbound_messages q
            JOIN v2_trusted_owner_ingress o ON o.message_id=q.message_id
            JOIN v2_task_checkpoints c ON c.task_id=q.message_id WHERE q.message_id=?''', (message_id,)).fetchone()
        try:
            if (row is None or row['owner_id'] != owner or row['checkpoint_owner'] != owner
                    or row['user_id'] != owner or row['peer_uid'] != peer_uid or row['lease_owner'] != daemon_id):
                return {'id': request_id, 'status': 'not_accepted'}
            payload = [row[key] for key in ('message_id', 'timestamp', 'prompt', 'user_id', 'project_id',
                'tag', 'correlation_id', 'memory_session_id', 'metadata_json')]
            if not _owner_ingress_matches(conn, row, payload):
                return {'id': request_id, 'status': 'not_accepted'}
            receipt = ClaimReceipt(message_id, owner, daemon_id, row['claim_generation'])
            _read_claim(conn, receipt, datetime.now(timezone.utc))
            raw = json.loads(row['task_state_json'])
            metadata = raw['metadata']
            envelope = metadata[STATE_KEY]
            state = CandidateState.restore(envelope['state'])
            budget = TaskBudget.from_dict(envelope['budget'])
            marker = metadata.get('owner_read_pending_action')
            if (raw['task_id'] != message_id or raw['message_id'] != message_id or raw['user'] != owner
                    or raw['goal'] != row['prompt'] or state.task_id != message_id or state.actor != owner
                    or state.status != 'running' or not state.inflight or not isinstance(marker, dict)
                    or marker != {'reservation_id': state.inflight, 'kind': 'model', 'tool_id': '',
                                  'arguments': None, 'claim_digest': _claim_digest(receipt)}
                    or budget.to_dict()['reservations'].get(state.inflight, {}).get('status') != 'reserved'):
                return {'id': request_id, 'status': 'not_accepted'}
        except Exception:
            return {'id': request_id, 'status': 'not_accepted'}
        if conn.execute('SELECT count(*) FROM v2_owner_steering WHERE owner_id=? AND message_id=?', (owner, message_id)).fetchone()[0] >= 20:
            raise ValueError('owner_steer_limit')
        try:
            _read_claim(conn, receipt, datetime.now(timezone.utc))
        except Exception:
            return {'id': request_id, 'status': 'not_accepted'}
        conn.execute('''INSERT INTO v2_owner_steering
            (id,owner_id,peer_uid,message_id,claim_digest,reservation_id,content,content_digest,status)
            VALUES (?,?,?,?,?,?,?,?, 'accepted')''',
            (request_id, owner, peer_uid, message_id, _claim_digest(receipt), state.inflight, content, digest))
        _read_claim(conn, receipt, datetime.now(timezone.utc))
        return {'id': request_id, 'status': 'accepted'}


def list_steering(queue, *, owner, peer_uid):
    """List only the current local owner's steering records."""
    if (type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing
            or queue.db_path == ':memory:' or type(peer_uid) is not int or peer_uid != os.getuid()
            or not isinstance(owner, str) or not owner):
        raise PermissionError('owner_steer_unavailable')
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('''SELECT 1 FROM v2_users u JOIN v2_user_settings s
            ON s.key='admin_user_id' AND s.value=u.user_id WHERE u.user_id=?
            AND u.is_active=1 AND u.role IN ('admin','owner')''', (owner,)).fetchone() is None:
            raise PermissionError('owner_steer_unavailable')
        ensure_table(conn)
        rows = conn.execute('''SELECT id,message_id,content,content_digest,status FROM v2_owner_steering
            WHERE owner_id=? AND peer_uid=? AND status!='removed' ORDER BY sequence''', (owner, peer_uid)).fetchall()
        result = []
        for row in rows:
            active = conn.execute('SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (row['message_id'],)).fetchone() is not None
            dependent = _has_dependents(conn, owner, row['message_id'])
            can_remove = not active and not dependent
            digest = hashlib.sha256((row['id'] + '\0' + row['content']).encode()).hexdigest()
            if not hmac.compare_digest(row['content_digest'], digest):
                result.append({'id': row['id'], 'active_message_id': row['message_id'],
                               'prompt': '', 'status': 'needs_attention', 'can_remove': can_remove,
                               'remove_blocked_reason': 'dependent_followups' if dependent else ('request_unresolved' if active else None)})
            else:
                result.append({'id': row['id'], 'active_message_id': row['message_id'],
                               'prompt': row['content'], 'status': row['status'], 'can_remove': can_remove,
                               'remove_blocked_reason': 'dependent_followups' if dependent else ('request_unresolved' if active else None)})
        return {'items': result}


def remove_steering(queue, *, owner, peer_uid, steering_id):
    """Clear a steer only after its request has left the inbound queue."""
    if (type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing
            or queue.db_path == ':memory:' or type(peer_uid) is not int or peer_uid != os.getuid()
            or not isinstance(owner, str) or not owner or not isinstance(steering_id, str)
            or re.fullmatch(r'[0-9a-f]{32}', steering_id) is None):
        raise ValueError('owner_steer_invalid')
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if conn.execute('''SELECT 1 FROM v2_users u JOIN v2_user_settings s
            ON s.key='admin_user_id' AND s.value=u.user_id WHERE u.user_id=?
            AND u.is_active=1 AND u.role IN ('admin','owner')''', (owner,)).fetchone() is None:
            raise PermissionError('owner_steer_unavailable')
        ensure_table(conn)
        row = conn.execute('SELECT * FROM v2_owner_steering WHERE id=? AND owner_id=? AND peer_uid=?',
                           (steering_id, owner, peer_uid)).fetchone()
        if row is None:
            raise ValueError('owner_steer_unavailable')
        if row['status'] != 'removed' and conn.execute(
                'SELECT 1 FROM v2_inbound_messages WHERE message_id=?', (row['message_id'],)).fetchone():
            raise ValueError('owner_steer_request_unresolved')
        if row['status'] != 'removed' and _has_dependents(conn, owner, row['message_id']):
            raise ValueError('owner_steer_dependent_followups')
        conn.execute("UPDATE v2_owner_steering SET status='removed',content='' WHERE id=?", (steering_id,))
        return {'id': steering_id, 'status': 'removed'}
