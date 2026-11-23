"""Trusted-local access to a quiescent pause after its worker lease expires.

The local OS account is the authority in this development mode; same-UID Bash
can impersonate Desktop. This adapter cannot dispatch work, change the queue
claim, or access a checkpoint that is not already durably paused.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
import os

from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.experimental_claim_checkpoint import _claim_digest
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue, _owner_ingress_matches, _row_to_message
from .owner_atomic_checkpoint import OwnerAtomicCheckpoint, _decision_digest
from .single_loop_checkpoint import CheckpointConflict, ExperimentalCheckpointStore


def _identity(conn, receipt, task):
    row = conn.execute('''SELECT q.*,o.owner_id,o.peer_uid,o.payload_sha256,o.history_version,o.history_event_id
        FROM v2_inbound_messages q JOIN v2_trusted_owner_ingress o ON o.message_id=q.message_id
        JOIN v2_user_settings s ON s.key='admin_user_id' AND s.value=q.user_id
        JOIN v2_users u ON u.user_id=q.user_id AND u.role IN ('admin','owner') AND u.is_active=1
        WHERE q.message_id=?''', (receipt.message_id,)).fetchone()
    if (row is None or row['status'] != 'processing' or row['user_id'] != receipt.user_id
            or row['owner_id'] != receipt.user_id or row['peer_uid'] != os.getuid()
            or row['lease_owner'] != receipt.lease_owner or row['claim_generation'] != receipt.claim_generation):
        raise CheckpointConflict('owner_detached_pause_unavailable')
    payload = [row[k] for k in ('message_id','timestamp','prompt','user_id','project_id','tag',
                                  'correlation_id','memory_session_id','metadata_json')]
    if (not _owner_ingress_matches(conn, row, payload) or task.goal != row['prompt']
            or task.message_id != receipt.message_id or task.task_id != receipt.message_id):
        raise CheckpointConflict('owner_detached_pause_unavailable')
    exists = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_continuation_cancellations'").fetchone()
    if exists and conn.execute('SELECT 1 FROM v2_owner_continuation_cancellations WHERE message_id=?',
                               (receipt.message_id,)).fetchone():
        raise CheckpointConflict('owner_detached_pause_cancelled')
    return tuple(row)


def persist_continuation_cancellation(queue, message_id, owner_id):
    """Persist a local owner's cancellation veto before acknowledging it.

    The veto survives daemon restarts and fences detached choice and all
    continuation reclamation. It cannot prove effects did not occur.
    """
    if type(queue) is not SQLiteMessageQueue or not message_id or not owner_id:
        raise ValueError('owner_continuation_cancel_unavailable')
    with queue._connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        row = conn.execute('''SELECT q.user_id,o.peer_uid FROM v2_inbound_messages q
            JOIN v2_trusted_owner_ingress o ON o.message_id=q.message_id AND o.owner_id=q.user_id
            WHERE q.message_id=? AND q.user_id=?''', (message_id, owner_id)).fetchone()
        if row is None or row['peer_uid'] != os.getuid():
            raise CheckpointConflict('owner_continuation_cancel_unavailable')
        conn.execute('''CREATE TABLE IF NOT EXISTS v2_owner_continuation_cancellations (
            message_id TEXT PRIMARY KEY,owner_id TEXT NOT NULL,created_at TEXT NOT NULL)''')
        conn.execute('''INSERT INTO v2_owner_continuation_cancellations(message_id,owner_id,created_at)
            VALUES(?,?,?) ON CONFLICT(message_id) DO NOTHING''',
            (message_id, owner_id, datetime.now(timezone.utc).isoformat()))


class DetachedPausedCheckpoint(OwnerAtomicCheckpoint):
    @contextmanager
    def _transaction(self, *, _finalize=None):
        if _finalize is not None:
            raise CheckpointConflict('owner_detached_pause_unavailable')
        with self.queue._connect() as conn:
            try:
                conn.execute('BEGIN IMMEDIATE')
                before = _identity(conn, self.receipt, self.task)
                row = self.store._row(conn, self.task.task_id)
                raw = self.store._validate_identity(row, self.task.task_id, self.task.user, row['version'])
                pause = raw['metadata'].get('owner_continuation_pause')
                lineage = raw['metadata'].get('owner_continuation_reclaim')
                valid_lineage = (isinstance(lineage, dict) and isinstance(pause, dict)
                    and lineage.get('original_claim_digest') == pause.get('claim_digest')
                    and lineage.get('current_claim_digest') == _claim_digest(self.receipt)
                    and lineage.get('pause_sha256') == _decision_digest(pause)
                    and lineage.get('grant_sha256') == _decision_digest(raw['metadata'].get('owner_continuation_grant')))
                if not isinstance(pause, dict) or (pause.get('claim_digest') != _claim_digest(self.receipt)
                                                    and not valid_lineage):
                    raise CheckpointConflict('owner_detached_pause_unavailable')
                yield conn
                if _identity(conn, self.receipt, self.task) != before:
                    raise CheckpointConflict('owner_detached_pause_unavailable')
                after_row = self.store._row(conn, self.task.task_id)
                after = self.store._validate_identity(after_row, self.task.task_id, self.task.user, after_row['version'])
                if after['metadata'].get('owner_continuation_pause') != pause:
                    raise CheckpointConflict('owner_detached_pause_unavailable')
                conn.commit()
            except BaseException:
                if conn.in_transaction:
                    conn.rollback()
                raise


def open_detached_paused_checkpoint(queue, store, message_id):
    if (type(queue) is not SQLiteMessageQueue or not queue.experimental_claim_fencing
            or not isinstance(store, ExperimentalCheckpointStore) or not isinstance(message_id, str)):
        raise ValueError('owner_detached_pause_unavailable')
    with queue._connect() as conn:
        row = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (message_id,)).fetchone()
        if row is None or row['status'] != 'processing':
            raise CheckpointConflict('owner_detached_pause_unavailable')
        acquired = _row_to_message(row, include_claim=True)
        task = TaskState.from_message(acquired.message, message_id=message_id)
        task.task_id = message_id
        checkpoint = DetachedPausedCheckpoint(queue, store, task, acquired.claim_receipt,
                                               clock=lambda: datetime.now(timezone.utc))
    with checkpoint._transaction() as conn:
        row = store._row(conn, message_id)
        checkpoint.version = row['version']
    return checkpoint
