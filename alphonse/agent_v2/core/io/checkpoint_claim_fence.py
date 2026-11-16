"""Experimental direct-owner claim and checkpoint transaction boundary.

Unwired: this does not stage, publish, or deliver an outbox item. The current
outbox makes inserted rows dispatchable, so callers must not use this primitive
to enqueue messages. A live claim is not authorization, and this transaction
does not fence external I/O, different stores, or legacy writers after commit.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import json
import sqlite3
from typing import Callable, Iterator

from alphonse.agent_v2.core.messages.claim_fence import (
    ClaimFenceUnavailable, ClaimReceipt, fenced_claim_transaction,
)


@contextmanager
def experimental_checkpoint_claim_transaction(
    conn: sqlite3.Connection, claim: ClaimReceipt, *, task_id: str, actor: str,
    expected_version: int, clock: Callable[[], datetime],
) -> Iterator[sqlite3.Connection]:
    """Fence same-store work against a live claim and exact running checkpoint.

    This intentionally accepts only the direct-owner experimental engine; it
    does not support delegated answers or scheduled handoffs. The caller must
    separately authenticate and authorize the actor and any operation. The
    callback is trusted and must not perform external I/O or commit/rollback.
    """
    if (not isinstance(claim, ClaimReceipt) or not isinstance(task_id, str) or not task_id.strip()
            or not isinstance(actor, str) or not actor.strip() or type(expected_version) is not int
            or expected_version < 1 or not callable(clock)):
        raise ValueError('checkpoint_claim_arguments_invalid')
    if claim.user_id != actor:
        raise ClaimFenceUnavailable('checkpoint_claim_unavailable')
    try:
        main = next((row for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), None)
        if main is None or not main[2]:
            raise ClaimFenceUnavailable('checkpoint_claim_shared_file_required')
        with fenced_claim_transaction(conn, claim, clock=clock):
            original = conn.execute('''SELECT owner_id,status,version,task_state_json
                FROM main.v2_task_checkpoints WHERE task_id=?''', (task_id,)).fetchone()
            if (original is None or original[0] != actor or original[1] != 'running'
                    or type(original[2]) is not int or original[2] != expected_version):
                raise ClaimFenceUnavailable('checkpoint_claim_unavailable')
            try:
                task = json.loads(original[3])
            except (TypeError, ValueError):
                raise ClaimFenceUnavailable('checkpoint_claim_unavailable') from None
            metadata = task.get('metadata') if isinstance(task, dict) else None
            if (not isinstance(task, dict) or not isinstance(metadata, dict)
                    or task.get('task_id') != task_id or task.get('message_id') != claim.message_id
                    or task.get('user') != actor or task.get('status') != 'running'
                    or task.get('intelligence_engine') != 'single_loop_experimental'
                    or type(task.get('intelligence_schema_version')) is not int
                    or task['intelligence_schema_version'] != 1
                    or metadata.get('intelligence_engine', 'single_loop_experimental') != 'single_loop_experimental'
                    or type(metadata.get('intelligence_schema_version', 1)) is not int
                    or metadata.get('intelligence_schema_version', 1) != 1):
                raise ClaimFenceUnavailable('checkpoint_claim_unavailable')
            snapshot = tuple(conn.execute('SELECT * FROM main.v2_task_checkpoints WHERE task_id=?',
                                          (task_id,)).fetchone())
            yield conn
            current = conn.execute('SELECT * FROM main.v2_task_checkpoints WHERE task_id=?',
                                   (task_id,)).fetchone()
            if current is None or tuple(current) != snapshot:
                raise ClaimFenceUnavailable('checkpoint_changed_by_caller')
    except sqlite3.Error:
        raise ClaimFenceUnavailable('checkpoint_claim_storage_unavailable') from None
