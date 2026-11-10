"""Unwired, read-only classification of an actor-bound superseded checkpoint.

This module never acknowledges, requeues or releases a message and never
mutates checkpoint state. It is not the #216 handoff: safe release requires an
atomic queue transition plus explicit recovery and reconciliation contracts.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import sqlite3
from typing import Callable
from datetime import datetime

from .claim_fence import ClaimReceipt, ClaimFenceUnavailable, fenced_claim_transaction

_ENGINE_SCHEMAS = {'tactical_v2': 2, 'hierarchical_v3': 3, 'single_loop_experimental': 1}


@dataclass(frozen=True)
class SupersessionObservation:
    """Safe status only: no task data, claim token, or evidence contents."""
    classification: str
    checkpoint_version: int | None = None


def inspect_supersession(conn: sqlite3.Connection, claim: ClaimReceipt, *, task_id: str,
                         actor: str, engine: str, observed_version: int,
                         clock: Callable[[], datetime]) -> SupersessionObservation:
    """Classify one direct-owner checkpoint under a fenced queue transaction.

    Caller must authenticate the actor, obtain `claim` at acquisition, and use
    the shared file-backed queue/checkpoint database. This initial contract does
    not support delegated answers, scheduled handoffs or actor changes: they
    return `unavailable` without leaking whether a task exists. `cancelled`
    requires later effect reconciliation; no classification authorizes a queue
    acknowledgment, model call, effect, or automatic resume.
    """
    if (not isinstance(claim, ClaimReceipt) or not isinstance(task_id, str) or not task_id.strip()
            or not isinstance(actor, str) or not actor.strip() or not isinstance(engine, str)
            or engine not in _ENGINE_SCHEMAS or type(observed_version) is not int or observed_version < 1):
        raise ValueError('supersession_arguments_invalid')
    if claim.user_id != actor:
        return SupersessionObservation('unavailable')
    try:
        # Never accept separate in-memory stores or an attached side database.
        main = next((row for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), None)
        if main is None or not main[2]:
            raise ClaimFenceUnavailable('supersession_shared_file_required')
        with fenced_claim_transaction(conn, claim, clock=clock):
            row = conn.execute('''SELECT owner_id,status,version,task_state_json
                FROM main.v2_task_checkpoints WHERE task_id=?''', (task_id,)).fetchone()
            if row is None or row[0] != actor or type(row[2]) is not int or row[2] < observed_version:
                return SupersessionObservation('unavailable')
            try:
                task = json.loads(row[3])
            except (TypeError, ValueError):
                return SupersessionObservation('unavailable')
            metadata = task.get('metadata') if isinstance(task, dict) else None
            if (not isinstance(task, dict) or not isinstance(metadata, dict) or task.get('task_id') != task_id
                    or task.get('message_id') != claim.message_id or task.get('user') != actor
                    or task.get('intelligence_engine') != engine
                    or type(task.get('intelligence_schema_version')) is not int
                    or task['intelligence_schema_version'] != _ENGINE_SCHEMAS[engine]
                    or metadata.get('intelligence_engine', engine) != engine
                    or type(metadata.get('intelligence_schema_version', _ENGINE_SCHEMAS[engine])) is not int
                    or metadata.get('intelligence_schema_version', _ENGINE_SCHEMAS[engine]) != _ENGINE_SCHEMAS[engine]):
                return SupersessionObservation('unavailable')
            status, version = row[1], row[2]
            if status == 'cancelled':
                return SupersessionObservation('cancelled', version)
            if status == 'waiting_user':
                return SupersessionObservation('waiting_user', version)
            if status in {'done', 'failed'}:
                return SupersessionObservation('terminal', version)
            if status in {'running', 'queued'}:
                return SupersessionObservation('newer_active' if version > observed_version
                                               else 'queued' if status == 'queued' else 'current', version)
            return SupersessionObservation('unavailable')
    except sqlite3.Error:
        raise ClaimFenceUnavailable('supersession_storage_unavailable') from None
