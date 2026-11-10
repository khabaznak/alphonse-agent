"""Opt-in SQLite hold primitive for superseded inbound queue claims.

No production caller installs or invokes this module. There is deliberately no
release, acknowledgment, requeue, continuation or effect-reconciliation API.
A hold is a fail-closed recovery state, not a completed task or permission.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import re
import sqlite3
from typing import Callable

from .claim_fence import ClaimFenceUnavailable, ClaimReceipt, fenced_claim_transaction

_ENGINE_SCHEMAS = {'tactical_v2': 2, 'hierarchical_v3': 3, 'single_loop_experimental': 1}
_TABLE = '''CREATE TABLE IF NOT EXISTS v2_supersession_holds (
    message_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    claim_generation TEXT NOT NULL,
    task_id TEXT NOT NULL,
    engine TEXT NOT NULL,
    observed_version INTEGER NOT NULL,
    current_version INTEGER NOT NULL,
    checkpoint_status TEXT NOT NULL,
    created_at TEXT NOT NULL
) STRICT'''
_UPDATE_TRIGGER = '''CREATE TRIGGER IF NOT EXISTS v2_supersession_hold_prevent_update
    BEFORE UPDATE ON v2_inbound_messages
    WHEN EXISTS (SELECT 1 FROM v2_supersession_holds WHERE message_id=OLD.message_id)
    BEGIN SELECT RAISE(ABORT, 'supersession_hold_requires_reconciliation'); END'''
_DELETE_TRIGGER = '''CREATE TRIGGER IF NOT EXISTS v2_supersession_hold_prevent_delete
    BEFORE DELETE ON v2_inbound_messages
    WHEN EXISTS (SELECT 1 FROM v2_supersession_holds WHERE message_id=OLD.message_id)
    BEGIN SELECT RAISE(ABORT, 'supersession_hold_requires_reconciliation'); END'''
_INSERT_TRIGGER = '''CREATE TRIGGER IF NOT EXISTS v2_supersession_hold_prevent_insert
    BEFORE INSERT ON v2_inbound_messages
    WHEN EXISTS (SELECT 1 FROM v2_supersession_holds WHERE message_id=NEW.message_id)
    BEGIN SELECT RAISE(ABORT, 'supersession_hold_requires_reconciliation'); END'''


@dataclass(frozen=True)
class HoldResult:
    status: str


@dataclass(frozen=True)
class HoldInspection:
    """Owner-visible status only; never contains claim or evidence contents."""
    status: str
    checkpoint_status: str | None = None
    next_step: str | None = None


def inspect_owned_hold(conn: sqlite3.Connection, session: object, *, message_id: str) -> HoldInspection:
    """Inspect an active owner's existing hold; never clear, resume or mutate it.

    Only trusted session/task code may construct AccountSession. This function
    does not authenticate an IPC user or authorize message delivery. The caller
    must recheck actor and audience when presenting the result. Incomplete,
    cross-user, deleted or inconsistent state is deliberately opaque.
    """
    from alphonse.agent_v2.accounts import AccountSession
    if (not isinstance(session, AccountSession) or not isinstance(session.actor_user_id, str)
            or not session.actor_user_id.strip() or not isinstance(message_id, str) or not message_id.strip()):
        raise ValueError('hold_inspection_arguments_invalid')
    if conn.in_transaction:
        raise ClaimFenceUnavailable('hold_transaction_already_open')
    unavailable = HoldInspection('unavailable')
    try:
        conn.execute('BEGIN IMMEDIATE')
        main = next((row for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), None)
        if main is None or not main[2]:
            raise ClaimFenceUnavailable('hold_shared_file_required')
        # A lookalike or partially installed hold cannot confer recovery state.
        normalize = lambda text: ' '.join(text.replace('IF NOT EXISTS ', '').split()).casefold()
        for kind, name, definition in [('table', 'v2_supersession_holds', _TABLE),
                                       ('trigger', 'v2_supersession_hold_prevent_insert', _INSERT_TRIGGER),
                                       ('trigger', 'v2_supersession_hold_prevent_update', _UPDATE_TRIGGER),
                                       ('trigger', 'v2_supersession_hold_prevent_delete', _DELETE_TRIGGER)]:
            schema = conn.execute('SELECT type,sql FROM main.sqlite_master WHERE name=?', (name,)).fetchone()
            if schema is None or schema[0] != kind or not isinstance(schema[1], str) or normalize(schema[1]) != normalize(definition):
                raise ClaimFenceUnavailable('hold_schema_unavailable')
        actor = session.actor_user_id
        if conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
            conn.rollback()
            return unavailable
        row = conn.execute('''SELECT h.user_id,h.claim_generation,h.task_id,h.engine,h.observed_version,
                h.current_version,q.user_id,q.status,q.claim_generation,c.owner_id,c.status,c.version,c.task_state_json,h.checkpoint_status
            FROM main.v2_supersession_holds h
            JOIN main.v2_inbound_messages q ON q.message_id=h.message_id
            JOIN main.v2_task_checkpoints c ON c.task_id=h.task_id
            WHERE h.message_id=? AND h.user_id=?''', (message_id, actor)).fetchone()
        if row is None:
            conn.rollback()
            return unavailable
        try:
            task = json.loads(row[12])
        except (TypeError, ValueError):
            conn.rollback()
            return unavailable
        metadata = task.get('metadata') if isinstance(task, dict) else None
        engine = row[3]
        schema_version = _ENGINE_SCHEMAS.get(engine)
        if (row[0] != actor or row[6] != actor or row[9] != actor or row[7] != 'processing'
                or not isinstance(row[1], str) or re.fullmatch(r'[0-9a-f]{32}', row[1]) is None or row[1] != row[8]
                or type(row[4]) is not int or type(row[5]) is not int or type(row[11]) is not int
                or row[4] < 1 or row[4] > row[5] or row[5] > row[11]
                or row[13] not in {'queued', 'running', 'waiting_user', 'done', 'failed', 'cancelled'}
                or (row[4] == row[5] and row[13] != 'cancelled')
                or row[10] not in {'queued', 'running', 'waiting_user', 'done', 'failed', 'cancelled'}
                or schema_version is None or not isinstance(task, dict) or not isinstance(metadata, dict)
                or task.get('task_id') != row[2] or task.get('message_id') != message_id
                or task.get('user') != actor or task.get('intelligence_engine') != engine
                or type(task.get('intelligence_schema_version')) is not int
                or task['intelligence_schema_version'] != schema_version
                or metadata.get('intelligence_engine', engine) != engine
                or type(metadata.get('intelligence_schema_version', schema_version)) is not int
                or metadata.get('intelligence_schema_version', schema_version) != schema_version):
            conn.rollback()
            return unavailable
        result = HoldInspection('held', str(row[10]), 'requires_authorized_reconciliation')
        conn.commit()
        return result
    except sqlite3.Error:
        if conn.in_transaction:
            conn.rollback()
        raise ClaimFenceUnavailable('hold_inspection_storage_unavailable') from None
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise


def install_hold_schema(conn: sqlite3.Connection) -> None:
    """Explicit one-time opt-in migration; refuses missing generation support.

    Existing old workers must be drained. No production startup invokes this.
    SQLite triggers fail closed if an old writer ignores the hold table. The
    current queue's reclamation path must exclude held messages before use.
    """
    if conn.in_transaction:
        raise ClaimFenceUnavailable('hold_transaction_already_open')
    try:
        conn.execute('BEGIN IMMEDIATE')
        main = next((row for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), None)
        columns = {row[1] for row in conn.execute('PRAGMA main.table_info(v2_inbound_messages)')}
        if main is None or not main[2] or not {'message_id', 'claim_generation'} <= columns:
            raise ClaimFenceUnavailable('hold_schema_prerequisites_unavailable')
        # A pre-existing table or trigger with these names must be exactly the
        # one this module owns; silently adopting a conflicting schema is unsafe.
        for kind, name, sql in [('table', 'v2_supersession_holds', _TABLE),
                                ('trigger', 'v2_supersession_hold_prevent_insert', _INSERT_TRIGGER),
                                ('trigger', 'v2_supersession_hold_prevent_update', _UPDATE_TRIGGER),
                                ('trigger', 'v2_supersession_hold_prevent_delete', _DELETE_TRIGGER)]:
            existing = conn.execute('SELECT type,sql FROM main.sqlite_master WHERE name=?', (name,)).fetchone()
            if existing is not None:
                normalize = lambda text: ' '.join(text.replace('IF NOT EXISTS ', '').split()).casefold()
                if existing[0] != kind or normalize(existing[1]) != normalize(sql):
                    raise ClaimFenceUnavailable('hold_schema_conflict')
            else:
                conn.execute(sql)
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise


def park_superseded(conn: sqlite3.Connection, claim: ClaimReceipt, *, task_id: str,
                    actor: str, engine: str, observed_version: int,
                    clock: Callable[[], datetime]) -> HoldResult:
    """Park an exact direct-owner claim only when a newer/cancelled row proves it stale.

    The same transaction reads the live claim and actor-bound checkpoint, then
    writes the hold. It never changes the checkpoint or an effect receipt.
    Cross-user question answers, missing identity, malformed state, unproven
    supersession and unsupported stores fail closed and retain the lease.
    """
    if (not isinstance(claim, ClaimReceipt) or not isinstance(task_id, str) or not task_id.strip()
            or not isinstance(actor, str) or not actor.strip() or engine not in _ENGINE_SCHEMAS
            or type(observed_version) is not int or observed_version < 1 or not callable(clock)):
        raise ValueError('supersession_hold_arguments_invalid')
    if actor != claim.user_id:
        return HoldResult('unavailable')
    try:
        main = next((row for row in conn.execute('PRAGMA database_list') if row[1] == 'main'), None)
        if main is None or not main[2]:
            raise ClaimFenceUnavailable('hold_shared_file_required')
        with fenced_claim_transaction(conn, claim, clock=clock):
            existing = conn.execute('''SELECT user_id,claim_generation,task_id,engine,observed_version
                FROM main.v2_supersession_holds WHERE message_id=?''', (claim.message_id,)).fetchone()
            if existing is not None:
                return HoldResult('already_held' if tuple(existing) ==
                                  (actor, claim.claim_generation, task_id, engine, observed_version) else 'unavailable')
            row = conn.execute('''SELECT owner_id,status,version,task_state_json
                FROM main.v2_task_checkpoints WHERE task_id=?''', (task_id,)).fetchone()
            if row is None or row[0] != actor or type(row[2]) is not int or row[2] < observed_version:
                return HoldResult('unavailable')
            try:
                task = json.loads(row[3])
            except (TypeError, ValueError):
                return HoldResult('unavailable')
            metadata = task.get('metadata') if isinstance(task, dict) else None
            schema = _ENGINE_SCHEMAS[engine]
            if (not isinstance(task, dict) or not isinstance(metadata, dict)
                    or task.get('task_id') != task_id or task.get('message_id') != claim.message_id
                    or task.get('user') != actor or task.get('intelligence_engine') != engine
                    or type(task.get('intelligence_schema_version')) is not int
                    or task['intelligence_schema_version'] != schema
                    or metadata.get('intelligence_engine', engine) != engine
                    or type(metadata.get('intelligence_schema_version', schema)) is not int
                    or metadata.get('intelligence_schema_version', schema) != schema):
                return HoldResult('unavailable')
            status, version = row[1], row[2]
            if status not in {'queued', 'running', 'waiting_user', 'done', 'failed', 'cancelled'}:
                return HoldResult('unavailable')
            if version == observed_version and status != 'cancelled':
                return HoldResult('not_superseded')
            stamp = clock()
            if not isinstance(stamp, datetime) or stamp.tzinfo is None or stamp.utcoffset() is None:
                raise ClaimFenceUnavailable('hold_clock_invalid')
            conn.execute('''INSERT INTO main.v2_supersession_holds
                (message_id,user_id,claim_generation,task_id,engine,observed_version,current_version,checkpoint_status,created_at)
                VALUES (?,?,?,?,?,?,?,?,?)''', (claim.message_id, actor, claim.claim_generation, task_id,
                engine, observed_version, version, status, stamp.astimezone(timezone.utc).isoformat()))
            return HoldResult('held')
    except sqlite3.Error:
        raise ClaimFenceUnavailable('supersession_hold_storage_unavailable') from None
