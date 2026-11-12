"""Opt-in staged outbox experiment; never install in a household database.

Staging has no release, delivery, deletion, or reconciliation operation. Older
schema-rebuilding writers can bypass the triggers; user deletion is blocked.
It is not a mixed-version or production publication protocol.
"""
from __future__ import annotations

from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
import sqlite3
from typing import Callable
from uuid import NAMESPACE_URL, uuid5

from .checkpoint_claim_fence import experimental_checkpoint_claim_transaction
from .outbox import SQLiteOutboundStore, _OUTBOX_STATUSES, _outbox_status_schema, _validate_outbox_rebuild
from alphonse.agent_v2.core.messages.claim_fence import ClaimReceipt, ClaimFenceUnavailable
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.single_loop_delivery_reservation import ACTION_KEY, DeliveryBinding


_OLD = "'pending', 'claimed', 'retry_wait', 'delivered', 'failed', 'unconfirmed'"
_NEW = _OLD + ", 'staged'"
_GUARDS = {
    'experimental_staged_outbox_update': """CREATE TRIGGER experimental_staged_outbox_update BEFORE UPDATE ON v2_outbox
        WHEN OLD.status='staged' BEGIN SELECT RAISE(ABORT, 'staged_outbox_requires_reconciliation'); END""",
    'experimental_staged_outbox_delete': """CREATE TRIGGER experimental_staged_outbox_delete BEFORE DELETE ON v2_outbox
        WHEN OLD.status='staged' BEGIN SELECT RAISE(ABORT, 'staged_outbox_requires_reconciliation'); END""",
    'experimental_staged_outbox_insert': """CREATE TRIGGER experimental_staged_outbox_insert BEFORE INSERT ON v2_outbox
        WHEN EXISTS (SELECT 1 FROM v2_outbox WHERE outbox_message_id=NEW.outbox_message_id AND status='staged')
        BEGIN SELECT RAISE(ABORT, 'staged_outbox_requires_reconciliation'); END""",
}


def _normalize(sql: str) -> str:
    # SQL string literals are case-sensitive; do not case-fold the schema.
    return ' '.join(sql.split())


@lru_cache(maxsize=1)
def _canonical_outbox_sql() -> str:
    # Derive the exact current outbox schema from a disposable in-memory store.
    store = SQLiteOutboundStore(':memory:')
    try:
        row = store._memory_connection.execute("SELECT sql FROM sqlite_master WHERE name='v2_outbox' AND type='table'").fetchone()
        return _normalize(row[0])
    finally:
        store._memory_connection.close()


def _validate_guards(conn: sqlite3.Connection) -> None:
    table = conn.execute("SELECT sql FROM main.sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()
    expected = _canonical_outbox_sql().replace(_normalize(_OLD), _normalize(_NEW))
    if table is None or not isinstance(table[0], str) or _normalize(table[0]) != expected:
        raise ClaimFenceUnavailable('staged_outbox_schema_unavailable')
    names = {row[0] for row in conn.execute(
        "SELECT name FROM main.sqlite_master WHERE type='trigger' AND tbl_name='v2_outbox'")}
    if names != set(_GUARDS):
        raise ClaimFenceUnavailable('staged_outbox_guards_unavailable')
    for name, expected in _GUARDS.items():
        row = conn.execute("SELECT type,sql FROM main.sqlite_master WHERE name=?", (name,)).fetchone()
        if row is None or row[0] != 'trigger' or not isinstance(row[1], str) or _normalize(row[1]) != _normalize(expected):
            raise ClaimFenceUnavailable('staged_outbox_guards_unavailable')


def install_experimental_staging(store: SQLiteOutboundStore) -> None:
    """Rebuild a canonical disposable outbox; old current runtimes refuse it.

    Explicit opt-in only. This is destructive to neither rows nor known indexes,
    but is not safe with running or older binaries or with household deletion.
    Refuse unfamiliar schemas and dependent objects instead of guessing.
    """
    if not isinstance(store, SQLiteOutboundStore) or store._memory_connection is not None:
        raise ValueError('staged_outbox_file_store_required')
    with store._schema_connect() as conn:
        conn.execute('BEGIN IMMEDIATE')
        if _outbox_status_schema(conn) != _OUTBOX_STATUSES:
            raise ValueError('staged_outbox_schema_unsupported')
        _validate_outbox_rebuild(conn)
        row = conn.execute("SELECT sql FROM main.sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()
        sql = str(row[0] or '') if row else ''
        if sql.count(_OLD) != 1 or _normalize(sql) != _canonical_outbox_sql():
            raise ValueError('staged_outbox_schema_unsupported')
        if any(conn.execute('SELECT 1 FROM main.sqlite_master WHERE name=?', (name,)).fetchone() for name in _GUARDS):
            raise ValueError('staged_outbox_schema_conflict')
        indexes = [row[0] for row in conn.execute("SELECT sql FROM main.sqlite_master WHERE type='index' AND tbl_name='v2_outbox' AND sql IS NOT NULL")]
        conn.execute('ALTER TABLE main.v2_outbox RENAME TO experimental_old_outbox')
        conn.execute(sql.replace(_OLD, _NEW))
        conn.execute('INSERT INTO main.v2_outbox SELECT * FROM main.experimental_old_outbox')
        conn.execute('DROP TABLE main.experimental_old_outbox')
        for index in indexes:
            conn.execute(index)
        for trigger in _GUARDS.values():
            conn.execute(trigger)


def stage_experimental_response(store: SQLiteOutboundStore, claim: ClaimReceipt, *,
                                checkpoints: ExperimentalCheckpointStore,
                                task_id: str, actor: str, expected_version: int,
                                integration_id: str, provider_key: str, channel_target: str,
                                message: str, idempotency_key: str,
                                clock: Callable[[], datetime]) -> str:
    """Persist an inert direct-owner response, not permission to publish it.

    This trusted, test-only entry point checks that the persisted pending action
    binds the exact message bytes and destination. It does not authenticate or
    authorize the actor, verify a grant, or deliver. A binding and idempotency
    key cannot prove that an effect has or has not occurred.
    """
    values = (task_id, actor, integration_id, provider_key, channel_target, message, idempotency_key)
    if (not isinstance(store, SQLiteOutboundStore) or store._memory_connection is not None
            or not isinstance(checkpoints, ExperimentalCheckpointStore)
            or any(not isinstance(value, str) or not value.strip() or len(value) > 12000 for value in values)):
        raise ValueError('staged_outbox_arguments_invalid')
    # Encode the pair unambiguously: neither task nor key forbids delimiters.
    identity = str(uuid5(NAMESPACE_URL, 'alphonse-experimental-stage:' +
                         json.dumps([task_id, idempotency_key], ensure_ascii=True, separators=(',', ':'))))
    with store._connect() as conn:
        with experimental_checkpoint_claim_transaction(conn, claim, task_id=task_id, actor=actor,
                                                       expected_version=expected_version, clock=clock):
            _validate_guards(conn)
            if conn.execute('SELECT 1 FROM main.v2_users WHERE user_id=? AND is_active=1', (actor,)).fetchone() is None:
                raise ClaimFenceUnavailable('staged_outbox_actor_unavailable')
            row = checkpoints._row(conn, task_id)
            raw = checkpoints._validate_identity(row, task_id, actor, expected_version)
            state, budget = checkpoints._decode(raw, task_id, actor)
            action = raw['metadata'].get(ACTION_KEY)
            if not isinstance(action, dict) or type(action.get('schema_version')) is not int:
                raise ClaimFenceUnavailable('staged_outbox_binding_unavailable')
            try:
                keys = {'action_id', 'integration_id', 'provider_key', 'channel_target', 'recipient_user_id',
                        'message_digest', 'idempotency_key', 'grant_revision'}
                if set(action) != keys | {'schema_version', 'operation', 'task_id', 'actor'}:
                    raise ValueError('invalid binding')
                binding = DeliveryBinding(**{key: action[key] for key in keys})
            except (ValueError, TypeError, KeyError):
                raise ClaimFenceUnavailable('staged_outbox_binding_unavailable') from None
            pending = budget.to_dict()['reservations'].get(state.inflight)
            if (action['schema_version'] != 1 or action['operation'] != 'delivery'
                    or action['task_id'] != task_id or action['actor'] != actor
                    or state.inflight != binding.action_id or not isinstance(pending, dict)
                    or pending['kind'] != 'tool' or pending['status'] != 'reserved'
                    or binding.integration_id != integration_id or binding.provider_key != provider_key
                    or binding.channel_target != channel_target
                    or binding.recipient_user_id != actor or binding.idempotency_key != idempotency_key
                    or binding.message_digest != hashlib.sha256(message.encode('utf-8')).hexdigest()):
                raise ClaimFenceUnavailable('staged_outbox_binding_unavailable')
            metadata = json.dumps({'experimental_staged': True, 'checkpoint_version': expected_version,
                                   'action_id': binding.action_id,
                                   'idempotency_key_sha256': hashlib.sha256(idempotency_key.encode('utf-8')).hexdigest()},
                                  sort_keys=True)
            existing = conn.execute('''SELECT integration_id,provider_key,channel_target,message,kind,
                audience_user_id,correlation_id,status,provider_message_id,reply_to_provider_message_id,
                task_id,question_id,project_id,metadata_json,claimed_at,delivered_at,created_at,updated_at,
                attempt_count,lease_owner,lease_expires_at,next_attempt_at,last_error,max_attempts
                FROM main.v2_outbox WHERE outbox_message_id=?''', (identity,)).fetchone()
            if existing is not None:
                # A restart may repeat an insertion whose commit was not seen.
                # Return its ID only if the entire inert record still matches
                # this exact action; never reinterpret a legacy, changed, or
                # possibly dispatched row as successful staging.
                row = tuple(existing)
                expected = (integration_id, provider_key, channel_target, message, 'response', actor,
                            '', 'staged', '', '', task_id, '', '', metadata, '', '')
                if (row[:16] != expected or not isinstance(row[16], str) or not row[16]
                        or row[17] != row[16] or row[18:] != (0, '', '', '', '', 5)):
                    raise ValueError('staged_outbox_idempotency_conflict')
                return identity
            stamp = datetime.now(timezone.utc).isoformat()
            conn.execute('''INSERT INTO main.v2_outbox(outbox_message_id,integration_id,provider_key,channel_target,
                message,kind,audience_user_id,correlation_id,status,task_id,metadata_json,created_at,updated_at)
                VALUES (?,?,?,?,?,'response',?,'','staged',?,?,?,?)''',
                (identity, integration_id, provider_key, channel_target, message, actor, task_id, metadata, stamp, stamp))
    return identity
