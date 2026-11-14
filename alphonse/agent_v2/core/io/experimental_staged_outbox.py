"""Opt-in staged outbox experiment; never install in a household database.

Staging has no release, delivery, deletion, or reconciliation operation. Older
schema-rebuilding writers can bypass the triggers; user deletion is blocked.
It is not a mixed-version or production publication protocol.
"""
from __future__ import annotations

from datetime import datetime, timezone
from contextlib import closing, contextmanager
from functools import lru_cache
import hashlib
import json
import os
import sqlite3
from pathlib import Path
from typing import Callable
from uuid import NAMESPACE_URL, uuid5

from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.skill_bindings import SkillBindings
from .checkpoint_claim_fence import experimental_checkpoint_claim_transaction
from .outbox import SQLiteOutboundStore, _OUTBOX_STATUSES, _outbox_status_schema, _validate_outbox_rebuild
from alphonse.agent_v2.core.messages.claim_fence import ClaimReceipt, ClaimFenceUnavailable
from alphonse.agent_v2.core.intelligence.v3.single_loop_checkpoint import ExperimentalCheckpointStore
from alphonse.agent_v2.core.intelligence.v3.single_loop_delivery_reservation import ACTION_KEY, ACTION_SCHEMA_VERSION, DeliveryBinding
from alphonse.agent_v2.core.intelligence.v3.single_loop_managed_delivery import MANAGED_PREFLIGHT_KEY, _capture, managed_record_matches


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


def rollback_empty_experimental_staging(db_path: str | Path) -> None:
    """Restore the canonical outbox only when no staged rows exist.

    This is an explicit recovery operation for disposable experiments, not a
    live downgrade protocol. All writers must be stopped; older binaries and
    direct DDL remain unfenced. It never deletes or releases staged messages.
    A foreign schema, dependent object or any remaining staged row aborts the
    transaction without modifying the outbox, its guards, or its data.
    """
    if not isinstance(db_path, (str, Path)) or str(db_path) in ('', ':memory:'):
        raise ValueError('staged_outbox_file_store_required')
    path = Path(db_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('staged_outbox_file_store_required')
    conn = sqlite3.connect(path, timeout=5)
    try:
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('BEGIN IMMEDIATE')
        _validate_guards(conn)
        if conn.execute("""SELECT 1 FROM main.v2_outbox WHERE status NOT IN
                ('pending','claimed','retry_wait','delivered','failed','unconfirmed','staged') LIMIT 1""").fetchone():
            raise ClaimFenceUnavailable('staged_outbox_schema_unavailable')
        if conn.execute("SELECT 1 FROM main.v2_outbox WHERE status='staged' LIMIT 1").fetchone():
            raise ClaimFenceUnavailable('staged_outbox_reconciliation_required')
        for name in _GUARDS:
            conn.execute(f'DROP TRIGGER main.{name}')
        # Validate after removing only our known guards. Any other dependent
        # view, trigger, foreign key or unfamiliar index aborts and restores
        # the dropped triggers as part of the same SQLite transaction.
        _validate_outbox_rebuild(conn)
        row = conn.execute("SELECT sql FROM main.sqlite_master WHERE type='table' AND name='v2_outbox'").fetchone()
        sql = str(row[0] or '') if row else ''
        if sql.count(_NEW) != 1:
            raise ClaimFenceUnavailable('staged_outbox_schema_unavailable')
        indexes = [row[0] for row in conn.execute(
            "SELECT sql FROM main.sqlite_master WHERE type='index' AND tbl_name='v2_outbox' AND sql IS NOT NULL")]
        conn.execute('ALTER TABLE main.v2_outbox RENAME TO experimental_old_outbox')
        conn.execute(sql.replace(_NEW, _OLD))
        conn.execute('INSERT INTO main.v2_outbox SELECT * FROM main.experimental_old_outbox')
        conn.execute('DROP TABLE main.experimental_old_outbox')
        for index in indexes:
            conn.execute(index)
        conn.commit()
    except BaseException:
        if conn.in_transaction:
            conn.rollback()
        raise
    finally:
        conn.close()


@contextmanager
def _stage_connection(target: SQLiteOutboundStore | Path):
    if isinstance(target, SQLiteOutboundStore):
        with target._connect() as conn:
            yield conn
        return
    # No regular outbox constructor or schema migration is used at restart.
    # The path must already exist; mode=rw cannot create a missing database.
    with closing(sqlite3.connect(target.as_uri() + '?mode=rw', uri=True, timeout=5)) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute('PRAGMA foreign_keys = ON')
        yield conn


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
    if not isinstance(store, SQLiteOutboundStore) or store._memory_connection is not None:
        raise ValueError('staged_outbox_arguments_invalid')
    return _stage_experimental_response(store, claim, checkpoints=checkpoints, task_id=task_id,
        actor=actor, expected_version=expected_version, integration_id=integration_id,
        provider_key=provider_key, channel_target=channel_target, message=message,
        idempotency_key=idempotency_key, clock=clock)


def restage_experimental_response_after_restart(db_path: str | Path, claim: ClaimReceipt, *,
                                checkpoints: ExperimentalCheckpointStore,
                                task_id: str, actor: str, expected_version: int,
                                integration_id: str, provider_key: str, channel_target: str,
                                message: str, idempotency_key: str,
                                clock: Callable[[], datetime]) -> str:
    """Restage a disposable experiment after restart without opening a live outbox.

    This explicitly opted-in path validates the same schema, guards, live claim,
    actor, checkpoint and action as initial staging, and only returns an
    existing matching row. A missing row requires reconciliation. Managed
    actions require a future authority-revalidated restart path. It never
    dispatches or authorizes delivery. It is not a household restart or migration
    protocol.
    """
    if not isinstance(db_path, (str, Path)) or str(db_path) in ('', ':memory:'):
        raise ValueError('staged_outbox_file_store_required')
    path = Path(db_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('staged_outbox_file_store_required')
    return _stage_experimental_response(path, claim, checkpoints=checkpoints, task_id=task_id,
        actor=actor, expected_version=expected_version, integration_id=integration_id,
        provider_key=provider_key, channel_target=channel_target, message=message,
        idempotency_key=idempotency_key, clock=clock, require_existing=True)


def stage_managed_experimental_response(store: SQLiteOutboundStore, claim: ClaimReceipt, *,
                                        checkpoints: ExperimentalCheckpointStore,
                                        bindings: SkillBindings, session: AccountSession,
                                        task_id: str, expected_version: int, message: str,
                                        clock: Callable[[], datetime]) -> str:
    """Revalidate managed access before staging an inert direct-owner response.

    The caller must independently authenticate the trusted session. This opt-in
    experiment cannot send, release or settle. A successful stage is not an
    authorization token: skill files and the credential vault can change outside
    SQLite, and must be checked again at any future dispatch.
    """
    if (not isinstance(store, SQLiteOutboundStore) or store._memory_connection is not None
            or not isinstance(checkpoints, ExperimentalCheckpointStore)
            or not isinstance(bindings, SkillBindings) or not isinstance(session, AccountSession)
            or not isinstance(claim, ClaimReceipt) or not isinstance(session.actor_user_id, str)
            or not session.actor_user_id.strip() or session.actor_user_id != claim.user_id
            or not callable(clock)):
        raise ValueError('managed_stage_arguments_invalid')
    return _stage_managed_experimental_response(store, claim, checkpoints=checkpoints, bindings=bindings,
        session=session, task_id=task_id, expected_version=expected_version, message=message, clock=clock)


def restage_managed_experimental_response_after_restart(db_path: str | Path, claim: ClaimReceipt, *,
                                        checkpoints: ExperimentalCheckpointStore,
                                        bindings: SkillBindings, session: AccountSession,
                                        task_id: str, expected_version: int, message: str,
                                        clock: Callable[[], datetime]) -> str:
    """Revalidate a managed experiment and find its existing inert row.

    The caller must independently authenticate the trusted session. Never
    creates an intent during restart and cannot authorize dispatch. This is not
    a household recovery or mixed-writer protocol.
    """
    if (not isinstance(db_path, (str, Path)) or str(db_path) in ('', ':memory:')
            or not isinstance(checkpoints, ExperimentalCheckpointStore)
            or not isinstance(bindings, SkillBindings) or not isinstance(session, AccountSession)
            or not isinstance(claim, ClaimReceipt) or not isinstance(session.actor_user_id, str)
            or not session.actor_user_id.strip() or session.actor_user_id != claim.user_id
            or not callable(clock)):
        raise ValueError('managed_stage_arguments_invalid')
    path = Path(db_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError('staged_outbox_file_store_required')
    return _stage_managed_experimental_response(path, claim, checkpoints=checkpoints, bindings=bindings,
        session=session, task_id=task_id, expected_version=expected_version, message=message, clock=clock,
        require_existing=True)


def _stage_managed_experimental_response(target: SQLiteOutboundStore | Path, claim: ClaimReceipt, *,
                                        checkpoints: ExperimentalCheckpointStore,
                                        bindings: SkillBindings, session: AccountSession,
                                        task_id: str, expected_version: int, message: str,
                                        clock: Callable[[], datetime], require_existing: bool = False) -> str:
    actor = session.actor_user_id
    try:
        target_path = target.db_path if isinstance(target, SQLiteOutboundStore) else target
        if not (os.path.samefile(target_path, checkpoints.store.db_path)
                and os.path.samefile(target_path, bindings.integrations.db_path)
                and os.path.samefile(target_path, bindings.users.db_path)):
            raise ClaimFenceUnavailable('managed_stage_unavailable')
        # One monotonic observer covers both fenced transactions. It cannot
        # detect time that advanced and moved back between observations.
        from alphonse.agent_v2.core.messages.claim_fence import _time
        previous = None
        def checked_clock():
            nonlocal previous
            now = _time(clock())
            if previous is not None and now < previous:
                raise ClaimFenceUnavailable('managed_stage_clock_reversed')
            previous = now
            return now
        with _stage_connection(target) as conn:
            with experimental_checkpoint_claim_transaction(conn, claim, task_id=task_id, actor=actor,
                                                           expected_version=expected_version, clock=checked_clock):
                _validate_guards(conn)
                row = checkpoints._row(conn, task_id)
                raw = checkpoints._validate_identity(row, task_id, actor, expected_version)
                state, budget = checkpoints._decode(raw, task_id, actor)
                action = raw['metadata'].get(ACTION_KEY)
                keys = set(DeliveryBinding.__dataclass_fields__)
                if (not isinstance(action, dict) or set(action) != keys | {'schema_version', 'operation', 'task_id', 'actor'}
                        or type(action['schema_version']) is not int or action['schema_version'] != ACTION_SCHEMA_VERSION
                        or action['operation'] != 'delivery' or action['task_id'] != task_id or action['actor'] != actor):
                    raise ClaimFenceUnavailable('managed_stage_unavailable')
                binding = DeliveryBinding(**{key: action[key] for key in keys})
                pending = budget.to_dict()['reservations'].get(state.inflight)
                if (state.inflight != binding.action_id or not isinstance(pending, dict)
                        or pending.get('kind') != 'tool' or pending.get('status') != 'reserved'
                        or not managed_record_matches(raw, task_id=task_id, actor=actor, binding=binding)):
                    raise ClaimFenceUnavailable('managed_stage_unavailable')
                managed = raw['metadata'][MANAGED_PREFLIGHT_KEY]
                before, visible = _capture(conn, actor=actor, skill_id=managed['skill_id'], role=managed['role'],
                                           binding_id=managed['binding_id'], action=binding)
                if any(visible[key] != managed[key] for key in ('binding_revision', 'required_scopes', 'required_executables')):
                    raise ClaimFenceUnavailable('managed_stage_unavailable')
        handle = bindings.resolve_account(session, skill_id=managed['skill_id'], name=managed['role'],
            operation=managed['operation'], resources=managed['resources'],
            recipient_user_ids=managed['recipient_user_ids'], project_id=managed['project_id'],
            expected_binding_id=managed['binding_id'], expected_connection_id=binding.integration_id,
            expected_grant_id=binding.grant_id, expected_grant_revision=binding.grant_revision)
        if (handle.binding_id != managed['binding_id'] or handle.connection_id != binding.integration_id
                or handle.grant_id != binding.grant_id or handle.provider_key != binding.provider_key
                or handle.connection_revision != binding.connection_revision or handle.grant_revision != binding.grant_revision
                or handle.operation != managed['operation'] or list(handle.resources) != managed['resources']
                or list(handle.recipient_user_ids) != managed['recipient_user_ids']):
            raise ClaimFenceUnavailable('managed_stage_unavailable')
        return _stage_experimental_response(target, claim, checkpoints=checkpoints, task_id=task_id,
            actor=actor, expected_version=expected_version, integration_id=binding.integration_id,
            provider_key=binding.provider_key, channel_target=binding.channel_target, message=message,
            idempotency_key=binding.idempotency_key, clock=checked_clock,
            managed_authority=(raw, before), require_existing=require_existing)
    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError, PermissionError) as exc:
        raise ClaimFenceUnavailable('managed_stage_unavailable') from exc


def _stage_experimental_response(target: SQLiteOutboundStore | Path, claim: ClaimReceipt, *,
                                checkpoints: ExperimentalCheckpointStore,
                                task_id: str, actor: str, expected_version: int,
                                integration_id: str, provider_key: str, channel_target: str,
                                message: str, idempotency_key: str,
                                clock: Callable[[], datetime], require_existing: bool = False,
                                managed_authority: tuple | None = None) -> str:
    values = (task_id, actor, integration_id, provider_key, channel_target, message, idempotency_key)
    if (not isinstance(checkpoints, ExperimentalCheckpointStore)
            or any(not isinstance(value, str) or not value.strip() or len(value) > 12000 for value in values)):
        raise ValueError('staged_outbox_arguments_invalid')
    # Encode the pair unambiguously: neither task nor key forbids delimiters.
    identity = str(uuid5(NAMESPACE_URL, 'alphonse-experimental-stage:' +
                         json.dumps([task_id, idempotency_key], ensure_ascii=True, separators=(',', ':'))))
    with _stage_connection(target) as conn:
        with experimental_checkpoint_claim_transaction(conn, claim, task_id=task_id, actor=actor,
                                                       expected_version=expected_version, clock=clock):
            _validate_guards(conn)
            # Checkpoint methods below read through this connection. Reject an
            # adapter bound to a different database instead of silently using
            # the target's checkpoint table under a misleading store identity.
            try:
                main_path = next(row[2] for row in conn.execute('PRAGMA database_list') if row[1] == 'main')
                if not main_path or not os.path.samefile(main_path, checkpoints.store.db_path):
                    raise ClaimFenceUnavailable('staged_outbox_checkpoint_store_mismatch')
            except (OSError, ValueError, TypeError, AttributeError, StopIteration):
                raise ClaimFenceUnavailable('staged_outbox_checkpoint_store_mismatch') from None
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
                        'message_digest', 'idempotency_key', 'grant_id', 'grant_revision', 'connection_revision'}
                if set(action) != keys | {'schema_version', 'operation', 'task_id', 'actor'}:
                    raise ValueError('invalid binding')
                binding = DeliveryBinding(**{key: action[key] for key in keys})
            except (ValueError, TypeError, KeyError):
                raise ClaimFenceUnavailable('staged_outbox_binding_unavailable') from None
            pending = budget.to_dict()['reservations'].get(state.inflight)
            if (action['schema_version'] != ACTION_SCHEMA_VERSION or action['operation'] != 'delivery'
                    or action['task_id'] != task_id or action['actor'] != actor
                    or state.inflight != binding.action_id or not isinstance(pending, dict)
                    or pending['kind'] != 'tool' or pending['status'] != 'reserved'
                    or binding.integration_id != integration_id or binding.provider_key != provider_key
                    or binding.channel_target != channel_target
                    or binding.recipient_user_id != actor or binding.idempotency_key != idempotency_key
                    or binding.message_digest != hashlib.sha256(message.encode('utf-8')).hexdigest()):
                raise ClaimFenceUnavailable('staged_outbox_binding_unavailable')
            # A managed preflight, when recorded, must describe this exact
            # action. Shape matching is not a current authorization check.
            if (MANAGED_PREFLIGHT_KEY in raw['metadata']
                    and not managed_record_matches(raw, task_id=task_id, actor=actor, binding=binding)):
                raise ClaimFenceUnavailable('staged_outbox_binding_unavailable')
            if MANAGED_PREFLIGHT_KEY in raw['metadata'] and managed_authority is None:
                # A shape check is not authority. Do not let the generic
                # synthetic staging/restart path bypass managed resolution.
                raise ClaimFenceUnavailable('managed_stage_authority_required')
            if managed_authority is not None:
                original, authority = managed_authority
                if raw != original or not managed_record_matches(raw, task_id=task_id, actor=actor, binding=binding):
                    raise ClaimFenceUnavailable('managed_stage_unavailable')
                managed = raw['metadata'][MANAGED_PREFLIGHT_KEY]
                current, visible = _capture(conn, actor=actor, skill_id=managed['skill_id'], role=managed['role'],
                                            binding_id=managed['binding_id'], action=binding)
                if current != authority or any(visible[key] != managed[key] for key in ('binding_revision', 'required_scopes', 'required_executables')):
                    raise ClaimFenceUnavailable('managed_stage_unavailable')
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
                # this exact action. The row alone does not reveal whether an
                # external effect occurred; callers must not interpret this
                # result as proof of non-dispatch or permission to retry.
                row = tuple(existing)
                expected = (integration_id, provider_key, channel_target, message, 'response', actor,
                            '', 'staged', '', '', task_id, '', '', metadata, '', '')
                if (row[:16] != expected or not isinstance(row[16], str) or not row[16]
                        or row[17] != row[16] or row[18:] != (0, '', '', '', '', 5)):
                    raise ValueError('staged_outbox_idempotency_conflict')
                return identity
            # A restart lookup cannot prove why the original row is missing.
            # Do not manufacture a fresh staged intent during recovery; its
            # absence must be resolved against durable action/effect evidence.
            if require_existing:
                raise ClaimFenceUnavailable('staged_outbox_reconciliation_required')
            stamp = datetime.now(timezone.utc).isoformat()
            conn.execute('''INSERT INTO main.v2_outbox(outbox_message_id,integration_id,provider_key,channel_target,
                message,kind,audience_user_id,correlation_id,status,task_id,metadata_json,created_at,updated_at)
                VALUES (?,?,?,?,?,'response',?,'','staged',?,?,?,?)''',
                (identity, integration_id, provider_key, channel_target, message, actor, task_id, metadata, stamp, stamp))
    return identity
