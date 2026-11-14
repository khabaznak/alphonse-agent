"""Unwired managed-account preflight for experimental delivery reservation.

This records which existing skill/account authority was checked. It is not an
authorization token: the skill package and credential vault are outside the
SQLite transaction and may change immediately after preflight. Dispatch and
settlement remain disabled until they reauthenticate and revalidate every
boundary and independently verify the effect.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
import sqlite3
from typing import Any, Callable

from alphonse.agent_v2.accounts import AccountSession, _deadline, _strings
from alphonse.agent_v2.skill_bindings import SkillBindings
from alphonse.agent_v2.core.messages.claim_fence import ClaimFenceUnavailable, ClaimReceipt
from .single_loop_checkpoint import Checkpoint, ExperimentalCheckpointStore
from .single_loop_delivery_reservation import DeliveryBinding, _reserve_delivery

MANAGED_PREFLIGHT_KEY = 'single_loop_managed_preflight'
_RECORD_KEYS = {'schema_version', 'actor', 'task_id', 'action_id', 'skill_id', 'role', 'binding_id',
                'binding_revision', 'required_scopes', 'required_executables', 'operation', 'resources',
                'project_id', 'recipient_user_ids', 'connection_id', 'provider_key', 'channel_target',
                'message_digest', 'idempotency_key', 'grant_id', 'grant_revision', 'connection_revision'}


def managed_record_matches(raw: dict, *, task_id: str, actor: str, binding: DeliveryBinding) -> bool:
    """Check stored preflight shape and action identity, never current authority."""
    record = raw.get('metadata', {}).get(MANAGED_PREFLIGHT_KEY)
    try:
        if (not isinstance(record, dict) or set(record) != _RECORD_KEYS
                or type(record['schema_version']) is not int or record['schema_version'] != 1
                or any(type(record[key]) is not int or record[key] < 1
                       for key in ('binding_revision', 'grant_revision', 'connection_revision'))
                or any(not isinstance(record[key], str) or not record[key].strip()
                       for key in ('skill_id', 'role', 'binding_id', 'operation'))
                or (record['project_id'] is not None and
                    (not isinstance(record['project_id'], str) or not record['project_id'].strip()))
                or (record['project_id'] or '') != str(raw.get('project_id') or '')
                or record['actor'] != actor or record['task_id'] != task_id
                or record['action_id'] != binding.action_id or record['connection_id'] != binding.integration_id
                or record['provider_key'] != binding.provider_key or record['channel_target'] != binding.channel_target
                or record['message_digest'] != binding.message_digest or record['idempotency_key'] != binding.idempotency_key
                or record['grant_id'] != binding.grant_id or record['grant_revision'] != binding.grant_revision
                or record['connection_revision'] != binding.connection_revision
                or record['recipient_user_ids'] != [binding.recipient_user_id]
                or binding.recipient_user_id != actor):
            return False
        for key in ('resources', 'required_scopes', 'required_executables'):
            if not isinstance(record[key], list) or _strings(record[key]) != record[key]:
                return False
        return bool(record['resources']) and binding.channel_target in record['resources']
    except (TypeError, ValueError, KeyError, AttributeError):
        return False


def _row(conn: sqlite3.Connection, sql: str, args: tuple) -> tuple:
    row = conn.execute(sql, args).fetchone()
    if row is None:
        raise PermissionError('delivery_managed_authority_unavailable')
    return tuple(row)


def _capture(conn: sqlite3.Connection, *, actor: str, skill_id: str, role: str,
             binding_id: str, action: DeliveryBinding) -> tuple[tuple, dict[str, Any]]:
    """Capture relevant database state; credential references never leave memory."""
    choice = _row(conn, '''SELECT binding_id FROM main.v2_skill_binding_choices
                            WHERE actor_id=? AND skill_id=? AND name=?''', (actor, skill_id, role))
    binding = _row(conn, '''SELECT binding_id,skill_id,owner_id,kind,target,requirements,
                            shared_with,enabled,revision FROM main.v2_skill_bindings WHERE binding_id=?''', (binding_id,))
    integration = _row(conn, '''SELECT integration_id,provider_key,enabled,connection_json,credential_ref,
                                credential_generation,revocation_pending,credential_deletion_pending
                                FROM main.v2_integrations WHERE integration_id=?''', (action.integration_id,))
    grant = _row(conn, '''SELECT grant_id,connection_id,revision,definition FROM main.v2_account_grants
                          WHERE grant_id=?''', (action.grant_id,))
    try:
        requirements, connection, grant_def = json.loads(binding[5]), json.loads(integration[3]), json.loads(grant[3])
        if not isinstance(requirements, dict) or not isinstance(connection, dict) or not isinstance(grant_def, dict):
            raise ValueError('invalid authority')
        scopes = _strings(requirements.get('scopes'))
        executables = _strings(requirements.get('executables'))
        owner = connection['owner_user_id']
        if (choice != (binding_id,) or binding[1] != skill_id or binding[3] != 'account'
                or binding[4] != action.integration_id or binding[7] != 1
                or type(binding[8]) is not int or binding[8] < 1
                or requirements.get('grant_id') != action.grant_id
                or type(connection.get('revision')) is not int or connection['revision'] != action.connection_revision
                or integration[1] != action.provider_key or integration[2] != 1
                or not isinstance(integration[4], str) or not integration[4]
                or type(integration[5]) is not int or integration[5] < 1
                or integration[6] != 0 or integration[7] != 0
                or grant[1] != action.integration_id or type(grant[2]) is not int
                or grant[2] != action.grant_revision or type(grant_def.get('revision')) is not int
                or grant_def['revision'] != grant[2]
                or grant_def.get('grant_id') != action.grant_id
                or not isinstance(owner, str) or not owner.strip()):
            raise ValueError('inconsistent authority')
        expiry = _deadline(grant_def['expires_at'])
        if expiry <= datetime.now(timezone.utc):
            raise ValueError('expired authority')
        principals = sorted({actor, action.recipient_user_id, binding[2], owner})
        users = tuple(_row(conn, 'SELECT user_id,role,is_active,updated_at FROM main.v2_users WHERE user_id=?', (user,))
                      for user in principals)
        if any(user[2] != 1 for user in users):
            raise ValueError('inactive principal')
    except (KeyError, TypeError, ValueError) as exc:
        raise PermissionError('delivery_managed_authority_unavailable') from exc
    snapshot = (choice, binding, integration, grant, users)
    visible = {'binding_revision': binding[8], 'required_scopes': scopes, 'required_executables': executables}
    return snapshot, visible


@dataclass(repr=False)
class _ManagedPreflight:
    snapshot: tuple = field(repr=False)
    record: dict[str, Any] = field(repr=False)

    def check_locked(self, conn: sqlite3.Connection, *, actor: str, task_id: str,
                     raw: dict, binding: DeliveryBinding) -> None:
        task_project = str(raw.get('project_id') or '')
        if (self.record['actor'] != actor or self.record['task_id'] != task_id
                or (self.record['project_id'] or '') != task_project
                or self.record['action_id'] != binding.action_id
                or self.record['connection_id'] != binding.integration_id
                or self.record['grant_id'] != binding.grant_id
                or self.record['connection_revision'] != binding.connection_revision
                or self.record['grant_revision'] != binding.grant_revision
                or self.record['provider_key'] != binding.provider_key
                or self.record['channel_target'] != binding.channel_target
                or self.record['message_digest'] != binding.message_digest
                or self.record['idempotency_key'] != binding.idempotency_key
                or self.record['recipient_user_ids'] != [binding.recipient_user_id]):
            raise PermissionError('delivery_managed_authority_unavailable')
        try:
            current, _ = _capture(conn, actor=actor, skill_id=self.record['skill_id'], role=self.record['role'],
                                  binding_id=self.record['binding_id'], action=binding)
        except sqlite3.Error as exc:
            raise PermissionError('delivery_managed_authority_unavailable') from exc
        if current != self.snapshot:
            raise PermissionError('delivery_managed_authority_changed')


def reserve_managed_delivery(store: ExperimentalCheckpointStore, claim: ClaimReceipt, *,
                             bindings: SkillBindings, session: AccountSession, skill_id: str, role: str,
                             operation: str, resources: list[str], project_id: str | None,
                             task_id: str, expected_version: int, binding: DeliveryBinding,
                             message: str, now_ms: int, clock: Callable[[], datetime]) -> Checkpoint:
    """Resolve actual skill/account access, then persist its scoped preflight.

    Only a direct owner, exact role and a shared file-backed database are
    supported. Revocation in that database between preflight and the reservation
    is detected under the queue claim's write lock. External skill files and the
    credential vault cannot be fenced by that lock; this record grants nothing.
    """
    if (not isinstance(store, ExperimentalCheckpointStore) or not isinstance(bindings, SkillBindings)
            or not isinstance(session, AccountSession) or not isinstance(claim, ClaimReceipt)
            or not isinstance(binding, DeliveryBinding) or not isinstance(skill_id, str) or not skill_id.strip()
            or not isinstance(role, str) or not role.strip() or not isinstance(operation, str) or not operation.strip()
            or (project_id is not None and (not isinstance(project_id, str) or not project_id.strip()))):
        raise ValueError('delivery_managed_arguments_invalid')
    actor = session.actor_user_id
    if not isinstance(actor, str) or not actor.strip() or actor != claim.user_id or actor != binding.recipient_user_id:
        raise PermissionError('delivery_managed_authority_unavailable')
    normalized_resources = _strings(resources)
    # This first slice supports only an exact destination named as a resource.
    # Do not infer that an authorized folder or account covers another channel.
    if not normalized_resources or binding.channel_target not in normalized_resources:
        raise ValueError('delivery_managed_destination_not_in_resources')
    try:
        if not (os.path.samefile(store.store.db_path, bindings.integrations.db_path)
                and os.path.samefile(store.store.db_path, bindings.users.db_path)):
            raise PermissionError('delivery_managed_shared_store_required')
        with store.store._connect() as conn:
            conn.execute('BEGIN')
            selected = _row(conn, '''SELECT binding_id FROM main.v2_skill_binding_choices
                                      WHERE actor_id=? AND skill_id=? AND name=?''', (actor, skill_id, role))[0]
            if not isinstance(selected, str) or not selected:
                raise PermissionError('delivery_managed_authority_unavailable')
            snapshot, visible = _capture(conn, actor=actor, skill_id=skill_id, role=role,
                                         binding_id=selected, action=binding)
        handle = bindings.resolve_account(session, skill_id=skill_id, name=role, operation=operation,
            resources=normalized_resources, recipient_user_ids=[binding.recipient_user_id], project_id=project_id,
            expected_binding_id=selected, expected_connection_id=binding.integration_id,
            expected_grant_id=binding.grant_id, expected_grant_revision=binding.grant_revision)
        if (handle.binding_id != selected or handle.connection_id != binding.integration_id
                or handle.grant_id != binding.grant_id or handle.provider_key != binding.provider_key
                or handle.connection_revision != binding.connection_revision
                or handle.grant_revision != binding.grant_revision or handle.operation != operation
                or list(handle.resources) != normalized_resources
                or list(handle.recipient_user_ids) != [binding.recipient_user_id]):
            raise PermissionError('delivery_managed_authority_unavailable')
    except (OSError, sqlite3.Error, KeyError, TypeError, ValueError) as exc:
        raise PermissionError('delivery_managed_authority_unavailable') from exc
    record = {'schema_version': 1, 'actor': actor, 'task_id': task_id, 'action_id': binding.action_id,
              'skill_id': skill_id, 'role': role, 'binding_id': selected, **visible,
              'operation': operation, 'resources': normalized_resources, 'project_id': project_id,
              'recipient_user_ids': [binding.recipient_user_id], 'connection_id': binding.integration_id,
              'provider_key': binding.provider_key, 'channel_target': binding.channel_target,
              'message_digest': binding.message_digest, 'idempotency_key': binding.idempotency_key,
              'grant_id': binding.grant_id,
              'grant_revision': binding.grant_revision, 'connection_revision': binding.connection_revision}
    return _reserve_delivery(store, claim, task_id=task_id, actor=actor, expected_version=expected_version,
                             binding=binding, message=message, now_ms=now_ms, clock=clock,
                             managed_preflight=_ManagedPreflight(snapshot, record))
