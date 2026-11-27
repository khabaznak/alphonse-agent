"""Owner-scoped worker handoffs for disposable Dojo fixtures.

The injected receiver and verifier are test boundaries. This module configures
no live worker and supports review or patch proposals only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from alphonse.agent_v2.database import transaction


OWNER_DISPOSABLE_HANDOFF_TOOL_ID = 'native.owner_disposable_handoff'
_OPERATIONS = {'review', 'propose_patch'}
_WORKER = 'codex'
_CODES = {
    'accepted': 'worker_accepted',
    'completed': 'result_verified',
    'failed': 'worker_failed',
    'unknown': 'dispatch_unconfirmed',
}
_BLOCKED_CODES = {'scope_unavailable', 'claim_unavailable', 'scope_conflict'}
_OBSERVATION_KEYS = {
    'status', 'operation_id', 'worker', 'target', 'revision', 'operation', 'result_code', 'effect_status',
}
_IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}')
_DIGEST = re.compile(r'[0-9a-f]{64}')


def _identifier(value: object) -> bool:
    return type(value) is str and _IDENTIFIER.fullmatch(value) is not None


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


@dataclass(frozen=True)
class DisposableHandoffBinding:
    actor: str
    target: str
    revision: str
    operation: str
    owner_prompt_sha256: str
    worker: str = _WORKER

    def __post_init__(self) -> None:
        if (not all(_identifier(value) for value in (self.actor, self.target, self.revision))
                or self.worker != _WORKER or type(self.operation) is not str or self.operation not in _OPERATIONS
                or type(self.owner_prompt_sha256) is not str or not _DIGEST.fullmatch(self.owner_prompt_sha256)):
            raise ValueError('disposable_handoff_binding_invalid')

    @property
    def digest(self) -> str:
        return _digest(json.dumps((self.actor, self.target, self.revision, self.operation,
                                   self.owner_prompt_sha256, self.worker)))


@dataclass(frozen=True)
class DisposableHandoffRequest:
    operation_id: str
    worker: str
    target: str
    revision: str
    operation: str


@dataclass(frozen=True)
class DisposableHandoffReceipt:
    request: DisposableHandoffRequest
    status: str


_Claim = Callable[[], bool]
_Receiver = Callable[[DisposableHandoffRequest, _Claim], DisposableHandoffReceipt | None]
_Verifier = Callable[[DisposableHandoffRequest, DisposableHandoffReceipt, _Claim], bool]


def _active(claim_active: _Claim) -> bool:
    try:
        return claim_active() is True
    except Exception:
        return False


def _observation(status: str, code: str, request: DisposableHandoffRequest | None = None) -> dict:
    return {
        'status': status,
        'operation_id': request.operation_id if request else None,
        'worker': _WORKER,
        'target': request.target if request else None,
        'revision': request.revision if request else None,
        'operation': request.operation if request else None,
        'result_code': code,
        'effect_status': 'not_requested',
    }


class DisposableHandoffService:
    """Send one minimal brief per approved fixture operation and reconcile by ID."""

    def __init__(self, db_path: str | Path, *, bindings: tuple[DisposableHandoffBinding, ...],
                 submit: _Receiver, reconcile: _Receiver, verify_result: _Verifier) -> None:
        self.db_path = Path(db_path)
        self._bindings = tuple(bindings)
        if (not self._bindings or any(type(binding) is not DisposableHandoffBinding for binding in self._bindings)
                or not all(callable(hook) for hook in (submit, reconcile, verify_result))):
            raise ValueError('disposable_handoff_configuration_invalid')
        destinations = {(binding.actor, binding.owner_prompt_sha256, binding.target, binding.operation)
                        for binding in self._bindings}
        if len(destinations) != len(self._bindings):
            raise ValueError('disposable_handoff_binding_ambiguous')
        self._submit, self._reconcile, self._verify_result = submit, reconcile, verify_result
        with transaction(self.db_path, immediate=True) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS dojo_worker_handoffs (
                operation_id TEXT PRIMARY KEY, actor TEXT NOT NULL, parent_id TEXT NOT NULL,
                binding_digest TEXT NOT NULL, state TEXT NOT NULL,
                accepted_seen INTEGER NOT NULL DEFAULT 0
            ) STRICT''')

    def _owner_bindings(self, actor: str, owner_prompt: str) -> tuple[DisposableHandoffBinding, ...]:
        if not _identifier(actor) or type(owner_prompt) is not str:
            return ()
        try:
            digest = _digest(owner_prompt)
        except UnicodeEncodeError:
            return ()
        return tuple(binding for binding in self._bindings
                     if binding.actor == actor and binding.owner_prompt_sha256 == digest)

    def targets_for(self, *, actor: str, owner_prompt: str, claim_active: _Claim) -> tuple[tuple[str, str], ...]:
        if not _active(claim_active):
            return ()
        return tuple((binding.target, binding.operation) for binding in self._owner_bindings(actor, owner_prompt))

    @staticmethod
    def _request(binding: DisposableHandoffBinding, parent_id: str) -> DisposableHandoffRequest:
        # The operation remains the same if a new fixture configuration changes
        # the pinned revision. The binding digest detects that scope conflict.
        operation_id = 'dojo-handoff:' + _digest(json.dumps((
            binding.actor, parent_id, binding.worker, binding.target, binding.operation)))
        return DisposableHandoffRequest(operation_id, binding.worker, binding.target,
                                        binding.revision, binding.operation)

    def dispatch(self, *, actor: str, parent_id: str, target: str, worker: str, operation: str,
                 owner_prompt: str, claim_active: _Claim) -> dict:
        if not _active(claim_active):
            return _observation('blocked', 'claim_unavailable')
        if not all(_identifier(value) for value in (parent_id, target, worker, operation)):
            return _observation('blocked', 'scope_unavailable')
        binding = next((binding for binding in self._owner_bindings(actor, owner_prompt)
                        if (binding.target, binding.worker, binding.operation) == (target, worker, operation)), None)
        if binding is None:
            return _observation('blocked', 'scope_unavailable')

        request = self._request(binding, parent_id)
        with transaction(self.db_path, immediate=True) as conn:
            created = conn.execute('''INSERT OR IGNORE INTO dojo_worker_handoffs
                (operation_id, actor, parent_id, binding_digest, state) VALUES (?, ?, ?, ?, 'pending')''',
                (request.operation_id, actor, parent_id, binding.digest)).rowcount == 1
            row = conn.execute('''SELECT binding_digest, state FROM dojo_worker_handoffs
                WHERE operation_id=? AND actor=? AND parent_id=?''', (request.operation_id, actor, parent_id)).fetchone()
        if row is None or row['binding_digest'] != binding.digest:
            return _observation('blocked', 'scope_conflict')
        if not _active(claim_active):
            return _observation('unknown', _CODES['unknown'], request)
        if row['state'] in {'completed', 'failed'}:
            return _observation(row['state'], _CODES[row['state']], request)

        # An existing pending row may represent a lost reply or interrupted
        # dispatch. Reconciliation queries the exact ID and never resubmits.
        callback = self._submit if created else self._reconcile
        try:
            receipt = callback(request, lambda: _active(claim_active))
            status = 'pending'
            if (type(receipt) is DisposableHandoffReceipt and type(receipt.request) is DisposableHandoffRequest
                    and receipt.request == request and type(receipt.status) is str
                    and receipt.status in {'accepted', 'completed', 'failed'}):
                # An executor's completion claim requires independent readback.
                # A previously recorded acceptance stays available if this
                # receipt cannot advance the operation.
                status = receipt.status
                if status == 'completed' and (not _active(claim_active)
                        or self._verify_result(request, receipt, lambda: _active(claim_active)) is not True):
                    status = 'pending'
        except Exception:
            status = 'pending'
        if not _active(claim_active):
            return _observation('unknown', _CODES['unknown'], request)

        with transaction(self.db_path, immediate=True) as conn:
            if status != 'pending':
                conn.execute('''UPDATE dojo_worker_handoffs SET state=?,
                    accepted_seen=MAX(accepted_seen, ?) WHERE operation_id=? AND actor=?
                    AND parent_id=? AND binding_digest=? AND state IN ('pending', 'accepted')''',
                    (status, int(status in {'accepted', 'completed'}), request.operation_id, actor,
                     parent_id, binding.digest))
            current = conn.execute('''SELECT state FROM dojo_worker_handoffs WHERE operation_id=? AND actor=?
                AND parent_id=? AND binding_digest=?''', (request.operation_id, actor, parent_id, binding.digest)).fetchone()
        status = 'unknown' if current is None or current['state'] == 'pending' else current['state']
        if not _active(claim_active):
            status = 'unknown'
        return _observation(status, _CODES[status], request)

    def valid_observation(self, *, actor: str, parent_id: str, observation: object) -> bool:
        """Validate an owner-scoped projection without contacting the worker."""
        if (not _identifier(actor) or not _identifier(parent_id) or type(observation) is not dict
                or set(observation) != _OBSERVATION_KEYS or type(observation['status']) is not str
                or type(observation['result_code']) is not str):
            return False
        status = observation['status']
        if status == 'blocked':
            return (observation['result_code'] in _BLOCKED_CODES
                    and observation == _observation('blocked', observation['result_code']))
        if status not in _CODES or type(observation['operation_id']) is not str:
            return False
        with transaction(self.db_path) as conn:
            row = conn.execute('''SELECT binding_digest, state, accepted_seen FROM dojo_worker_handoffs
                WHERE operation_id=? AND actor=? AND parent_id=?''',
                (observation['operation_id'], actor, parent_id)).fetchone()
        if row is None or not any(item.actor == actor and item.digest == row['binding_digest']
                                  and _observation(status, _CODES[status], self._request(item, parent_id)) == observation
                                  for item in self._bindings):
            return False
        return row['state'] == status or status == 'unknown' or status == 'accepted' and row['accepted_seen'] == 1
