"""Synthetic, in-process connection of the prototype to a Dojo owner worker.

This adapter is deliberately test-only: the supplied connections and service
are not protected from the worker, shell, or other same-user processes. It is
not an ingress authentication mechanism or a production authorization grant.
"""
from __future__ import annotations

import hashlib
import re

from alphonse.agent_v2.authority_service.prototype import AuthorityDenied, AuthorityService


class DojoOwnerAuthority:
    def __init__(self, service, desktop_connection, worker_connection, emit=lambda _event: None):
        if type(service) is not AuthorityService or not callable(emit):
            raise ValueError('dojo_authority_configuration_invalid')
        self.service = service
        self.desktop = desktop_connection
        self.worker = worker_connection
        self.emit = emit
        self._claims = {}
        self._socket_bridge = None

    @classmethod
    def from_worker_bridge(cls, bridge, *, emit=lambda _event: None):
        """Adapt a socket-backed worker bridge for disposable container Bash.

        The caller must submit the request before enqueuing the owner task.
        Synthetic peer verification does not establish an OS boundary.
        """
        from alphonse.agent_v2.core.intelligence.simple_loop.owner_authority_bridge import OwnerAuthorityBridge
        if type(bridge) is not OwnerAuthorityBridge or not callable(emit):
            raise ValueError('dojo_authority_configuration_invalid')
        instance = object.__new__(cls)
        instance.emit, instance._socket_bridge = emit, bridge
        return instance

    def begin(self, task):
        """Attach after the real worker has validated its normal queue claim.

        The synthetic Desktop call occurs here, not at real owner IPC ingress.
        """
        if self._socket_bridge is not None:
            try:
                self._socket_bridge.begin(task)
            except Exception:
                self.emit({'event': 'authority_denied', 'stage': 'request_or_claim'})
                raise
            self.emit({'event': 'authority_request_claimed'})
            return
        key = (task.user, task.task_id, task.goal)
        if key in self._claims:
            return
        # Real owner task IDs are 32 hex digits. The legacy synthetic unit
        # fixtures use arbitrary names, for which this Dojo-only path derives
        # a deterministic stand-in; the production bridge never does that.
        request_key = (task.task_id if re.fullmatch(r'[0-9a-f]{32}', task.task_id)
                       else hashlib.sha256((task.user + '\0' + task.task_id).encode()).hexdigest()[:32])
        try:
            if (self.service._peer(self.desktop, 'desktop').actor != task.user
                    or self.service._peer(self.worker, 'worker').actor != task.user):
                raise AuthorityDenied('dojo_authority_actor_mismatch')
            request = self.service.submit(self.desktop, goal=task.goal, idempotency_key=request_key)
            claim = self.service.claim(self.worker, request_id=request['request_id'], message_id=request_key,
                                       expected_actor=task.user, goal=task.goal,
                                       lease_seconds=300)
        except Exception:
            self.emit({'event': 'authority_denied', 'stage': 'request_or_claim'})
            raise
        self._claims[key] = claim
        self.emit({'event': 'authority_request_claimed'})

    def _claim(self, task):
        claim = self._claims.get((task.user, task.task_id, task.goal))
        if claim is None:
            raise AuthorityDenied('dojo_authority_claim_missing')
        return claim

    def active(self, task):
        if self._socket_bridge is not None:
            return self._socket_bridge.active(task)
        try:
            return self.service.check_claim(self.worker, claim=self._claim(task)) is True
        except Exception:
            return False

    def start(self, task, tool_id, args, *, reservation_id):
        if self._socket_bridge is not None:
            try:
                action_id = self._socket_bridge.start(task, tool_id, args, reservation_id=reservation_id)
            except Exception:
                self.emit({'event': 'authority_denied', 'stage': 'action_start'})
                raise
            self.emit({'event': 'authority_action_started'})
            return action_id
        action = {'tool_id': tool_id, 'arguments': args}
        claim = self._claim(task)
        try:
            # The owner checkpoint persists this reservation before dispatch.
            # Reusing it binds a retry to the same authority action. This is
            # still a synthetic adapter: cross-store recovery and claim
            # restoration are not implemented by this mapping.
            if not isinstance(reservation_id, str) or re.fullmatch(r'tool-[0-9a-f]{32}', reservation_id) is None:
                raise AuthorityDenied('dojo_authority_reservation_invalid')
            action_key = hashlib.sha256(('dojo-owner-action\0' + task.user + '\0' + task.task_id
                                         + '\0' + reservation_id).encode()).hexdigest()[:32]
            reserved = self.service.reserve_action(self.worker, claim=claim, action=action,
                                                   idempotency_key=action_key)
            self.service.start_action(self.worker, claim=claim, action_id=reserved['action_id'], action=action)
        except Exception:
            self.emit({'event': 'authority_denied', 'stage': 'action_start'})
            raise
        self.emit({'event': 'authority_action_started'})
        return reserved['action_id']

    def returned(self, task, action_id, result):
        # This is the worker's report, not independent effect verification.
        if self._socket_bridge is not None:
            report = {'exit_code': result.get('exit_code'), 'timed_out': result.get('status') == 'timeout',
                      'cancelled': result.get('status') == 'cancelled', 'effect_status': 'unverified'}
            self._socket_bridge.returned(task, action_id, report)
            self.emit({'event': 'authority_return_reported', 'effect': 'unverified'})
            return
        report = {name: result.get(name) for name in ('status', 'exit_code', 'truncated')}
        self.service.record_process_return(self.worker, claim=self._claim(task), action_id=action_id,
                                           process_result=report)
        self.emit({'event': 'authority_return_reported', 'effect': 'unverified'})
