"""Experimental worker bridge to an existing authority-service request.

This is a wiring seam, not an OS authorization boundary. The client transport,
the origin and immutability of the request binding, the worker identity, and
cross-store recovery must be protected independently before production use.
This worker bridge can never submit an owner request on behalf of a Desktop.
"""
from __future__ import annotations

import hashlib
import re
import threading


class OwnerAuthorityUnavailable(ValueError):
    pass


def context_authority_active(context):
    """Recheck an attached service claim; unavailable or malformed fails closed.

    Contexts without an experimental authority retain their existing behavior.
    This check cannot atomically fence a separate service and SQLite store.
    """
    required = getattr(context, '_owner_authority_required', False)
    if required is False:
        return True
    if required is not True:
        return False
    check = getattr(context, '_owner_authority_check', None)
    try:
        return callable(check) and check() is True
    except Exception:
        return False


class OwnerAuthorityBridge:
    """Consume pre-existing request bindings and record exact Bash actions.

    ``request_binding`` is supplied by trusted ingress/runtime wiring; it must
    not derive authority from a model, a same-user-writable row, or this class.
    No such production wiring exists yet. The service must authenticate both
    its peer and its client independently of fields passed here.
    """

    def __init__(self, client, request_binding):
        methods = ('claim', 'check_claim', 'check_bash_capability', 'reserve_action', 'start_action', 'record_process_return')
        if not callable(request_binding) or any(not callable(getattr(client, name, None)) for name in methods):
            raise OwnerAuthorityUnavailable('owner_authority_configuration_invalid')
        self._client = client
        self._request_binding = request_binding
        self._claims = {}
        self._lock = threading.RLock()

    @staticmethod
    def _key(task):
        actor, task_id, goal = getattr(task, 'user', None), getattr(task, 'task_id', None), getattr(task, 'goal', None)
        if not all(isinstance(value, str) and value.strip() for value in (actor, task_id, goal)):
            raise OwnerAuthorityUnavailable('owner_authority_task_invalid')
        return actor, task_id, goal

    def begin(self, task):
        key = self._key(task)
        with self._lock:
            if key in self._claims:
                if not self.active(task):
                    raise OwnerAuthorityUnavailable('owner_authority_claim_unavailable')
                return
            try:
                request_id = self._request_binding(task)
                if not isinstance(request_id, str) or re.fullmatch(r'[0-9a-f]{32}', request_id) is None:
                    raise ValueError()
                issued = self._client.claim(request_id=request_id, message_id=key[1], goal=key[2],
                                            expected_actor=key[0], lease_seconds=300)
                if not isinstance(issued, dict) or set(issued) != {'claim', 'actor'} or issued['actor'] != key[0]:
                    raise ValueError()
                claim = issued['claim']
                if (not isinstance(claim, dict) or set(claim) != {'request_id', 'generation', 'token'}
                        or claim['request_id'] != request_id or type(claim['generation']) is not int
                        or claim['generation'] < 1 or not isinstance(claim['token'], str)
                        or re.fullmatch(r'[0-9a-f]{64}', claim['token']) is None):
                    raise ValueError()
            except Exception:
                raise OwnerAuthorityUnavailable('owner_authority_claim_unavailable') from None
            self._claims[key] = claim

    def _claim(self, task):
        claim = self._claims.get(self._key(task))
        if claim is None:
            raise OwnerAuthorityUnavailable('owner_authority_claim_unavailable')
        return claim

    def active(self, task):
        try:
            with self._lock:
                return self._client.check_claim(claim=self._claim(task)) is True
        except Exception:
            return False

    def bash_available(self, task):
        """A failed capability lookup hides Bash; it never authorizes execution."""
        try:
            with self._lock:
                return self._client.check_bash_capability(claim=self._claim(task)) is True
        except Exception:
            return False

    def start(self, task, tool_id, arguments, *, reservation_id):
        if (tool_id != 'native.bash' or not isinstance(arguments, dict)
                or not isinstance(reservation_id, str)
                or re.fullmatch(r'tool-[0-9a-f]{32}', reservation_id) is None):
            raise OwnerAuthorityUnavailable('owner_authority_action_invalid')
        actor, task_id, _ = self._key(task)
        action = {'tool_id': tool_id, 'arguments': arguments}
        action_key = hashlib.sha256(('owner-action\0' + actor + '\0' + task_id + '\0'
                                     + reservation_id).encode()).hexdigest()[:32]
        try:
            with self._lock:
                claim = self._claim(task)
                reserved = self._client.reserve_action(claim=claim, action=action, idempotency_key=action_key)
                if (not isinstance(reserved, dict) or reserved.get('status') != 'reserved'
                        or not isinstance(reserved.get('action_id'), str)
                        or re.fullmatch(r'[0-9a-f]{32}', reserved['action_id']) is None):
                    raise ValueError()
                started = self._client.start_action(claim=claim, action_id=reserved['action_id'], action=action)
                if not isinstance(started, dict) or started.get('status') != 'started' or started.get('effect') != 'unverified':
                    raise ValueError()
                return reserved['action_id']
        except Exception:
            raise OwnerAuthorityUnavailable('owner_authority_action_unavailable') from None

    def returned(self, task, action_id, result):
        try:
            if not isinstance(result, dict):
                raise ValueError()
            report = {name: result.get(name) for name in ('exit_code', 'timed_out', 'cancelled', 'effect_status')}
            with self._lock:
                returned = self._client.record_process_return(claim=self._claim(task), action_id=action_id,
                                                               process_result=report)
            if (not isinstance(returned, dict) or returned.get('status') != 'return_reported'
                    or returned.get('effect') != 'unverified'):
                raise ValueError()
        except Exception:
            raise OwnerAuthorityUnavailable('owner_authority_return_unavailable') from None
