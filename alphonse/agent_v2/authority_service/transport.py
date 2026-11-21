"""One-request Unix-stream protocol for the authority ledger prototype.

The server binds each endpoint to a provisioned role and actor; JSON never
supplies a peer identity. The caller must install a real kernel-backed peer
verifier and protect the service code, endpoint configuration and database.
This module does not provision those protections, prove human intent, connect
the owner loop, or establish cross-store crash reconciliation.
"""
from __future__ import annotations

import json
import selectors
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable

from .prototype import AuthenticatedPeer, AuthorityDenied, AuthorityService

_MAX_FRAME = 131072
_CONNECTION_DEADLINE_SECONDS = 3.0
_OPS = {
    'bash_capability': ('desktop', set()),
    'grant_bash': ('desktop', {'expected_generation'}),
    'revoke_bash': ('desktop', {'expected_generation'}),
    'submit': ('desktop', {'goal', 'idempotency_key'}),
    'inspect_request': ('desktop', {'request_id', 'message_id', 'goal'}),
    'lookup_request': ('desktop', {'message_id', 'goal'}),
    'cancel': ('desktop', {'request_id'}),
    'claim': ('worker', {'request_id', 'message_id', 'goal', 'expected_actor', 'lease_seconds'}),
    'check_claim': ('worker', {'claim'}),
    'check_bash_capability': ('worker', {'claim'}),
    'reserve_action': ('worker', {'claim', 'action', 'idempotency_key'}),
    'start_action': ('worker', {'claim', 'action_id', 'action'}),
    'record_process_return': ('worker', {'claim', 'action_id', 'process_result'}),
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate_key')
        result[key] = value
    return result


def _remaining(connection, deadline):
    if deadline is not None:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('authority_deadline_exceeded')
        connection.settimeout(remaining)


def _read_exact(connection, count, *, deadline=None):
    chunks = bytearray()
    while len(chunks) < count:
        _remaining(connection, deadline)
        chunk = connection.recv(count - len(chunks))
        if not chunk:
            raise ValueError('short_frame')
        chunks.extend(chunk)
    return bytes(chunks)


def _read_frame(connection, *, deadline=None):
    size = struct.unpack('!I', _read_exact(connection, 4, deadline=deadline))[0]
    if not 0 < size <= _MAX_FRAME:
        raise ValueError('invalid_frame_size')
    def bad_constant(_):
        raise ValueError('invalid_number')
    data = json.loads(_read_exact(connection, size, deadline=deadline).decode('utf-8'),
                      object_pairs_hook=_unique_object, parse_constant=bad_constant)
    if not isinstance(data, dict):
        raise ValueError('invalid_frame')
    return data


def _write_frame(connection, value, *, deadline=None):
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')
    if not 0 < len(payload) <= _MAX_FRAME:
        raise ValueError('invalid_frame_size')
    _remaining(connection, deadline)
    connection.sendall(struct.pack('!I', len(payload)) + payload)


@dataclass(frozen=True)
class EndpointIdentity:
    """Provisioned by the service, never deserialized from a request."""
    role: str
    actor: str
    client_id: str

    def __post_init__(self):
        if (self.role not in {'desktop', 'worker'}
                or not isinstance(self.actor, str) or not self.actor.strip() or len(self.actor) > 256
                or not isinstance(self.client_id, str) or not self.client_id.strip() or len(self.client_id) > 256):
            raise ValueError('authority_endpoint_identity_invalid')


class _Session:
    def __init__(self, seal, identity):
        self._seal, self._identity = seal, identity


class AuthoritySocketService:
    """Bridge a verified connection to the ledger, one framed request each.

    ``verify_peer`` must inspect kernel peer identity on the accepted socket
    against service-owned role policy; accepting a JSON field, PID or shared
    same-user secret here is not a valid security boundary. An injected fake
    verifier is appropriate only for synthetic tests.
    """
    def __init__(self, path, *, endpoints: dict[str, EndpointIdentity], verify_peer: Callable,
                 verify_listener: Callable | None = None):
        if (not isinstance(endpoints, dict) or set(endpoints) != {'desktop', 'worker'}
                or any(type(endpoints[r]) is not EndpointIdentity or endpoints[r].role != r for r in endpoints)
                or endpoints['desktop'].actor != endpoints['worker'].actor
                or endpoints['desktop'].client_id == endpoints['worker'].client_id
                or not callable(verify_peer) or (verify_listener is not None and not callable(verify_listener))):
            raise ValueError('authority_endpoint_configuration_invalid')
        self._endpoints = dict(endpoints)
        self._verify_peer = verify_peer
        # Production verifier supplies a native kernel-backed predicate. Tests
        # may explicitly inject a fake; missing/failed checks never start a loop.
        self._verify_listener = verify_listener or getattr(verify_peer, 'is_listening', None)
        self._seal = object()
        def authenticate(session):
            if type(session) is not _Session or session._seal is not self._seal:
                raise AuthorityDenied('authority_peer_denied')
            item = session._identity
            return AuthenticatedPeer(item.role, item.actor, item.client_id)
        self._ledger = AuthorityService(path, authenticate=authenticate)

    def serve_listeners(self, listeners: dict[str, socket.socket], *, stop: threading.Event,
                        max_connections: int = 16):
        """Serve prebound role-specific listeners until stopped.

        Provisioning owns socket creation, path permissions and cleanup; this
        method never creates, changes permissions on or removes socket paths.
        The configured verifier must be bounded and nonblocking. Each active
        connection has a three-second I/O timeout. Excess peers are closed.
        """
        if (not isinstance(listeners, dict) or set(listeners) != set(self._endpoints)
                or not isinstance(stop, threading.Event) or type(max_connections) is not int
                or not 1 <= max_connections <= 64):
            raise ValueError('authority_listener_configuration_invalid')
        sockets = tuple(listeners.values())
        try:
            valid_listeners = (callable(self._verify_listener)
                and len({id(item) for item in sockets}) == len(sockets)
                and all(isinstance(item, socket.socket) and item.family == socket.AF_UNIX
                        and item.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) == socket.SOCK_STREAM
                        and self._verify_listener(item) is True for item in sockets))
        except Exception:
            valid_listeners = False
        if not valid_listeners:
            raise ValueError('authority_listener_configuration_invalid')
        slots = threading.BoundedSemaphore(max_connections)
        selector = selectors.DefaultSelector()
        original_timeouts = {item: item.gettimeout() for item in sockets}
        try:
            for role, listener in listeners.items():
                listener.setblocking(False)
                selector.register(listener, selectors.EVENT_READ, role)
            with ThreadPoolExecutor(max_workers=max_connections, thread_name_prefix='authority-peer') as pool:
                while not stop.is_set():
                    for key, _ in selector.select(.2):
                        try:
                            peer, _ = key.fileobj.accept()
                        except (BlockingIOError, InterruptedError):
                            continue
                        if stop.is_set() or not slots.acquire(blocking=False):
                            peer.close()
                            continue
                        try:
                            future = pool.submit(self.serve_connection, peer, endpoint_role=key.data)
                            future.add_done_callback(lambda _future: slots.release())
                        except Exception:
                            slots.release()
                            peer.close()
                            raise
        finally:
            selector.close()
            for listener, timeout in original_timeouts.items():
                try:
                    listener.settimeout(timeout)
                except OSError:
                    pass

    def serve_connection(self, connection: socket.socket, *, endpoint_role: str):
        """Validate peer before reading payload, then serve a single request."""
        verified = False
        deadline = time.monotonic() + _CONNECTION_DEADLINE_SECONDS
        try:
            if (not isinstance(connection, socket.socket) or connection.family != socket.AF_UNIX
                    or connection.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM
                    or endpoint_role not in self._endpoints):
                return
            _remaining(connection, deadline)
            # Verification occurs before reading or replying, with role taken
            # only from the server's selected endpoint.
            if self._verify_peer(connection, endpoint_role) is not True:
                return
            verified = True
            session = _Session(self._seal, self._endpoints[endpoint_role])
            request = _read_frame(connection, deadline=deadline)
            if set(request) != {'op', 'arguments'} or not isinstance(request['op'], str):
                raise ValueError('invalid_request')
            operation = _OPS.get(request['op'])
            args = request['arguments']
            if (operation is None or operation[0] != endpoint_role or not isinstance(args, dict)
                    or set(args) != operation[1]):
                raise ValueError('invalid_request')
            result = getattr(self._ledger, request['op'])(session, **args)
            if request['op'] == 'claim':
                # The worker must bind the protected service actor to its
                # current task. Do not add it inside the claim token object:
                # ledger claim validation requires that object's exact shape.
                result = {'claim': result, 'actor': session._identity.actor}
            _write_frame(connection, {'ok': True, 'result': result}, deadline=deadline)
        except Exception:
            # Do not disclose exception text, SQL, goals or credentials.
            if verified:
                try:
                    _write_frame(connection, {'ok': False, 'error': 'authority_request_denied'}, deadline=deadline)
                except Exception:
                    pass
        finally:
            try:
                connection.close()
            except Exception:
                pass


class AuthorityClient:
    """One operation per socket from a service-authenticating factory.

    Production callers must use a factory that checks the protected service's
    kernel UID *before sending any bytes*. A bare socket connector is test-only.
    """
    def __init__(self, connect: Callable[[], socket.socket]):
        if not callable(connect):
            raise ValueError('authority_connection_required')
        self._connect = connect

    def _call(self, op, **arguments):
        connection = self._connect()
        deadline = time.monotonic() + _CONNECTION_DEADLINE_SECONDS
        try:
            _write_frame(connection, {'op': op, 'arguments': arguments}, deadline=deadline)
            response = _read_frame(connection, deadline=deadline)
        except Exception:
            raise AuthorityDenied('authority_transport_unavailable') from None
        finally:
            connection.close()
        if set(response) != {'ok', 'result'} or response['ok'] is not True:
            raise AuthorityDenied('authority_request_denied')
        return response['result']

    def bash_capability(self):
        return self._call('bash_capability')

    def grant_bash(self, *, expected_generation):
        return self._call('grant_bash', expected_generation=expected_generation)

    def revoke_bash(self, *, expected_generation):
        return self._call('revoke_bash', expected_generation=expected_generation)

    def submit(self, *, goal, idempotency_key):
        return self._call('submit', goal=goal, idempotency_key=idempotency_key)

    def cancel(self, *, request_id):
        return self._call('cancel', request_id=request_id)

    def inspect_request(self, *, request_id, message_id, goal):
        return self._call('inspect_request', request_id=request_id, message_id=message_id, goal=goal)

    def lookup_request(self, *, message_id, goal):
        return self._call('lookup_request', message_id=message_id, goal=goal)

    def claim(self, *, request_id, message_id, goal, expected_actor, lease_seconds=60):
        return self._call('claim', request_id=request_id, message_id=message_id, goal=goal,
                          expected_actor=expected_actor, lease_seconds=lease_seconds)

    def check_claim(self, *, claim):
        return self._call('check_claim', claim=claim)

    def check_bash_capability(self, *, claim):
        return self._call('check_bash_capability', claim=claim)

    def reserve_action(self, *, claim, action, idempotency_key):
        return self._call('reserve_action', claim=claim, action=action, idempotency_key=idempotency_key)

    def start_action(self, *, claim, action_id, action):
        return self._call('start_action', claim=claim, action_id=action_id, action=action)

    def record_process_return(self, *, claim, action_id, process_result):
        return self._call('record_process_return', claim=claim, action_id=action_id, process_result=process_result)


def macos_authority_client(path, *, service_uid: int) -> AuthorityClient:
    """Use kernel UID verification before sending. UID must be protected config.

    This does not authenticate Desktop or worker code and is not sufficient
    to enable host Bash without an installed, protected service and worker.
    """
    from .macos.local_transport import connect_verified_service
    return AuthorityClient(lambda: connect_verified_service(path, service_uid=service_uid))
