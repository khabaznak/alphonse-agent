"""macOS local authority connection with kernel-checked service identity.

This is a transport building block, not an installed authority service or an
authorization grant. The caller must supply a provisioned service UID from
configuration that the shell user cannot alter. The service must run under a
dedicated OS identity that the shell user cannot impersonate; otherwise a UID
does not establish the identity of the service program. Desktop and worker
authentication, owner intent, and durable action coordination are separate.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path
import socket
import sys


class AuthorityConnectionError(ConnectionError):
    """The protected authority transport could not be established."""


def _peer_effective_uid(connection: socket.socket) -> int:
    """Return the kernel-provided effective UID of the connected Unix peer."""
    if sys.platform != 'darwin':
        raise AuthorityConnectionError('authority_peer_platform_unsupported')
    if (not isinstance(connection, socket.socket) or connection.family != socket.AF_UNIX
            or connection.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM):
        raise AuthorityConnectionError('authority_peer_socket_invalid')
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        getpeereid = libc.getpeereid
        getpeereid.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_uint)]
        getpeereid.restype = ctypes.c_int
        uid, gid = ctypes.c_uint(), ctypes.c_uint()
        if getpeereid(connection.fileno(), ctypes.byref(uid), ctypes.byref(gid)) != 0:
            raise AuthorityConnectionError('authority_peer_credentials_unavailable')
        return int(uid.value)
    except AuthorityConnectionError:
        raise
    except (AttributeError, OSError, TypeError, ValueError):
        raise AuthorityConnectionError('authority_peer_credentials_unavailable') from None


def connect_verified_service(path: str | Path, *, service_uid: int, timeout_seconds: float = 3) -> socket.socket:
    """Connect to a distinct service UID and verify it before writing any data.

    The returned connected stream can be used only by trusted integration
    code after its own authenticated client/owner and request checks. The
    function never sends a handshake, prompt, token, or other bytes. A failed
    check closes the stream. The UID argument is trusted provisioning, never
    an IPC request field or same-user-writable preference.
    """
    if sys.platform != 'darwin':
        raise AuthorityConnectionError('authority_peer_platform_unsupported')
    if (type(service_uid) is not int or not 0 <= service_uid < 0xffffffff
            or service_uid == os.geteuid()):
        raise AuthorityConnectionError('authority_service_identity_invalid')
    if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
            or not 0 < timeout_seconds <= 30):
        raise AuthorityConnectionError('authority_connection_timeout_invalid')
    try:
        name = os.fspath(path)
        if not isinstance(name, str) or not os.path.isabs(name) or '\x00' in name:
            raise AuthorityConnectionError('authority_socket_path_invalid')
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.set_inheritable(False)
        connection.settimeout(timeout_seconds)
        try:
            connection.connect(name)
            if _peer_effective_uid(connection) != service_uid:
                raise AuthorityConnectionError('authority_service_identity_mismatch')
            return connection
        except Exception:
            connection.close()
            raise
    except AuthorityConnectionError:
        raise
    except (OSError, TypeError, ValueError):
        raise AuthorityConnectionError('authority_connection_unavailable') from None
