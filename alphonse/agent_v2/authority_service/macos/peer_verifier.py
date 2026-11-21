"""Load the protected service's native macOS socket-peer verifier.

The library path and signing identities must come from root/service-owned
provisioning, not from model output or a same-user-writable preference.
Loading this helper without those protections establishes no OS boundary.
"""
from __future__ import annotations

import ctypes
import os
import re
import socket
import sys


class MacOSSocketPeerVerifier:
    def __init__(self, library_path, *, desktop_team=None, desktop_identifier=None,
                 worker_team=None, worker_identifier=None, desktop_cdhash=None, worker_cdhash=None):
        if sys.platform != 'darwin' or not os.path.isabs(os.fspath(library_path)):
            raise ValueError('authority_peer_verifier_unavailable')
        pinned = desktop_cdhash is not None or worker_cdhash is not None
        if pinned:
            if (any(value is not None for value in (desktop_team, desktop_identifier, worker_team, worker_identifier))
                    or any(not isinstance(value, str) or re.fullmatch(r'[0-9a-f]{40}', value) is None
                           for value in (desktop_cdhash, worker_cdhash))
                    or desktop_cdhash == worker_cdhash):
                raise ValueError('authority_peer_verifier_configuration_invalid')
            values = (desktop_cdhash, worker_cdhash)
        else:
            values = (desktop_team, desktop_identifier, worker_team, worker_identifier)
        if any(not isinstance(value, str) or not value or '\x00' in value or len(value) > 255 for value in values):
            raise ValueError('authority_peer_verifier_configuration_invalid')
        try:
            self._library = ctypes.CDLL(os.fspath(library_path))
            self._verify = (self._library.alphonse_authority_verify_socket_peer_cdhash if pinned
                            else self._library.alphonse_authority_verify_socket_peer)
            self._verify.argtypes = [ctypes.c_int, ctypes.c_char_p] + [ctypes.c_char_p] * len(values)
            self._verify.restype = ctypes.c_int
            self._listener = self._library.alphonse_authority_is_unix_listener
            self._listener.argtypes = [ctypes.c_int]
            self._listener.restype = ctypes.c_int
            self._values = tuple(value.encode('utf-8') for value in values)
        except (OSError, AttributeError, TypeError, ValueError, UnicodeError):
            raise ValueError('authority_peer_verifier_unavailable') from None

    def __call__(self, connection, role):
        if (not isinstance(connection, socket.socket) or connection.family != socket.AF_UNIX
                or role not in {'desktop', 'worker'}):
            return False
        try:
            return self._verify(connection.fileno(), role.encode('ascii'), *self._values) == 1
        except (OSError, TypeError, ValueError):
            return False

    def is_listening(self, connection):
        """Check Darwin kernel fd metadata; do not accept or change backlog."""
        if not isinstance(connection, socket.socket) or connection.family != socket.AF_UNIX:
            return False
        try:
            return self._listener(connection.fileno()) == 1
        except (OSError, TypeError, ValueError):
            return False
