"""Obtain preconfigured role sockets from the calling launchd job.

This code never registers a job or creates a user. The caller must already be
running as the dedicated service identity under protected launch configuration.
The service entrypoint independently validates every returned listener.
"""
from __future__ import annotations

from contextlib import contextmanager
import ctypes
import os
import sys


class LaunchdActivationError(RuntimeError):
    pass


def _library():
    return ctypes.CDLL(None, use_errno=True)


def _activate(name):
    """Return one launchd-owned descriptor; release the allocated fd array."""
    if sys.platform != 'darwin' or name not in {'Desktop', 'Worker'}:
        raise LaunchdActivationError('authority_launchd_unavailable')
    pointer = ctypes.POINTER(ctypes.c_int)()
    count = ctypes.c_size_t()
    try:
        libc = _library()
        activate = libc.launch_activate_socket
        activate.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_int)),
                             ctypes.POINTER(ctypes.c_size_t)]
        activate.restype = ctypes.c_int
        free = libc.free
        free.argtypes = [ctypes.c_void_p]
        free.restype = None
        result = activate(name.encode('ascii'), ctypes.byref(pointer), ctypes.byref(count))
    except (AttributeError, OSError, TypeError, ValueError):
        raise LaunchdActivationError('authority_launchd_unavailable') from None
    try:
        if result != 0 or not pointer or count.value != 1:
            # A valid response can own multiple FDs; close them before failing.
            # The system library is trusted, but never iterate unbounded data.
            if result == 0 and pointer and count.value <= 64:
                for index in range(count.value):
                    try:
                        os.close(pointer[index])
                    except OSError:
                        pass
            raise LaunchdActivationError('authority_launchd_socket_invalid')
        fd = int(pointer[0])
        if fd < 3:
            raise LaunchdActivationError('authority_launchd_socket_invalid')
        try:
            os.set_inheritable(fd, False)
        except OSError:
            try:
                os.close(fd)
            except OSError:
                pass
            raise LaunchdActivationError('authority_launchd_socket_invalid') from None
        return fd
    finally:
        if pointer:
            free(pointer)


@contextmanager
def activated_listeners():
    """Check in both roles and close descriptors on any exit or failure."""
    descriptors = []
    try:
        descriptors.append(_activate('Desktop'))
        descriptors.append(_activate('Worker'))
        if descriptors[0] == descriptors[1]:
            raise LaunchdActivationError('authority_launchd_socket_invalid')
        yield tuple(descriptors)
    finally:
        for descriptor in set(descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
