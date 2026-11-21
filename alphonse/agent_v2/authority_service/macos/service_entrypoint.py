"""Experimental foreground authority service for provisioned macOS hosts.

This is not an installer, not a human-authorization mechanism, and not
connected to normal owner conversations. It never provisions an identity,
creates or unlinks listeners, or falls back to synthetic peer verification.

The launcher must itself be protected before Python starts: this module cannot
prove the integrity of code already loaded into its process. Protected launchd
configuration, interpreter/executable, imports, dependencies, and native
library are essential external prerequisites. Host Bash remains disabled.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import json
import os
import re
import signal
import socket
import stat
import sys
import threading
from pathlib import PurePosixPath
from contextlib import nullcontext

from .peer_verifier import MacOSSocketPeerVerifier
from ..transport import AuthoritySocketService, EndpointIdentity


class ServiceConfigurationError(ValueError):
    """Provisioning or runtime verification failed."""


def _invalid():
    raise ServiceConfigurationError('authority_service_configuration_unavailable')


def _absolute_parts(path):
    if (not isinstance(path, str) or not path.startswith('/') or '\x00' in path
            or path == '/' or any(part in {'.', '..', ''} for part in path[1:].split('/'))):
        _invalid()
    return PurePosixPath(path).parts[1:]


def _no_acl(fd):
    """Reject any extended ACL; uncertainty or unsupported filesystems fail."""
    if sys.platform != 'darwin':
        _invalid()
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        get = libc.acl_get_fd_np
        get.argtypes = [ctypes.c_int, ctypes.c_int]
        get.restype = ctypes.c_void_p
        free = libc.acl_free
        free.argtypes = [ctypes.c_void_p]
        free.restype = ctypes.c_int
        ctypes.set_errno(0)
        acl = get(fd, 0x00000100)  # ACL_TYPE_EXTENDED in Darwin sys/acl.h
        if acl:
            free(acl)
            return False
        return ctypes.get_errno() == errno.ENOENT
    except (AttributeError, OSError, TypeError, ValueError):
        return False


def _open_checked(path, *, directory_owners, final_owners, final_kind):
    """Open through trusted directories without following links or ACLs."""
    parts = _absolute_parts(path)
    opened = []
    try:
        root = os.open('/', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        opened.append(root)
        root_info = os.fstat(root)
        if root_info.st_uid != 0 or root_info.st_mode & 0o022 or not _no_acl(root):
            _invalid()
        current = root
        for index, part in enumerate(parts):
            final = index == len(parts) - 1
            flags = os.O_RDONLY | os.O_NOFOLLOW | (os.O_DIRECTORY if not final or final_kind == 'directory' else 0)
            fd = os.open(part, flags, dir_fd=current)
            opened.append(fd)
            info = os.fstat(fd)
            owners = final_owners if final else directory_owners
            if info.st_uid not in owners or info.st_mode & 0o022 or not _no_acl(fd):
                _invalid()
            if final:
                if not ((final_kind == 'file' and stat.S_ISREG(info.st_mode) and info.st_nlink == 1)
                        or (final_kind == 'directory' and stat.S_ISDIR(info.st_mode))):
                    _invalid()
            elif not stat.S_ISDIR(info.st_mode):
                _invalid()
            current = fd
        return os.dup(current)
    except (OSError, ValueError, TypeError):
        _invalid()
    finally:
        for fd in reversed(opened):
            os.close(fd)


def _read_json(fd):
    data = bytearray()
    while len(data) <= 16384:
        chunk = os.read(fd, min(4096, 16385 - len(data)))
        if not chunk:
            break
        data.extend(chunk)
    if len(data) > 16384:
        _invalid()
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                _invalid()
            obj[key] = value
        return obj
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=unique,
                          parse_constant=lambda _: _invalid())
    except (ValueError, UnicodeError):
        _invalid()


def validate_config(config, *, effective_uid):
    """Validate structure and refuse owner/root identity before any serving."""
    fields = {'version', 'service_uid', 'owner_uid', 'actor', 'ledger_path',
              'verifier_library_path', 'desktop', 'worker'}
    if not isinstance(config, dict) or set(config) != fields or type(config['version']) is not int or config['version'] != 1:
        _invalid()
    service, owner = config['service_uid'], config['owner_uid']
    if (type(service) is not int or type(owner) is not int or type(effective_uid) is not int
            or not 0 < service < 0xffffffff or not 0 < owner < 0xffffffff
            or service == owner or service != effective_uid):
        _invalid()
    if (not isinstance(config['actor'], str) or not config['actor'].strip()
            or len(config['actor']) > 256 or '\x00' in config['actor']):
        _invalid()
    for path in ('ledger_path', 'verifier_library_path'):
        _absolute_parts(config[path])
    for role in ('desktop', 'worker'):
        item = config[role]
        if (not isinstance(item, dict) or set(item) not in (
                {'team', 'identifier', 'client_id', 'socket_path'}, {'cdhash', 'client_id', 'socket_path'})
                or any(not isinstance(item[k], str) or not item[k].strip() or len(item[k]) > 256 or '\x00' in item[k]
                       for k in item)):
            _invalid()
        if ('cdhash' in item and re.fullmatch(r'[0-9a-f]{40}', item['cdhash']) is None):
            _invalid()
        if ('team' in item and (re.fullmatch(r'[A-Z0-9]{10}', item['team']) is None
                or re.fullmatch(r'[A-Za-z0-9._-]{1,255}', item['identifier']) is None)):
            _invalid()
        _absolute_parts(item['socket_path'])
    desktop, worker = config['desktop'], config['worker']
    if (('cdhash' in desktop) != ('cdhash' in worker)
            or desktop['socket_path'] == worker['socket_path'] or desktop['client_id'] == worker['client_id']
            or (('cdhash' in desktop and desktop['cdhash'] == worker['cdhash'])
                or ('team' in desktop and (desktop['team'], desktop['identifier']) == (worker['team'], worker['identifier'])))):
        _invalid()
    return config


def _listener(fd, path, service_uid, verify_listener):
    if type(fd) is not int or fd < 3:
        _invalid()
    # The socket itself may be group-writable so authorized clients can
    # connect; only protected ancestors may replace its filesystem name.
    parent, name = path.rsplit('/', 1)
    if not parent:
        _invalid()
    directory_fd = _open_checked(parent, directory_owners={0, service_uid},
                                 final_owners={0, service_uid}, final_kind='directory')
    listener = None
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if not stat.S_ISSOCK(info.st_mode) or info.st_uid not in {0, service_uid}:
            _invalid()
        listener = socket.socket(fileno=os.dup(fd))
        if (not callable(verify_listener) or listener.family != socket.AF_UNIX
                or listener.getsockopt(socket.SOL_SOCKET, socket.SO_TYPE) != socket.SOCK_STREAM
                or verify_listener(listener) is not True
                or listener.getsockname() != path):
            _invalid()
        return listener
    except Exception:
        if listener is not None:
            listener.close()
        _invalid()
    finally:
        os.close(directory_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='root-owned protected configuration path')
    parser.add_argument('--desktop-fd', type=int)
    parser.add_argument('--worker-fd', type=int)
    parser.add_argument('--launchd', action='store_true', help='check in launchd Desktop and Worker sockets')
    args = parser.parse_args()
    if (sys.platform != 'darwin' or sys.flags.isolated != 1 or sys.flags.no_user_site != 1
            or os.getuid() != os.geteuid()):
        _invalid()
    fd = _open_checked(args.config, directory_owners={0}, final_owners={0}, final_kind='file')
    try:
        config = validate_config(_read_json(fd), effective_uid=os.geteuid())
    finally:
        os.close(fd)
    if ((args.launchd and (args.desktop_fd is not None or args.worker_fd is not None))
            or (not args.launchd and (args.desktop_fd is None or args.worker_fd is None
                                      or args.desktop_fd == args.worker_fd))):
        _invalid()
    # Validate files before loading any configured native verifier.
    fd = _open_checked(config['ledger_path'], directory_owners={0, config['service_uid']},
                       final_owners={config['service_uid']}, final_kind='file')
    try:
        if os.fstat(fd).st_mode & 0o077:
            _invalid()
    finally:
        os.close(fd)
    fd = _open_checked(config['verifier_library_path'], directory_owners={0},
                       final_owners={0}, final_kind='file')
    os.close(fd)
    desktop, worker = config['desktop'], config['worker']
    verifier_kwargs = ({'desktop_cdhash': desktop['cdhash'], 'worker_cdhash': worker['cdhash']}
                       if 'cdhash' in desktop else
                       {'desktop_team': desktop['team'], 'desktop_identifier': desktop['identifier'],
                        'worker_team': worker['team'], 'worker_identifier': worker['identifier']})
    verifier = MacOSSocketPeerVerifier(config['verifier_library_path'], **verifier_kwargs)
    endpoints = {role: EndpointIdentity(role, config['actor'], config[role]['client_id'])
                 for role in ('desktop', 'worker')}
    service = AuthoritySocketService(config['ledger_path'], endpoints=endpoints, verify_peer=verifier)
    stopped = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stopped.set())
    if args.launchd:
        from .launchd_activation import activated_listeners
        activated = activated_listeners()
    else:
        activated = nullcontext((args.desktop_fd, args.worker_fd))
    with activated as (desktop_fd, worker_fd):
        listeners = {}
        try:
            for role, raw_fd in (('desktop', desktop_fd), ('worker', worker_fd)):
                listeners[role] = _listener(raw_fd, config[role]['socket_path'],
                                            config['service_uid'], verifier.is_listening)
            service.serve_listeners(listeners, stop=stopped)
        finally:
            for listener in listeners.values():
                listener.close()


if __name__ == '__main__':
    main()
