"""Provisioned worker connection building block; not a worker launcher.

The service must authenticate the actual worker process, and the launcher
must protect its code, imports, environment, arguments and database selection.
This module alone does not establish those properties or enable host Bash.
"""
from __future__ import annotations

import os
import sys

from .service_entrypoint import _absolute_parts, _open_checked, _read_json
from ..experimental_ingress import queue_request_binding
from ..transport import macos_authority_client
from ...core.intelligence.simple_loop.owner_authority_bridge import OwnerAuthorityBridge


# Provisioned by an administrator, never chosen by a conversation or process
# environment. The entire path is checked before any content is read.
_CONFIG_PATH = '/private/var/db/alphonse/worker-authority.json'


class WorkerEndpointUnavailable(ValueError):
    pass


def _validated_config(config, *, real_uid, effective_uid):
    if (not isinstance(config, dict) or set(config) != {'version', 'service_uid', 'owner_uid', 'socket_path'}
            or type(config['version']) is not int or config['version'] != 1
            or type(real_uid) is not int or type(effective_uid) is not int
            or type(config['owner_uid']) is not int or type(config['service_uid']) is not int
            or not 0 < config['service_uid'] < 0xffffffff
            or not 0 < config['owner_uid'] < 0xffffffff
            or config['owner_uid'] != real_uid or real_uid != effective_uid
            or config['service_uid'] == effective_uid):
        raise WorkerEndpointUnavailable('worker_authority_configuration_unavailable')
    try:
        _absolute_parts(config['socket_path'])
    except (ValueError, TypeError):
        raise WorkerEndpointUnavailable('worker_authority_configuration_unavailable') from None
    return config


def provisioned_worker_bridge(queue):
    """Create a worker bridge from fixed root-owned configuration.

    This does not authenticate the worker or start a connection. Service peer
    authentication occurs at connection time; each request must still be
    bound and claimed by the authority service. The caller must run under an
    independently protected launcher before integrating this into a daemon.
    """
    if sys.platform != 'darwin':
        raise WorkerEndpointUnavailable('worker_authority_configuration_unavailable')
    try:
        fd = _open_checked(_CONFIG_PATH, directory_owners={0}, final_owners={0}, final_kind='file')
        try:
            config = _validated_config(_read_json(fd), real_uid=os.getuid(), effective_uid=os.geteuid())
        finally:
            os.close(fd)
        client = macos_authority_client(config['socket_path'], service_uid=config['service_uid'])
        return OwnerAuthorityBridge(client, lambda task: queue_request_binding(queue, task))
    except Exception:
        raise WorkerEndpointUnavailable('worker_authority_configuration_unavailable') from None
