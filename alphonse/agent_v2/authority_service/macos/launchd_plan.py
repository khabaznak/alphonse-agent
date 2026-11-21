"""Build a reviewable launchd plist in memory; this module does not install it.

The returned plist assumes separately provisioned, root-protected code and
configuration, an existing dedicated service account, and trusted signed client
identities. Generating a plist does not establish any of those protections.
"""
from __future__ import annotations

import re

from .service_entrypoint import ServiceConfigurationError, _absolute_parts, validate_config


def build_launchd_plist(config, *, service_user, interpreter_path, config_path, owner_gid):
    """Return a system-domain job definition, without reading or writing files."""
    if not isinstance(config, dict):
        raise ServiceConfigurationError('authority_service_configuration_unavailable')
    validate_config(config, effective_uid=config.get('service_uid'))
    _absolute_parts(interpreter_path)
    _absolute_parts(config_path)
    if (not isinstance(service_user, str) or re.fullmatch(r'[a-z_][a-z0-9_-]{0,31}', service_user) is None
            or service_user == 'root' or type(owner_gid) is not int or not 0 < owner_gid < 0xffffffff):
        raise ServiceConfigurationError('authority_service_configuration_unavailable')
    sockets = {}
    for label, role in (('Desktop', 'desktop'), ('Worker', 'worker')):
        sockets[label] = {
            'SockPathName': config[role]['socket_path'],
            'SockPathOwner': config['service_uid'],
            'SockPathGroup': owner_gid,
            'SockPathMode': 0o660,
            'SockType': 'stream',
            'SockPassive': True,
        }
    return {
        'Label': 'local.alphonse.authority',
        'UserName': service_user,
        'ProgramArguments': [interpreter_path, '-I', '-m',
                             'alphonse.agent_v2.authority_service.macos.service_entrypoint',
                             '--config', config_path, '--launchd'],
        'Sockets': sockets,
    }
