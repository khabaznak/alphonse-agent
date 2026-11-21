import json
import os
import socket
import sys
import tempfile
from pathlib import Path

import pytest

from alphonse.agent_v2.authority_service.macos.service_entrypoint import (
    ServiceConfigurationError, _absolute_parts, _listener, _open_checked, _read_json, main, validate_config,
)


def config():
    return {'version': 1, 'service_uid': 501, 'owner_uid': 502, 'actor': 'synthetic-owner',
            'ledger_path': '/protected/ledger.sqlite', 'verifier_library_path': '/protected/verifier.dylib',
            'desktop': {'team': 'ABCDEFGHI1', 'identifier': 'fixture.desktop', 'client_id': 'desktop',
                        'socket_path': '/protected/desktop.sock'},
            'worker': {'team': 'ABCDEFGHI1', 'identifier': 'fixture.worker', 'client_id': 'worker',
                       'socket_path': '/protected/worker.sock'}}


def test_refuses_root_owner_wrong_uid_and_ambiguous_role_identities():
    value = config()
    assert validate_config(value, effective_uid=501) == value
    for uid in (0, 502, 503):
        with pytest.raises(ServiceConfigurationError):
            validate_config(value, effective_uid=uid)
    with pytest.raises(ServiceConfigurationError):
        validate_config({**value, 'service_uid': 502}, effective_uid=502)
    with pytest.raises(ServiceConfigurationError):
        validate_config({**value, 'worker': value['desktop']}, effective_uid=501)
    with pytest.raises(ServiceConfigurationError):
        validate_config({**value, 'version': True}, effective_uid=501)


def test_pinned_code_hash_mode_is_explicit_distinct_and_never_mixed():
    value = config()
    value['desktop'] = {'cdhash': 'a' * 40, 'client_id': 'desktop', 'socket_path': '/protected/desktop.sock'}
    value['worker'] = {'cdhash': 'b' * 40, 'client_id': 'worker', 'socket_path': '/protected/worker.sock'}
    assert validate_config(value, effective_uid=501) == value
    for bad_hash in ('a' * 39, 'A' * 40, 'g' * 40, 'a' * 41, 'a' * 40):
        invalid = {**value, 'worker': {**value['worker'], 'cdhash': bad_hash}}
        with pytest.raises(ServiceConfigurationError):
            validate_config(invalid, effective_uid=501)
    mixed = {**value, 'worker': config()['worker']}
    with pytest.raises(ServiceConfigurationError):
        validate_config(mixed, effective_uid=501)


@pytest.mark.parametrize('path', ['/protected/../bad', '/protected//bad', 'relative', '/', '/a/./b'])
def test_rejects_ambiguous_or_relative_paths(path):
    with pytest.raises(ServiceConfigurationError):
        _absolute_parts(path)


def test_duplicate_configuration_fields_are_rejected(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text('{"version":1,"version":1}')
    fd = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(ServiceConfigurationError):
            _read_json(fd)
    finally:
        os.close(fd)


@pytest.mark.skipif(sys.flags.isolated == 1, reason='requires ordinary non-isolated test interpreter')
def test_entrypoint_refuses_nonisolated_interpreter(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['service', '--config', '/tmp/untrusted', '--desktop-fd', '3', '--worker-fd', '4'])
    with pytest.raises(ServiceConfigurationError):
        main()


def test_prebound_listener_must_match_provisioned_socket_name(monkeypatch):
    # Only the root-owned directory check is simulated; binding and socket
    # properties are real and the fixture is not a production trust boundary.
    with tempfile.TemporaryDirectory(prefix='alpha-listener-', dir='/private/tmp') as directory:
        path = str(Path(directory) / 'desktop.sock')
        alternate = str(Path(directory) / 'worker.sock')
        monkeypatch.setattr('alphonse.agent_v2.authority_service.macos.service_entrypoint._open_checked',
                            lambda *_args, **_kwargs: os.open(directory, os.O_RDONLY))
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as source:
            source.bind(path)
            source.listen(1)
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as other:
                other.bind(alternate)
                other.listen(1)
                duplicate = _listener(source.fileno(), path, os.geteuid(), lambda _: True)
                try:
                    assert duplicate.getsockname() == path
                finally:
                    duplicate.close()
                with pytest.raises(ServiceConfigurationError):
                    _listener(source.fileno(), alternate, os.geteuid(), lambda _: True)
                with pytest.raises(ServiceConfigurationError):
                    _listener(source.fileno(), path, os.geteuid(), lambda _: False)


@pytest.mark.skipif(sys.platform != 'darwin', reason='Darwin ACL semantics')
def test_user_writable_ancestor_cannot_be_a_protected_configuration(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config()))
    with pytest.raises(ServiceConfigurationError):
        _open_checked(str(path), directory_owners={0}, final_owners={0}, final_kind='file')
