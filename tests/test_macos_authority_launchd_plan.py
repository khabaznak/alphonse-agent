import plistlib

import pytest

from alphonse.agent_v2.authority_service.macos.launchd_plan import build_launchd_plist
from alphonse.agent_v2.authority_service.macos.service_entrypoint import ServiceConfigurationError


def fixture():
    return {'version': 1, 'service_uid': 501, 'owner_uid': 502, 'actor': 'synthetic-owner',
            'ledger_path': '/protected/ledger.sqlite', 'verifier_library_path': '/protected/verifier.dylib',
            'desktop': {'cdhash': 'a' * 40, 'client_id': 'desktop', 'socket_path': '/protected/desktop.sock'},
            'worker': {'cdhash': 'b' * 40, 'client_id': 'worker', 'socket_path': '/protected/worker.sock'}}


def test_plan_matches_service_configuration_without_installing_anything():
    value = build_launchd_plist(fixture(), service_user='_authority', interpreter_path='/protected/bin/python3',
                                config_path='/protected/service.json', owner_gid=20)
    assert plistlib.loads(plistlib.dumps(value)) == value
    assert value['UserName'] == '_authority'
    assert value['ProgramArguments'] == ['/protected/bin/python3', '-I', '-m',
        'alphonse.agent_v2.authority_service.macos.service_entrypoint', '--config', '/protected/service.json', '--launchd']
    assert value['Sockets']['Desktop']['SockPathName'] == fixture()['desktop']['socket_path']
    assert value['Sockets']['Worker']['SockPathName'] == fixture()['worker']['socket_path']
    assert value['Sockets']['Desktop']['SockPathMode'] == 0o660
    assert value['Sockets']['Desktop']['SockPathOwner'] == fixture()['service_uid']


@pytest.mark.parametrize('kwargs', [
    {'service_user': 'root'}, {'service_user': 'x/unsafe'}, {'owner_gid': 0},
    {'interpreter_path': '/protected/../user/python'}, {'config_path': 'relative'},
])
def test_plan_rejects_unprovisionable_values(kwargs):
    values = {'service_user': '_authority', 'interpreter_path': '/protected/bin/python3',
              'config_path': '/protected/service.json', 'owner_gid': 20}
    with pytest.raises(ServiceConfigurationError):
        build_launchd_plist(fixture(), **{**values, **kwargs})
