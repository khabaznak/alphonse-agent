"""Worker configuration and bridge assembly, not a macOS security proof."""
import json
import os
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.authority_service.macos import worker_endpoint as endpoint
from alphonse.agent_v2.core.intelligence.simple_loop.owner_authority_bridge import OwnerAuthorityBridge


def _config():
    return {'version': 1, 'service_uid': 501, 'owner_uid': 502,
            'socket_path': '/private/var/run/alphonse/worker.sock'}


@pytest.mark.parametrize('changes', [
    {'version': True}, {'service_uid': True}, {'owner_uid': 501},
    {'service_uid': 0}, {'service_uid': 502}, {'socket_path': '../worker.sock'},
    {'socket_path': '/private/var/../worker.sock'}, {'unexpected': 1},
])
def test_rejects_ambiguous_or_untrusted_config(changes):
    with pytest.raises(endpoint.WorkerEndpointUnavailable):
        endpoint._validated_config(_config() | changes, real_uid=502, effective_uid=502)


def test_rejects_setuid_context():
    with pytest.raises(endpoint.WorkerEndpointUnavailable):
        endpoint._validated_config(_config(), real_uid=502, effective_uid=503)


def test_bridge_reads_fixed_root_checked_config_and_binds_queue(monkeypatch, tmp_path):
    source = tmp_path / 'test-config.json'
    source.write_text(json.dumps(_config()))
    observed = {}

    def checked(path, **kwargs):
        observed['path'], observed['policy'] = path, kwargs
        return os.open(source, os.O_RDONLY)

    class Client:
        def claim(self, **kwargs): pass
        def check_claim(self, **kwargs): pass
        def check_bash_capability(self, **kwargs): pass
        def reserve_action(self, **kwargs): pass
        def start_action(self, **kwargs): pass
        def record_process_return(self, **kwargs): pass

    def client(path, *, service_uid):
        observed['socket'], observed['uid'] = path, service_uid
        return Client()

    monkeypatch.setattr(endpoint.sys, 'platform', 'darwin')
    monkeypatch.setattr(endpoint.os, 'getuid', lambda: 502)
    monkeypatch.setattr(endpoint.os, 'geteuid', lambda: 502)
    monkeypatch.setattr(endpoint, '_open_checked', checked)
    monkeypatch.setattr(endpoint, 'macos_authority_client', client)
    monkeypatch.setattr(endpoint, 'queue_request_binding', lambda queue, task: (queue, task))
    queue, task = object(), SimpleNamespace()
    bridge = endpoint.provisioned_worker_bridge(queue)
    assert isinstance(bridge, OwnerAuthorityBridge)
    assert bridge._request_binding(task) == (queue, task)
    assert observed == {'path': '/private/var/db/alphonse/worker-authority.json',
                        'policy': {'directory_owners': {0}, 'final_owners': {0}, 'final_kind': 'file'},
                        'socket': '/private/var/run/alphonse/worker.sock', 'uid': 501}


def test_missing_or_unprotected_config_never_constructs_a_client(monkeypatch):
    monkeypatch.setattr(endpoint.sys, 'platform', 'darwin')
    def reject(*args, **kwargs):
        raise ValueError('unprotected')
    monkeypatch.setattr(endpoint, '_open_checked', reject)
    monkeypatch.setattr(endpoint, 'macos_authority_client', lambda *args, **kwargs: pytest.fail('connected'))
    with pytest.raises(endpoint.WorkerEndpointUnavailable):
        endpoint.provisioned_worker_bridge(object())
