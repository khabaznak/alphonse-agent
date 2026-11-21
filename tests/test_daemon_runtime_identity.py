"""The identity report describes live bindings, without claiming a Git revision."""
from types import SimpleNamespace
import tempfile

import pytest

from alphonse.agent_v2.core.intelligence.simple_loop import owner_read_processor
from alphonse.agent_v2.daemon import V2Daemon, _configure_trusted_local_bash, _trusted_local_bash_enabled
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer


def owner_daemon():
    daemon = object.__new__(V2Daemon)
    processor = object.__new__(owner_read_processor.OwnerReadConversationProcessor)
    processor._owner_host_bash_enabled = False
    processor._allow_unprotected_host_bash_for_testing = False
    processor._experimental_owner_authority = None
    daemon.runtime = SimpleNamespace(processor=SimpleNamespace(owner_read_candidate=processor))
    daemon.experimental_owner_only = True
    daemon.daemon_id = 'test-daemon'
    daemon._processor_thread = SimpleNamespace(is_alive=lambda: True)
    daemon._identity_started_at = 'fixture'
    return daemon


def test_identity_reports_loaded_simple_loop_and_detects_changed_binding(monkeypatch):
    daemon = owner_daemon()
    identity = daemon.runtime_identity()
    assert identity['loop'] == 'simple_loop'
    assert identity['loop_status'] == 'configured_worker_alive'
    assert identity['owner_host_bash_mode'] == 'disabled'
    assert identity['source_revision'] is None
    assert set(identity['loaded_code_sha256']) == {'owner_worker', 'owner_processor', 'simple_loop_candidate'}
    assert all(len(value) == 64 for value in identity['loaded_code_sha256'].values())
    monkeypatch.setattr(owner_read_processor, 'run_candidate', lambda *_: None)
    changed = daemon.runtime_identity()
    assert changed['loop'] == 'unknown'
    assert changed['loop_status'] == 'unverified'
    assert changed['owner_host_bash_mode'] == 'unverified'
    assert changed['loaded_code_sha256']['simple_loop_candidate'] != identity['loaded_code_sha256']['simple_loop_candidate']


def test_identity_does_not_claim_running_worker_when_thread_is_stopped():
    daemon = owner_daemon()
    daemon._processor_thread = None
    assert daemon.runtime_identity()['loop_status'] == 'configured_worker_not_alive'
    daemon.experimental_owner_only = False
    assert daemon.runtime_identity()['loop'] == 'unknown'
    assert daemon.runtime_identity()['owner_host_bash_mode'] == 'unverified'


def test_host_bash_identity_reports_configuration_without_claiming_security():
    daemon = owner_daemon()
    daemon.runtime.processor.owner_read_candidate._owner_host_bash_enabled = True
    assert daemon.runtime_identity()['owner_host_bash_mode'] == 'blocked_missing_authority'
    daemon.runtime.processor.owner_read_candidate._allow_unprotected_host_bash_for_testing = True
    assert daemon.runtime_identity()['owner_host_bash_mode'] == 'experimental_unprotected'
    daemon.runtime.processor.owner_read_candidate._allow_unprotected_host_bash_for_testing = False
    daemon.runtime.processor.owner_read_candidate._experimental_owner_authority = object()
    assert daemon.runtime_identity()['owner_host_bash_mode'] == 'experimental_authority_unverified'
    _configure_trusted_local_bash(daemon.runtime, enabled=True)
    assert daemon.runtime_identity()['owner_host_bash_mode'] == 'trusted_local_unprotected'
    daemon.runtime.processor.owner_read_candidate._owner_host_bash_enabled = 'true'
    assert daemon.runtime_identity()['owner_host_bash_mode'] == 'unverified'


def test_trusted_local_bash_requires_explicit_valid_preference(tmp_path):
    setting = tmp_path / 'development.json'
    assert not _trusted_local_bash_enabled(setting)
    for text in ('{}', '{"trusted_local_bash":false}', '{"trusted_local_bash":1}',
                 '{"trusted_local_bash":true,"other":true}', '{invalid'):
        setting.write_text(text)
        assert not _trusted_local_bash_enabled(setting)
    setting.write_text('{"trusted_local_bash":true}')
    assert _trusted_local_bash_enabled(setting)
    setting.unlink()
    assert not _trusted_local_bash_enabled(setting)


def test_ipc_exposes_read_only_identity_and_rejects_parameters():
    server = V2DaemonServer(owner_daemon())
    result = server._dispatch({'method': 'runtime_identity', 'params': {}})
    assert result['loop'] == 'simple_loop'
    with pytest.raises(ValueError, match='runtime_identity_payload_denied'):
        server._dispatch({'method': 'runtime_identity', 'params': {'unexpected': 1}})


def test_identity_round_trip_through_local_socket():
    with tempfile.TemporaryDirectory(prefix='gym-identity-', dir='/private/tmp') as directory:
        server = V2DaemonServer(owner_daemon(), socket_path=directory + '/i.sock')
        server.start()
        try:
            identity = V2DaemonClient(server.socket_path).runtime_identity()
            assert identity['loop'] == 'simple_loop'
            assert identity['daemon_id'] == 'test-daemon'
        finally:
            server.stop()
