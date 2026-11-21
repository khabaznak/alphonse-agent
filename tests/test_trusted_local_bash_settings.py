"""Desktop development preference; uses only temporary household-free files."""
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2 import daemon as daemon_module
from alphonse.agent_v2.core.intelligence.simple_loop import owner_read_processor
from alphonse.agent_v2.daemon import V2Daemon, _configure_trusted_local_bash, _trusted_local_bash_enabled
from alphonse.agent_v2.ipc import V2DaemonServer


def new_daemon():
    daemon = object.__new__(V2Daemon)
    processor = object.__new__(owner_read_processor.OwnerReadConversationProcessor)
    processor._owner_host_bash_enabled = False
    processor._allow_unprotected_host_bash_for_testing = False
    processor._trusted_local_host_bash = False
    processor._experimental_owner_authority = None
    daemon.runtime = SimpleNamespace(processor=SimpleNamespace(owner_read_candidate=processor))
    daemon.experimental_owner_only = True
    daemon.daemon_id = 'fixture'
    daemon._processor_thread = SimpleNamespace(is_alive=lambda: True)
    daemon._identity_started_at = 'fixture'
    return daemon


def dispatch(server, method, params=None):
    return server._dispatch({'method': method, 'params': params or {}})


def test_toggle_requires_restart_and_reports_actual_running_mode(tmp_path, monkeypatch):
    path = tmp_path / 'development.json'
    monkeypatch.setattr(daemon_module, '_trusted_local_bash_preference_path', lambda: path)
    daemon = new_daemon()
    server = V2DaemonServer(daemon)
    assert dispatch(server, 'get_trusted_local_bash_settings') == {
        'enabled': False, 'active': False, 'restart_required': False}
    assert dispatch(server, 'set_trusted_local_bash_settings', {'enabled': True}) == {
        'enabled': True, 'active': False, 'restart_required': True}
    assert json.loads(path.read_text()) == {'trusted_local_bash': True}
    # Reconstructing the worker models a restart. No household data or daemon is touched.
    restarted = new_daemon()
    _configure_trusted_local_bash(restarted.runtime, enabled=_trusted_local_bash_enabled())
    server = V2DaemonServer(restarted)
    assert dispatch(server, 'get_trusted_local_bash_settings') == {
        'enabled': True, 'active': True, 'restart_required': False}
    assert dispatch(server, 'set_trusted_local_bash_settings', {'enabled': False}) == {
        'enabled': False, 'active': True, 'restart_required': True}
    assert _trusted_local_bash_enabled() is False


def test_unverified_worker_never_claims_active_or_restart_status(tmp_path, monkeypatch):
    monkeypatch.setattr(daemon_module, '_trusted_local_bash_preference_path', lambda: tmp_path / 'development.json')
    daemon = new_daemon()
    daemon._processor_thread = None
    assert daemon.trusted_local_bash_settings() == {
        'enabled': False, 'active': None, 'restart_required': None}


@pytest.mark.parametrize('params', [{}, {'enabled': 1}, {'enabled': 'true'}, {'enabled': True, 'x': 1}])
def test_invalid_ipc_payload_never_creates_setting(tmp_path, monkeypatch, params):
    path = tmp_path / 'development.json'
    monkeypatch.setattr(daemon_module, '_trusted_local_bash_preference_path', lambda: path)
    with pytest.raises(ValueError, match='trusted_local_bash_setting_invalid'):
        dispatch(V2DaemonServer(new_daemon()), 'set_trusted_local_bash_settings', params)
    assert not path.exists()


def test_malformed_or_symlinked_setting_is_preserved(tmp_path, monkeypatch):
    path = tmp_path / 'development.json'
    monkeypatch.setattr(daemon_module, '_trusted_local_bash_preference_path', lambda: path)
    server = V2DaemonServer(new_daemon())
    path.write_text('{"other": "keep"}')
    with pytest.raises(ValueError, match='trusted_local_bash_setting_invalid'):
        dispatch(server, 'set_trusted_local_bash_settings', {'enabled': True})
    assert path.read_text() == '{"other": "keep"}'
    path.write_text('{"trusted_local_bash": false, "trusted_local_bash": true}')
    with pytest.raises(ValueError, match='trusted_local_bash_setting_invalid'):
        dispatch(server, 'set_trusted_local_bash_settings', {'enabled': False})
    assert path.read_text() == '{"trusted_local_bash": false, "trusted_local_bash": true}'
    path.unlink()
    target = tmp_path / 'external.json'
    target.write_text('{"trusted_local_bash": false}')
    path.symlink_to(target)
    with pytest.raises(ValueError):
        dispatch(server, 'set_trusted_local_bash_settings', {'enabled': True})
    assert target.read_text() == '{"trusted_local_bash": false}'
