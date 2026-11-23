import json
import threading
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from test_owner_atomic_checkpoint import setup
from test_owner_continuation_question import paused_checkpoint
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonServer
from alphonse.agent_v2.core.intelligence.simple_loop.single_loop_checkpoint import CheckpointConflict


def daemon_for(queue, store):
    daemon = object.__new__(V2Daemon)
    daemon.runtime = SimpleNamespace(queue=queue, processor=SimpleNamespace(
        owner_read_candidate=SimpleNamespace(atomic_checkpoints=store)))
    daemon._stop = threading.Event()
    daemon.kill_switch = SimpleNamespace(is_cancelled=lambda _: False)
    return daemon


def test_choice_survives_old_lease_expiry_and_binds_question(setup):
    queue, _, task, store, _, _ = paused_checkpoint(setup)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_inbound_messages SET lease_expires_at=? WHERE message_id=?',
                     ((datetime.now(timezone.utc) - timedelta(minutes=2)).isoformat(), task.message_id))
    daemon = daemon_for(queue, store)
    first = daemon.owner_continuation_list()
    assert len(first) == 1 and first[0]['message_id'] == task.message_id
    assert daemon.owner_continuation_list() == first  # discovery does not duplicate a question
    item = first[0]
    with pytest.raises(CheckpointConflict):
        daemon.owner_continuation_choose(task.message_id, item['checkpoint_version'] - 1, True)
    assert daemon.owner_continuation_choose(task.message_id, item['checkpoint_version'], True) == {'status': 'accepted'}
    assert daemon.owner_continuation_list() == []
    with pytest.raises(CheckpointConflict):
        daemon.owner_continuation_choose(task.message_id, item['checkpoint_version'], True)
    with queue._connect() as conn:
        raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                      (task.task_id,)).fetchone()[0])['metadata']
    assert raw['owner_continuation_choice']['approved'] is True
    assert raw['owner_continuation_handoff']['status'] == 'held'
    assert raw['owner_continuation_grant']['status'] == 'consumed'


def test_decline_never_creates_handoff(setup):
    queue, _, task, store, _, _ = paused_checkpoint(setup)
    daemon = daemon_for(queue, store)
    item = daemon.owner_continuation_list()[0]
    assert daemon.owner_continuation_choose(task.message_id, item['checkpoint_version'], False) == {'status': 'declined'}
    with queue._connect() as conn:
        raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                      (task.task_id,)).fetchone()[0])['metadata']
    assert raw['owner_continuation_choice']['approved'] is False
    assert 'owner_continuation_handoff' not in raw


def test_cancelled_choice_does_not_grant(setup):
    queue, _, task, store, _, _ = paused_checkpoint(setup)
    daemon = daemon_for(queue, store)
    item = daemon.owner_continuation_list()[0]
    daemon.kill_switch = SimpleNamespace(is_cancelled=lambda _: True)
    with pytest.raises(CheckpointConflict):
        daemon.owner_continuation_choose(task.message_id, item['checkpoint_version'], True)


def test_ipc_requires_local_peer_and_rejects_client_supplied_authority(setup, monkeypatch):
    queue, _, task, store, _, _ = paused_checkpoint(setup)
    daemon = daemon_for(queue, store)
    daemon.experimental_owner_only = True
    daemon._continuation_experimental_enabled = True
    server = V2DaemonServer(daemon, experimental_owner_ingress=True)
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: 1)
    item = server._owner_continuation(None, {'version': 1, 'method': 'experimental_owner_continuation_list',
                                               'params': {}})['items'][0]
    with pytest.raises(ValueError):
        server._owner_continuation(None, {'version': 1, 'method': 'experimental_owner_continuation_choose',
            'params': {'message_id': task.message_id, 'checkpoint_version': item['checkpoint_version'],
                       'approve': True, 'actor': 'owner'}})
    def denied(_):
        raise PermissionError('peer denied')
    monkeypatch.setattr(server, '_local_peer_uid', denied)
    with pytest.raises(PermissionError):
        server._owner_continuation(None, {'version': 1, 'method': 'experimental_owner_continuation_choose',
            'params': {'message_id': task.message_id, 'checkpoint_version': item['checkpoint_version'], 'approve': True}})


def test_killswitch_persists_veto_before_reporting_success(setup):
    queue, _, task, store, _, _ = paused_checkpoint(setup)
    daemon = daemon_for(queue, store)
    daemon.experimental_owner_only = True
    daemon.runtime.user_store = SimpleNamespace(is_admin=lambda _: True)
    active = SimpleNamespace(message_id=task.message_id, task_id=task.task_id, user_id=task.user, project_id='')
    daemon.kill_switch = SimpleNamespace(request_cancel_status=lambda: ('cancel_requested', active))
    daemon.kill_switch_audit = SimpleNamespace(record=lambda **_: 'audit')
    result = daemon.trigger_killswitch(actor_user_id=task.user)
    assert result['status'] == 'cancel_requested'
    with queue._connect() as conn:
        assert conn.execute('SELECT owner_id FROM v2_owner_continuation_cancellations WHERE message_id=?',
                            (task.message_id,)).fetchone()['owner_id'] == task.user
    assert daemon.owner_continuation_list() == []


def test_checkpoint_settings_are_explicit_and_validate_limits(setup):
    _, users, _, queue, _, _, store, _ = setup
    daemon = daemon_for(queue, store)
    daemon.runtime.user_store = users
    assert daemon.owner_continuation_settings() == {'enabled': False}
    assert daemon.set_owner_continuation_settings({'enabled': True, 'metric': 'actions', 'interval': 3}) == {
        'enabled': True, 'metric': 'actions', 'interval': 3}
    with pytest.raises(ValueError):
        daemon.set_owner_continuation_settings({'enabled': True, 'metric': 'actions', 'interval': 10})
    assert daemon.set_owner_continuation_settings({'enabled': False}) == {'enabled': False}
