"""Disposable read-only owner inventory: counts are not recovery authority."""
import json
import os
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.legacy_state_inventory import inspect_legacy_state
from alphonse.agent_v2.core.messages.legacy_state_items import inspect_legacy_items
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.users import V2UserStore


@pytest.fixture
def state(tmp_path):
    path = tmp_path / 'fixture.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    member = users.create_user(display_name='Synthetic member', role='member')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    questions = SQLiteQuestionStore(path)
    outbox = SQLiteOutboundStore(path)
    schedules = ScheduledTaskStore(path)
    with sqlite3.connect(path) as conn:
        conn.execute('''CREATE TABLE v2_trusted_owner_ingress
            (message_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, peer_uid INTEGER, payload_sha256 TEXT, history_version INTEGER)''')
        conn.executemany('INSERT INTO v2_trusted_owner_ingress VALUES (?,?,?,?,?)', [
            ('trusted', owner.user_id, os.getuid(), 'a' * 64, 1),
            ('tampered', owner.user_id, -1, 'invalid-digest', 99),
            ('mismatched-owner', member.user_id, os.getuid(), 'a' * 64, 1),
        ])
        for sequence, (identity, user, status, metadata) in enumerate([
            ('old-v3', owner.user_id, 'pending', '{"intelligence_engine":"hierarchical_v3"}'),
            ('old-broken', owner.user_id, 'failed', '{"unparseable"'),
            ('trusted', owner.user_id, 'processing', '{"intelligence_engine":"single_loop_experimental"}'),
            ('tampered', owner.user_id, 'pending', '{"intelligence_engine":"tactical_v2"}'),
            ('mismatched-owner', owner.user_id, 'failed', '{"intelligence_engine":"tactical_v2"}'),
            ('member-private', member.user_id, 'retry_wait', '{"intelligence_engine":"tactical_v2"}'),
        ], 1):
            conn.execute('''INSERT INTO v2_inbound_messages
                (message_id,sequence,timestamp,prompt,user_id,metadata_json,queued_at,status)
                VALUES (?,?,?,?,?,?,?,?)''',
                (identity, sequence, '2026-01-01T00:00:00Z', 'PRIVATE SYNTHETIC TEXT', user, metadata, '2026-01-01T00:00:00Z', status))
        for identity, user, status, payload in [
            ('checkpoint-v2', owner.user_id, 'running', '{"intelligence_engine":"tactical_v2","goal":"PRIVATE"}'),
            ('checkpoint-missing', owner.user_id, 'failed', '{}'),
            ('trusted', owner.user_id, 'running', '{"intelligence_engine":"single_loop_experimental"}'),
            ('member-checkpoint', member.user_id, 'failed', '{"intelligence_engine":"hierarchical_v3"}'),
        ]:
            conn.execute('''INSERT INTO v2_task_checkpoints(task_id,task_state_json,status,owner_id,created_at,updated_at)
                VALUES (?,?,?,?,?,?)''', (identity, payload, status, user, 'synthetic', 'synthetic'))
        for identity, audience, status, attempts in [
            ('one', owner.user_id, 'pending', 0), ('two', owner.user_id, 'unconfirmed', 2),
            ('private-member', member.user_id, 'failed', 9), ('unassigned', '', 'pending', 0),
        ]:
            conn.execute('''INSERT INTO v2_outbox(outbox_message_id,integration_id,provider_key,channel_target,
                message,kind,audience_user_id,status,created_at,updated_at,attempt_count)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)''',
                (identity, 'fixture', 'fixture', 'opaque-target', 'PRIVATE SYNTHETIC TEXT',
                 'message', audience, status, 'synthetic', 'synthetic', attempts))
        for identity, user, status in [('owner-job', owner.user_id, 'active'), ('member-job', member.user_id, 'paused')]:
            conn.execute('''INSERT INTO v2_scheduled_tasks
                (scheduled_task_id,owner_user_id,name,prompt,schedule_json,timezone,status,created_at,updated_at)
                VALUES (?,?,?,?,?,?,?,?,?)''', (identity, user, 'PRIVATE JOB', 'PRIVATE PROMPT', '{}', 'UTC', status, 'synthetic', 'synthetic'))
            conn.execute('''INSERT INTO v2_scheduled_task_executions
                (scheduled_task_id,run_id,status,started_at) VALUES (?,?,?,?)''',
                (identity, identity + '-run', 'unconfirmed', 'synthetic'))
    runtime = SimpleNamespace(queue=queue, user_store=users, question_store=questions, outbox=outbox, schedule_store=schedules)
    daemon = SimpleNamespace(runtime=runtime, experimental_owner_only=True)
    return path, users, owner, member, daemon


def test_inventory_is_owner_scoped_and_never_exposes_payloads(state):
    path, _, owner, member, _ = state
    result = inspect_legacy_state(path, expected_owner=owner.user_id, include_schedules=True)
    assert result is not None
    queue = result['queue_other_ingress']
    assert queue['statuses']['pending'] == 1 and queue['statuses']['failed'] == 2
    assert queue['statuses']['processing'] == 0 and queue['statuses']['retry_wait'] == 0
    assert queue['engine_stamps']['hierarchical_v3'] == 1 and queue['engine_stamps']['malformed'] == 1
    assert queue['engine_stamps']['tactical_v2'] == 1
    # Matching origin markers are counted separately, including invalid ones;
    # this diagnostic never represents marker presence as verified authority.
    marked = result['queue_owner_origin_marked']
    assert marked['statuses']['pending'] == 1 and marked['statuses']['processing'] == 1
    assert marked['engine_stamps']['tactical_v2'] == 1
    checkpoints = result['checkpoints_other_ingress']
    assert checkpoints['statuses']['running'] == 1 and checkpoints['statuses']['failed'] == 1
    assert checkpoints['engine_stamps']['tactical_v2'] == 1 and checkpoints['engine_stamps']['missing'] == 1
    outbox = result['owner_addressed_outbox']
    assert outbox['statuses']['pending'] == 1 and outbox['statuses']['unconfirmed'] == 1
    assert outbox['statuses']['failed'] == 0
    assert outbox['attempt_counter']['zero'] == 1 and outbox['attempt_counter']['positive'] == 1
    scheduled = result['owner_scheduled_work']
    assert scheduled['status'] == 'available'
    assert scheduled['tasks']['active'] == 1 and scheduled['tasks']['paused'] == 0
    assert scheduled['occurrences']['unconfirmed'] == 1
    rendered = json.dumps(result)
    for secret in ('PRIVATE', 'old-v3', 'member-private', owner.user_id, member.user_id, 'opaque-target'):
        assert secret not in rendered
    assert inspect_legacy_state(path, expected_owner=member.user_id) is None


def test_item_inspection_is_bounded_owner_scoped_and_read_only(state):
    path, _, owner, member, _ = state
    with sqlite3.connect(path) as conn:
        before = list(conn.iterdump())
    result = inspect_legacy_items(path, expected_owner=owner.user_id)
    assert result is not None
    queue = {item['id']: item for item in result['queue']['items']}
    assert queue['old-v3'] == {'id': 'old-v3', 'status': 'pending', 'engine': 'hierarchical_v3', 'origin_marker': 'absent'}
    assert queue['tampered']['origin_marker'] == 'present'  # Presence is not verification.
    assert queue['old-broken']['engine'] == 'malformed'
    assert 'member-private' not in queue
    assert result['queue']['truncated'] is False
    rendered = json.dumps(result)
    assert 'PRIVATE' not in rendered and member.user_id not in rendered and 'member-checkpoint' not in rendered
    assert inspect_legacy_items(path, expected_owner=member.user_id) is None
    with sqlite3.connect(path) as conn:
        assert list(conn.iterdump()) == before
        conn.executemany('''INSERT INTO v2_inbound_messages
            (message_id,sequence,timestamp,prompt,user_id,metadata_json,queued_at,status)
            VALUES (?,?,?,?,?,?,?,?)''',
            [(f'extra-{i:03}', 100 + i, 'fixture', 'SECRET', owner.user_id, '{}', 'fixture', 'pending')
             for i in range(55)])
    limited = inspect_legacy_items(path, expected_owner=owner.user_id)
    assert limited['queue']['truncated'] is True and len(limited['queue']['items']) == 50


def test_item_inspection_ipc_enforces_peer_and_same_store(state, tmp_path, monkeypatch):
    path, _, owner, member, daemon = state
    server = V2DaemonServer(daemon, '/private/tmp/unused-inventory.sock', experimental_owner_ingress=True)
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: os.getuid())
    request = {'version': 1, 'method': 'legacy_state_items', 'params': {'actor_user_id': owner.user_id}}
    assert server._legacy_state_items(None, request)['status'] == 'available'
    request['params']['actor_user_id'] = member.user_id
    assert server._legacy_state_items(None, request) == {'status': 'unavailable'}
    request['params']['actor_user_id'] = owner.user_id
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: os.getuid() + 1)
    with pytest.raises(PermissionError):
        server._legacy_state_items(None, request)
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: os.getuid())
    daemon.runtime.question_store = SQLiteQuestionStore(tmp_path / 'split.sqlite')
    with pytest.raises(PermissionError):
        server._legacy_state_items(None, request)


def test_item_inspection_missing_origin_and_busy_are_explicit(state):
    path, users, owner, _, _ = state
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TABLE v2_trusted_owner_ingress')
    result = inspect_legacy_items(path, expected_owner=owner.user_id)
    assert all(item['origin_marker'] == 'unavailable' for item in result['queue']['items'])
    with sqlite3.connect(path) as writer:
        writer.execute('BEGIN IMMEDIATE')
        assert inspect_legacy_items(path, expected_owner=owner.user_id) is None
        writer.rollback()
    users.update_user(owner.user_id, is_active=False)
    assert inspect_legacy_items(path, expected_owner=owner.user_id) is None


def test_scheduled_inventory_unavailable_for_missing_schema_or_split_store(state, tmp_path, monkeypatch):
    path, _, owner, _, daemon = state
    assert inspect_legacy_state(path, expected_owner=owner.user_id)['owner_scheduled_work'] == {'status': 'unavailable'}
    server = V2DaemonServer(daemon, '/private/tmp/unused-inventory.sock', experimental_owner_ingress=True)
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: os.getuid())
    request = {'version': 1, 'method': 'legacy_state_inventory', 'params': {'actor_user_id': owner.user_id}}
    assert server._legacy_state_inventory(None, request)['inventory']['owner_scheduled_work']['status'] == 'available'
    daemon.runtime.schedule_store = ScheduledTaskStore(tmp_path / 'separate.sqlite')
    assert server._legacy_state_inventory(None, request)['inventory']['owner_scheduled_work'] == {'status': 'unavailable'}
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TABLE v2_scheduled_task_executions')
    assert inspect_legacy_state(path, expected_owner=owner.user_id, include_schedules=True)['owner_scheduled_work'] == {'status': 'unavailable'}


def test_inventory_is_read_only_and_fails_closed_on_missing_or_corrupt_schema(state, tmp_path):
    path, _, owner, _, _ = state
    with sqlite3.connect(path) as conn:
        before = list(conn.iterdump())
        counts = [conn.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0] for name in
                  ('v2_inbound_messages', 'v2_task_checkpoints', 'v2_outbox')]
    assert inspect_legacy_state(path, expected_owner=owner.user_id) is not None
    with sqlite3.connect(path) as conn:
        assert [conn.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0] for name in
                ('v2_inbound_messages', 'v2_task_checkpoints', 'v2_outbox')] == counts
        assert list(conn.iterdump()) == before
        conn.execute('DROP TABLE v2_task_checkpoints')
    assert inspect_legacy_state(path, expected_owner=owner.user_id) is None
    bad = tmp_path / 'corrupt.sqlite'
    bad.write_bytes(b'not sqlite')
    assert inspect_legacy_state(bad) is None
    assert inspect_legacy_state(tmp_path / 'missing.sqlite') is None


def test_inactive_or_noncanonical_actor_receives_no_inventory(state):
    path, users, owner, member, _ = state
    other_owner = users.create_user(display_name='Synthetic other owner', role='owner')
    assert inspect_legacy_state(path, expected_owner=member.user_id) is None
    assert inspect_legacy_state(path, expected_owner=other_owner.user_id) is None
    users.update_user(owner.user_id, is_active=False)
    assert inspect_legacy_state(path, expected_owner=owner.user_id) is None
    assert inspect_legacy_state(path) is None


def test_inventory_reserves_owner_authority_and_never_writes(state, monkeypatch):
    from alphonse.agent_v2.core.messages import legacy_state_inventory as module
    path, _, owner, _, _ = state
    original_counts = module._counts
    observed = []

    def try_revoke_while_counting(rows, known, column):
        if not observed:
            with sqlite3.connect(path, timeout=0) as concurrent:
                with pytest.raises(sqlite3.OperationalError, match='locked'):
                    concurrent.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner.user_id,))
            observed.append(True)
        return original_counts(rows, known, column)

    monkeypatch.setattr(module, '_counts', try_revoke_while_counting)
    assert inspect_legacy_state(path, expected_owner=owner.user_id) is not None
    assert observed == [True]
    with sqlite3.connect(path) as conn:
        assert conn.execute('SELECT is_active FROM v2_users WHERE user_id=?', (owner.user_id,)).fetchone()[0] == 1
        conn.execute('UPDATE v2_users SET is_active=0 WHERE user_id=?', (owner.user_id,))
    assert inspect_legacy_state(path, expected_owner=owner.user_id) is None


def test_busy_inventory_is_unavailable_without_mutation(state):
    path, _, owner, _, _ = state
    with sqlite3.connect(path) as writer:
        writer.execute('BEGIN IMMEDIATE')
        assert inspect_legacy_state(path, expected_owner=owner.user_id) is None
        writer.rollback()
    assert inspect_legacy_state(path, expected_owner=owner.user_id) is not None


def test_inventory_query_budget_fails_closed_and_releases_lock(state, monkeypatch):
    from alphonse.agent_v2.core.messages import legacy_state_inventory as module
    path, _, owner, _, _ = state
    monkeypatch.setattr(module, '_MAX_VM_CALLBACKS', 0)
    assert inspect_legacy_state(path, expected_owner=owner.user_id) is None
    with sqlite3.connect(path, timeout=0) as conn:
        conn.execute('BEGIN IMMEDIATE')
        conn.rollback()


def test_ipc_requires_owner_mode_verified_peer_and_empty_params(state, monkeypatch):
    _, users, owner, _, daemon = state
    server = V2DaemonServer(daemon, '/private/tmp/unused-inventory.sock', experimental_owner_ingress=True)
    request = {'version': 1, 'method': 'legacy_state_inventory', 'params': {'actor_user_id': owner.user_id}}
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: os.getuid() + 1)
    with pytest.raises(PermissionError, match='legacy_state_inventory_unavailable'):
        server._legacy_state_inventory(None, request)
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: os.getuid())
    with pytest.raises(ValueError, match='legacy_state_inventory_payload_denied'):
        server._legacy_state_inventory(None, {**request, 'params': {'owner': 'other'}})
    assert server._legacy_state_inventory(None, request)['status'] == 'available'
    noncanonical = users.create_user(display_name='Second synthetic owner', role='owner')
    assert server._legacy_state_inventory(None, {**request, 'params': {'actor_user_id': noncanonical.user_id}}) == {'status': 'unavailable'}
    users._set_setting('admin_user_id', noncanonical.user_id)
    assert server._legacy_state_inventory(None, request) == {'status': 'unavailable'}
    assert server._legacy_state_inventory(None, {**request, 'params': {'actor_user_id': noncanonical.user_id}})['status'] == 'available'
    users._set_setting('admin_user_id', owner.user_id)
    users.update_user(owner.user_id, is_active=False)
    assert server._legacy_state_inventory(None, request) == {'status': 'unavailable'}
    daemon.experimental_owner_only = False
    with pytest.raises(PermissionError, match='legacy_state_inventory_unavailable'):
        server._legacy_state_inventory(None, request)


def test_ipc_refuses_split_stores(state, tmp_path, monkeypatch):
    _, _, owner, _, daemon = state
    daemon.runtime.question_store = SQLiteQuestionStore(tmp_path / 'different.sqlite')
    server = V2DaemonServer(daemon, '/private/tmp/unused-inventory.sock', experimental_owner_ingress=True)
    monkeypatch.setattr(server, '_local_peer_uid', lambda _: os.getuid())
    with pytest.raises(PermissionError, match='legacy_state_inventory_unavailable'):
        server._legacy_state_inventory(None, {'version': 1, 'method': 'legacy_state_inventory', 'params': {'actor_user_id': owner.user_id}})


def test_inventory_ipc_uses_real_local_socket(state):
    _, _, owner, _, daemon = state
    with tempfile.TemporaryDirectory(prefix='legacy-inventory-', dir='/private/tmp') as directory:
        server = V2DaemonServer(daemon, Path(directory) / 'daemon.sock', experimental_owner_ingress=True)
        server.start()
        try:
            client = V2DaemonClient(server.socket_path)
            result = client.request('legacy_state_inventory', actor_user_id=owner.user_id)
            assert result['status'] == 'available'
            assert result['inventory']['queue_other_ingress']['statuses']['failed'] == 2
            records = client.request('legacy_state_items', actor_user_id=owner.user_id)
            assert records['status'] == 'available'
            assert any(item['id'] == 'old-v3' for item in records['items']['queue']['items'])
            with pytest.raises(RuntimeError, match='legacy_state_inventory_payload_denied'):
                client.request('legacy_state_inventory', actor_user_id=owner.user_id, user='forged')
        finally:
            server.stop()
