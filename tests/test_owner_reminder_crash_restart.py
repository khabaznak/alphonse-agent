"""Process-death coverage for a committed local reminder, with disposable data."""
import json
import multiprocessing
import shutil
import sqlite3
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def _build(root, socket_dir, daemon_id, provider):
    db = Path(root) / 'fixture.sqlite'
    users = V2UserStore(db)
    if users.admin_user() is None:
        owner = users.onboard(display_name='Synthetic owner', users_root=Path(root) / 'users')
        users.set_timezone('UTC', owner.user_id)
    queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(db)
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline'))
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
        conversation_store=conversation, outbox=SQLiteOutboundStore(db), tools=registry, inference=router,
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id=daemon_id, experimental_owner_only=True)
    daemon.ipc = V2DaemonServer(daemon, str(Path(socket_dir) / f'{daemon_id}.sock'))
    return daemon, users, conversation


def _child_after_commit(root, socket_dir, signal, due, goal, before_commit=False):
    from alphonse.agent_v2 import daemon as module
    module.default_socket_path = lambda: Path(socket_dir) / 'daemon.sock'
    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            if self.calls == 1:
                value = {'kind': 'tool', 'tool_id': 'native.owner_clock', 'arguments': {}, 'text': None}
            else:
                value = {'kind': 'tool', 'tool_id': 'native.owner_reminder_create',
                         'arguments': {'message': 'call the plumber', 'due_at': due, 'request_quote': goal},
                         'text': None}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    daemon, _, _ = _build(root, socket_dir, 'first', Provider())
    if before_commit:
        create = OwnerDesktopReminders.create_locked
        def pause_inside_transaction(conn, *args, **kwargs):
            result = create(conn, *args, **kwargs)
            assert conn.in_transaction and result['state'] == 'pending'
            signal.put('inserted_uncommitted')
            while True:
                time.sleep(1)
        OwnerDesktopReminders.create_locked = staticmethod(pause_inside_transaction)
    else:
        commit = daemon.kill_switch.commit_if_active
        def pause_after_commit(message_id, callback):
            result = commit(message_id, callback)
            signal.put('committed')
            while True:
                time.sleep(1)
        daemon.kill_switch.commit_if_active = pause_after_commit
    daemon.start()
    signal.put('started')
    while True:
        time.sleep(1)


def test_committed_reminder_survives_process_death_without_duplicate(tmp_path, monkeypatch):
    from alphonse.agent_v2 import daemon as module
    socket_dir = Path(tempfile.mkdtemp(prefix='reminder-crash-', dir='/private/tmp'))
    monkeypatch.setattr(module, 'default_socket_path', lambda: socket_dir / 'daemon.sock')
    due = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(microsecond=0).isoformat()
    goal = f'Please remind me to call the plumber at {due}'
    ctx = multiprocessing.get_context('spawn')
    signal = ctx.Queue()
    process = ctx.Process(target=_child_after_commit, args=(str(tmp_path), str(socket_dir), signal, due, goal))
    process.start()
    replacement = None
    try:
        assert signal.get(timeout=20) == 'started'
        client = V2DaemonClient(str(socket_dir / 'first.sock'))
        message_id = client.request('experimental_owner_ingress', prompt=goal)['message_id']
        assert signal.get(timeout=10) == 'committed'
        process.kill()
        process.join(timeout=10)
        assert not process.is_alive()

        calls = []
        class NeverProvider:
            supports_image_input = False
            def generate_json(self, request):
                calls.append(True)
                raise AssertionError('committed reminder request was replayed')
        replacement, users, conversation = _build(tmp_path, socket_dir, 'second', NeverProvider())
        owner = users.admin_user()
        reminders = OwnerDesktopReminders(str(tmp_path / 'fixture.sqlite'))
        before = reminders.list_open(owner.user_id)
        assert len(before) == 1 and before[0]['message'] == 'call the plumber'
        assert [(event.role, event.content) for event in conversation.list(owner_user_id=owner.user_id)] == [('user', goal)]
        replacement.start()
        resumed = V2DaemonClient(replacement.ipc.socket_path)
        assert resumed.request('experimental_owner_status', message_id=message_id)['status'] == 'ready'
        draft = resumed.request('experimental_owner_take_draft', message_id=message_id)
        assert draft['message'].startswith('Reminder saved: call the plumber')
        assert resumed.request('experimental_owner_ack_draft', message_id=message_id,
            acknowledgement_token=draft['acknowledgement_token'])['status'] == 'client_acknowledged'
        assert reminders.list_open(owner.user_id) == before
        assert [(event.role, event.content) for event in conversation.list(owner_user_id=owner.user_id)] == [
            ('user', goal), ('assistant', draft['message'])]
        assert calls == []
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        if replacement is not None:
            replacement.stop()
        signal.close()
        shutil.rmtree(socket_dir)


def test_uncommitted_reminder_rolls_back_after_process_death(tmp_path, monkeypatch):
    from alphonse.agent_v2 import daemon as module
    socket_dir = Path(tempfile.mkdtemp(prefix='reminder-rollback-', dir='/private/tmp'))
    monkeypatch.setattr(module, 'default_socket_path', lambda: socket_dir / 'daemon.sock')
    due = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(microsecond=0).isoformat()
    goal = f'Please remind me to call the plumber at {due}'
    ctx = multiprocessing.get_context('spawn')
    signal = ctx.Queue()
    process = ctx.Process(target=_child_after_commit,
                          args=(str(tmp_path), str(socket_dir), signal, due, goal, True))
    process.start()
    replacement = None
    try:
        assert signal.get(timeout=20) == 'started'
        client = V2DaemonClient(str(socket_dir / 'first.sock'))
        message_id = client.request('experimental_owner_ingress', prompt=goal)['message_id']
        assert signal.get(timeout=10) == 'inserted_uncommitted'
        process.kill()
        process.join(timeout=10)
        assert not process.is_alive()
        db = tmp_path / 'fixture.sqlite'
        with sqlite3.connect(db) as conn:
            table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='owner_desktop_reminders'").fetchone()
            assert table is None or conn.execute('SELECT COUNT(*) FROM owner_desktop_reminders').fetchone()[0] == 0
            saved = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                            (message_id,)).fetchone()[0])['metadata']
            assert 'owner_read_pending_decision' in saved
            assert 'owner_read_ready_handoff' not in saved
            assert 'owner_read_selected_draft' not in saved
            before = conn.execute('SELECT status,claim_generation FROM v2_inbound_messages WHERE message_id=?',
                                  (message_id,)).fetchone()
        assert before[0] == 'processing' and before[1]

        calls = []
        class NeverProvider:
            supports_image_input = False
            def generate_json(self, request):
                calls.append(True)
                raise AssertionError('uncertain reminder request was replayed')
        replacement, users, conversation = _build(tmp_path, socket_dir, 'second', NeverProvider())
        owner = users.admin_user()
        assert [(event.role, event.content) for event in conversation.list(owner_user_id=owner.user_id)] == [('user', goal)]
        replacement.start()
        resumed = V2DaemonClient(replacement.ipc.socket_path)
        assert resumed.request('experimental_owner_status', message_id=message_id)['status'] == 'unavailable'
        with pytest.raises(RuntimeError, match='owner_desktop_pickup_unavailable'):
            resumed.request('experimental_owner_take_draft', message_id=message_id)
        time.sleep(.2)
        assert OwnerDesktopReminders(str(db)).list_open(owner.user_id) == []
        with sqlite3.connect(db) as conn:
            after = conn.execute('SELECT status,claim_generation FROM v2_inbound_messages WHERE message_id=?',
                                 (message_id,)).fetchone()
        assert after == before
        assert calls == []
        assert [(event.role, event.content) for event in conversation.list(owner_user_id=owner.user_id)] == [('user', goal)]
    finally:
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        if replacement is not None:
            replacement.stop()
        signal.close()
        shutil.rmtree(socket_dir)
