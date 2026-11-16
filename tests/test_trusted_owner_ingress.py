"""Disposable local socket proof; experimental ingress is never enabled by runtime."""
import os
import hashlib
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.users import V2UserStore


def test_peer_authenticated_owner_ingress_and_rejection(tmp_path):
    database = tmp_path / 'local.db'
    users = V2UserStore(database)
    owner = users.onboard(display_name='Fixture owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(database, experimental_claim_fencing=True)
    SQLiteConversationStore(database)
    daemon = SimpleNamespace(runtime=SimpleNamespace(queue=queue))
    with tempfile.TemporaryDirectory(dir='/tmp') as directory:
        path = Path(directory) / 'socket'
        server = V2DaemonServer(daemon, path)
        server.start()
        try:
            client = V2DaemonClient(path)
            with pytest.raises(RuntimeError, match='owner_ingress_unavailable'):
                client.request('experimental_owner_ingress', prompt='fixture')
        finally:
            server.stop()
        server = V2DaemonServer(daemon, path, experimental_owner_ingress=True)
        server.start()
        try:
            client = V2DaemonClient(path)
            for extra in ({'user': 'other'}, {'metadata': {}}, {'channel': 'desktop'},
                          {'intelligence_engine': 'hierarchical_v3'}, {'message_id': 'chosen'}):
                with pytest.raises(RuntimeError, match='owner_ingress_payload_denied'):
                    client.request('experimental_owner_ingress', prompt='fixture', **extra)
            assert queue.size() == 0
            result = client.request('experimental_owner_ingress', prompt=' fixture prompt ')
            queued = queue.peek()
            assert queued.message_id == result['message_id']
            assert queued.message.user == owner.user_id and queued.message.prompt == 'fixture prompt'
            assert queued.message.metadata == {'intelligence_engine': 'single_loop_experimental',
                'intelligence_schema_version': 1, 'routing_disposition': 'pdca_task'}
            with queue._connect() as conn:
                row = conn.execute('SELECT * FROM v2_trusted_owner_ingress WHERE message_id=?',
                                   (queued.message_id,)).fetchone()
                assert row['owner_id'] == owner.user_id and row['peer_uid'] == os.getuid()
                stored = conn.execute('SELECT * FROM v2_inbound_messages WHERE message_id=?', (queued.message_id,)).fetchone()
                payload = [stored[key] for key in ('message_id', 'timestamp', 'prompt', 'user_id',
                    'project_id', 'tag', 'correlation_id', 'memory_session_id', 'metadata_json')]
                digest = hashlib.sha256(json.dumps(['owner-history-v1', payload, row['history_event_id']],
                                                   ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
                assert row['payload_sha256'] == digest
                assert result['accepted_at'] == stored['timestamp']
                saved = conn.execute('SELECT * FROM v2_conversation_events WHERE event_id=?',
                                     (row['history_event_id'],)).fetchone()
                assert (saved['owner_user_id'], saved['role'], saved['content'], saved['source_message_id'], saved['created_at']) == (
                    owner.user_id, 'user', 'fixture prompt', f'inbound:{queued.message_id}', stored['timestamp'])
            users.update_user(owner.user_id, is_active=False)
            with pytest.raises(RuntimeError, match='owner_ingress_admin_unavailable'):
                client.request('experimental_owner_ingress', prompt='another')
            assert queue.size() == 1
        finally:
            server.stop()


def test_owner_ingress_transaction_rolls_back_queue_on_origin_failure(tmp_path):
    database = tmp_path / 'local.db'
    users = V2UserStore(database)
    users.onboard(display_name='Fixture owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(database, experimental_claim_fencing=True)
    SQLiteConversationStore(database)
    with queue._connect() as conn:
        conn.execute('CREATE TABLE v2_trusted_owner_ingress (message_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, peer_uid INTEGER NOT NULL, payload_sha256 TEXT NOT NULL, created_at TEXT NOT NULL) STRICT')
        conn.execute("CREATE TRIGGER reject_origin BEFORE INSERT ON v2_trusted_owner_ingress BEGIN SELECT RAISE(ABORT, 'fixture failure'); END")
    with pytest.raises(Exception, match='fixture failure'):
        queue.enqueue_trusted_local_owner(prompt='fixture', peer_uid=os.getuid())
    assert queue.size() == 0
    with pytest.raises(PermissionError, match='owner_ingress_peer_denied'):
        queue.enqueue_trusted_local_owner(prompt='fixture', peer_uid=os.getuid() + 1)
    assert queue.size() == 0


def test_owner_ingress_requires_history_and_rolls_back_on_history_failure(tmp_path):
    database = tmp_path / 'local.db'
    users = V2UserStore(database)
    users.onboard(display_name='Fixture owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(database, experimental_claim_fencing=True)
    with pytest.raises(Exception):
        queue.enqueue_trusted_local_owner(prompt='fixture', peer_uid=os.getuid())
    assert queue.size() == 0
    SQLiteConversationStore(database)
    with queue._connect() as conn:
        conn.execute("CREATE TRIGGER reject_history BEFORE INSERT ON v2_conversation_events BEGIN SELECT RAISE(ABORT, 'history failure'); END")
    with pytest.raises(Exception, match='history failure'):
        queue.enqueue_trusted_local_owner(prompt='fixture', peer_uid=os.getuid())
    with queue._connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM v2_inbound_messages').fetchone()[0] == 0
        assert conn.execute('SELECT COUNT(*) FROM v2_conversation_events').fetchone()[0] == 0
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_trusted_owner_ingress'").fetchone()


@pytest.mark.parametrize('tamper', ['delete_event', 'event_id', 'content', 'speaker', 'observed', 'schema', 'downgrade'])
def test_owner_claim_rejects_missing_or_tampered_accepted_history(tmp_path, tamper):
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    database = tmp_path / 'local.db'
    users = V2UserStore(database)
    owner = users.onboard(display_name='Fixture owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(database, experimental_claim_fencing=True)
    SQLiteConversationStore(database)
    message_id = queue.enqueue_trusted_local_owner(prompt='fixture', peer_uid=os.getuid())
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    assert queue.verify_trusted_local_owner_claim(acquired.claim_receipt, task)
    with queue._connect() as conn:
        if tamper == 'delete_event':
            conn.execute('DELETE FROM v2_conversation_events WHERE source_message_id=?', (f'inbound:{message_id}',))
        elif tamper == 'event_id':
            conn.execute("UPDATE v2_trusted_owner_ingress SET history_event_id='replacement' WHERE message_id=?", (message_id,))
        elif tamper == 'downgrade':
            conn.execute("UPDATE v2_trusted_owner_ingress SET history_version=0,history_event_id='' WHERE message_id=?", (message_id,))
        else:
            column, value = {'content': ('content', 'altered'), 'speaker': ('speaker_id', 'someone-else'),
                             'observed': ('observed_at', ''), 'schema': ('schema_version', 2)}[tamper]
            conn.execute(f'UPDATE v2_conversation_events SET {column}=? WHERE source_message_id=?', (value, f'inbound:{message_id}'))
    assert not queue.verify_trusted_local_owner_claim(acquired.claim_receipt, task)
    with pytest.raises(ValueError, match='owner_abandon_unavailable'):
        queue.abandon_trusted_owner_message(message_id)


def test_owner_origin_verified_by_real_candidate_against_claim_and_payload(tmp_path):
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor
    from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import SQLiteTaskBudgetStore
    from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
    from alphonse.agent_v2.core.inference.router import InferenceRouter
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    database = tmp_path / 'local.db'
    users = V2UserStore(database)
    owner = users.onboard(display_name='Fixture owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(database, experimental_claim_fencing=True)
    SQLiteConversationStore(database)
    message_id = queue.enqueue_trusted_local_owner(prompt='fixture', peer_uid=os.getuid())
    acquired = queue.claim_next()
    claim = acquired.claim_receipt
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.db')),
        authorize_claim=lambda task, context: queue.verify_trusted_local_owner_claim(claim, task),
        save_checkpoint=lambda *_: None)
    context = CoreLoopContext(messages=queue)
    # Passing auth reaches the next explicit host requirement; no inference or effect occurs.
    assert processor.process(task, context).error == 'owner_read_host_unavailable'
    assert not queue.verify_trusted_local_owner_claim(replace(claim, user_id='other'), task)
    assert not queue.verify_trusted_local_owner_claim(replace(claim, claim_generation='0' * 32), task)
    assert not queue.verify_trusted_local_owner_claim(claim, replace(task, user='other'))
    assert not queue.verify_trusted_local_owner_claim(claim, replace(task, goal='tampered'))
    assert not queue.verify_trusted_local_owner_claim(claim, replace(task, metadata={**task.metadata, 'intelligence_schema_version': True}))
    assert not queue.verify_trusted_local_owner_claim(claim, task, clock=lambda: datetime.now(timezone.utc) + timedelta(days=1))
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET prompt='tampered' WHERE message_id=?", (message_id,))
    assert processor.process(task, context).error == 'owner_read_authority_unavailable'
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET prompt='fixture' WHERE message_id=?", (message_id,))
    assert queue.verify_trusted_local_owner_claim(claim, task)
    with queue._connect() as conn:
        conn.execute("UPDATE v2_inbound_messages SET metadata_json='{}' WHERE message_id=?", (message_id,))
    assert not queue.verify_trusted_local_owner_claim(claim, task)
    with queue._connect() as conn:
        conn.execute('UPDATE v2_inbound_messages SET metadata_json=? WHERE message_id=?',
                     (json.dumps(task.metadata, sort_keys=True), message_id))
    users.update_user(owner.user_id, is_active=False)
    assert not queue.verify_trusted_local_owner_claim(claim, task)
    users.update_user(owner.user_id, is_active=True)
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            result = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Fixture draft.'}
            return InferenceResult(json_value=result, raw_response=json.dumps(result),
                                   usage={'input_tokens': 20, 'output_tokens': 5})
    context.tools = registry
    context.conversation_store = SQLiteConversationStore(tmp_path / 'conversation.db')
    context.inference = InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline'))
    outcome = processor.process(task, context)
    assert outcome.snapshot.metadata.get('draft') == 'Fixture draft.', outcome.error
    assert outcome.snapshot.metadata['delivery'] == 'not_integrated'
    with queue._connect() as conn:
        conn.execute('DELETE FROM v2_trusted_owner_ingress WHERE message_id=?', (message_id,))
    assert not queue.verify_trusted_local_owner_claim(claim, task)


def test_only_active_canonical_owner_or_admin_can_create_owner_ingress(tmp_path):
    database = tmp_path / 'canonical.sqlite'
    users = V2UserStore(database)
    owner = users.onboard(display_name='Fixture owner', users_root=tmp_path / 'users')
    users.update_user(owner.user_id, role='owner')
    other = users.create_user(display_name='Other owner', role='owner')
    queue = SQLiteMessageQueue(database, experimental_claim_fencing=True)
    SQLiteConversationStore(database)
    message_id = queue.enqueue_trusted_local_owner(prompt='fixture', peer_uid=os.getuid())
    acquired = queue.claim_next()
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    assert task.user == owner.user_id
    assert queue.verify_trusted_local_owner_claim(acquired.claim_receipt, task)
    # A different active user with the owner role is not the canonical owner.
    assert not queue.verify_trusted_local_owner_claim(acquired.claim_receipt, replace(task, user=other.user_id))
    users.update_user(owner.user_id, is_active=False)
    assert not queue.verify_trusted_local_owner_claim(acquired.claim_receipt, task)
    with pytest.raises(PermissionError, match='owner_ingress_admin_unavailable'):
        queue.enqueue_trusted_local_owner(prompt='second', peer_uid=os.getuid())
