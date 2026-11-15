"""Owner read pagination with disposable sources and scripted inference."""
import json
import os
import tempfile
import time
from copy import deepcopy

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import OwnerReadConversationProcessor
from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import SQLiteTaskBudgetStore
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import (
    build_read_memory_source_tool_definition, build_search_memory_tool_definition,
    complete_conversation_read, conversation_read_sequences, execute_read_memory_source, READ_MEMORY_SOURCE_TOOL_ID,
)
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host


def _run(tmp_path, provider_factory, *, text=None):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Fixture', users_root=tmp_path / 'users')
    conversation = SQLiteConversationStore(tmp_path / 'conversation.sqlite')
    content = text or ('Willow picnic: Thursday. ' + 'ordinary filler ' * 800 + 'Update: Saturday.')
    event = conversation.record(owner_user_id=owner.user_id, role='user', content=content,
                                source='test', source_message_id='source')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    provider = provider_factory('conversation:' + event.event_id, conversation, event, content)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        conversation_store=conversation, cancellation_checker=lambda: False)
    task = TaskState(task_id='owner-long', message_id='owner-long', user=owner.user_id,
        goal='When is the Willow picnic?', intelligence_engine='single_loop_experimental', intelligence_schema_version=1)
    processor = OwnerReadConversationProcessor(users=users, budgets=SQLiteTaskBudgetStore(str(tmp_path / 'budget.sqlite')),
        authorize_claim=lambda *_: True, save_checkpoint=lambda *_: None, now_ms=lambda: 100)
    return processor.process(task, context), provider


class Pages:
    supports_image_input = False
    def __init__(self, source_id, *_):
        self.source_id = source_id
        self.requests = []
    def generate_json(self, request):
        self.requests.append(request)
        step = len(self.requests)
        if step == 1:
            value = {'kind':'tool', 'tool_id':'native.search_memory',
                     'arguments':{'query':'Willow picnic','source_types':['conversation']}, 'text':None}
        elif step == 2:
            assert 'Thursday' not in request.prompt
            value = {'kind':'tool', 'tool_id':'native.read_memory_source',
                     'arguments':{'source_id':self.source_id, 'neighbors':0, 'max_chars':12000}, 'text':None}
        elif step == 3:
            assert 'Thursday' not in request.prompt and 'Saturday' not in request.prompt
            assert '12000' in request.prompt
            value = {'kind':'tool', 'tool_id':'native.read_memory_source',
                     'arguments':{'source_id':self.source_id, 'neighbors':0, 'max_chars':12000, 'offset':12000}, 'text':None}
        else:
            assert 'Thursday' in request.prompt and 'Saturday' in request.prompt
            value = {'kind':'answer', 'tool_id':None, 'arguments':None, 'text':'Saturday.'}
        return InferenceResult(json_value=value, raw_response=json.dumps(value),
                               usage={'input_tokens':100, 'output_tokens':20})


def test_long_source_is_assembled_before_exposure_and_answer(tmp_path):
    result, provider = _run(tmp_path, Pages)
    assert result.status == ProcessingStatus.WAITING, result.error
    assert result.snapshot.metadata['draft'] == 'Saturday.'
    assert len(provider.requests) == 4


def test_gap_is_not_authorized_or_treated_as_complete(tmp_path):
    class Gap(Pages):
        def generate_json(self, request):
            if len(self.requests) == 2:
                self.requests.append(request)
                value = {'kind':'tool', 'tool_id':'native.read_memory_source',
                         'arguments':{'source_id':self.source_id, 'neighbors':0, 'max_chars':12000, 'offset':12001}, 'text':None}
                return InferenceResult(json_value=value, raw_response=json.dumps(value), usage={'input_tokens':100,'output_tokens':20})
            return super().generate_json(request)
    result, provider = _run(tmp_path, Gap)
    assert result.status == ProcessingStatus.FAILED
    assert len(provider.requests) == 3


def test_oversized_source_is_not_exposed_as_complete(tmp_path):
    result, provider = _run(tmp_path, Pages, text='Willow picnic ' + 'private filler ' * 3000)
    assert result.status == ProcessingStatus.FAILED
    assert len(provider.requests) == 2


def test_source_change_between_pages_stops_continuation(tmp_path):
    class Changed(Pages):
        def __init__(self, source_id, store, event, content):
            super().__init__(source_id)
            self.store, self.event, self.content = store, event, content
        def generate_json(self, request):
            if len(self.requests) == 2:
                # A same-length change before the next call must be caught by
                # fresh page validation, not merely by the reported length.
                with self.store._connect() as conn:
                    conn.execute('UPDATE v2_conversation_events SET content=? WHERE event_id=?',
                                 (self.content.replace('Thursday', 'Tuesday '), self.event.event_id))
            return super().generate_json(request)
    result, provider = _run(tmp_path, Changed)
    assert result.status == ProcessingStatus.FAILED
    assert len(provider.requests) == 3


def test_real_owner_worker_delivers_assembled_source_under_claim(tmp_path, monkeypatch):
    db = tmp_path / 'runtime.sqlite'
    users = V2UserStore(db)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(db)
    content = 'Willow picnic: Thursday. ' + 'ordinary filler ' * 800 + 'Update: Saturday.'
    event = conversation.record(owner_user_id=owner.user_id, role='user', content=content,
                                source='test', source_message_id='source')
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    provider = Pages('conversation:' + event.event_id)
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
        conversation_store=conversation, outbox=SQLiteOutboundStore(db), tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test','fixture','offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='owner-long-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    with tempfile.TemporaryDirectory(prefix='owner-long-', dir='/private/tmp') as socket_dir:
        daemon.ipc = V2DaemonServer(daemon, os.path.join(socket_dir, 'p.sock'))
        daemon.start()
        try:
            client = V2DaemonClient(daemon.ipc.socket_path)
            message_id = client.request('experimental_owner_ingress', prompt='When is the Willow picnic?')['message_id']
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                status = client.request('experimental_owner_status', message_id=message_id)['status']
                if status != 'pending':
                    break
                time.sleep(.02)
            assert status == 'ready'
            result = client.request('experimental_owner_take_draft', message_id=message_id)
            assert result['message'] == 'Saturday.'
            assert client.request('experimental_owner_ack_draft', message_id=message_id,
                acknowledgement_token=result['acknowledgement_token'])['status'] == 'client_acknowledged'
            assert len(conversation.list(owner_user_id=owner.user_id)) == 3
            with queue._connect() as conn:
                raw = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                   (message_id,)).fetchone()['task_state_json']
            assert 'ordinary filler' not in raw
            assert len(provider.requests) == 4
        finally:
            daemon.stop()


def test_chunk_validation_rejects_malformed_or_mixed_pages(tmp_path):
    users = V2UserStore(':memory:')
    owner = users.onboard(display_name='Fixture', users_root=tmp_path / 'users')
    store = SQLiteConversationStore(tmp_path / 'conversation.sqlite')
    event = store.record(owner_user_id=owner.user_id, role='user', content='A' * 13000,
                         source='test', source_message_id='long')
    task = TaskState(task_id='owner', user=owner.user_id, goal='fixture')
    execution = CoreLoopContext(messages=InMemoryMessageQueue(), conversation_store=store).tool_execution_context(task)
    source = 'conversation:' + event.event_id
    first = {'source_id':source, 'neighbors':0, 'max_chars':12000}
    second = {'source_id':source, 'neighbors':0, 'max_chars':12000, 'offset':12000}
    page1 = execute_read_memory_source(first, context=execution)
    page2 = execute_read_memory_source(second, context=execution)
    assert complete_conversation_read(owner.user_id, READ_MEMORY_SOURCE_TOOL_ID, first, page1, allow_chunks=True)
    assert complete_conversation_read(owner.user_id, READ_MEMORY_SOURCE_TOOL_ID, second, page2, allow_chunks=True)
    with pytest.raises(ValueError, match='owner_read_sequence_invalid'):
        conversation_read_sequences(owner.user_id, {'second':(READ_MEMORY_SOURCE_TOOL_ID, second, page2)})
    for key in ('start_char', 'end_char', 'total_chars'):
        damaged = deepcopy(page1)
        damaged['items'][0][key] = True
        assert not complete_conversation_read(owner.user_id, READ_MEMORY_SOURCE_TOOL_ID, first, damaged, allow_chunks=True)
    assert not complete_conversation_read(owner.user_id, READ_MEMORY_SOURCE_TOOL_ID,
                                          {**first, 'offset': True}, page1, allow_chunks=True)
    assert not complete_conversation_read('wrong-owner', READ_MEMORY_SOURCE_TOOL_ID, first, page1, allow_chunks=True)
    for mutate in ('overlap', 'total', 'source'):
        args, damaged = deepcopy(second), deepcopy(page2)
        if mutate == 'overlap':
            args['offset'] = 11999
            damaged['items'][0]['start_char'] = 11999
            damaged['items'][0]['content'] = 'A' + damaged['items'][0]['content']
        elif mutate == 'total':
            damaged['items'][0]['total_chars'] += 1
            damaged['next_offset'] = damaged['items'][0]['end_char']
        else:
            damaged['items'][0]['source']['provenance'] = 'different provenance'
        with pytest.raises(ValueError, match='owner_read_sequence_invalid'):
            conversation_read_sequences(owner.user_id, {'first':(READ_MEMORY_SOURCE_TOOL_ID, first, page1),
                                                       'second':(READ_MEMORY_SOURCE_TOOL_ID, args, damaged)})
    switched_args, switched_page = deepcopy(second), deepcopy(page2)
    switched_args['source_id'] = 'conversation:other'
    switched_page['items'][0]['source_id'] = 'conversation:other'
    switched_page['items'][0]['source']['source_id'] = 'conversation:other'
    with pytest.raises(ValueError, match='owner_read_sequence_invalid'):
        conversation_read_sequences(owner.user_id, {'first':(READ_MEMORY_SOURCE_TOOL_ID, first, page1),
                                                   'second':(READ_MEMORY_SOURCE_TOOL_ID, switched_args, switched_page)})
