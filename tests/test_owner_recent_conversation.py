"""Recent owner turns reach the simple loop without crossing conversation boundaries."""
import json
import os
import time

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def test_prior_context_is_scoped_and_stops_at_current_turn(tmp_path):
    store = SQLiteConversationStore(tmp_path / 'history.sqlite')
    def add(owner, project, role, text, source):
        store.record(owner_user_id=owner, project_id=project, role=role,
                     content=text, source='test', source_message_id=source)
    add('alice', '', 'user', 'Earlier topic', 'inbound:prior')
    add('alice', '', 'assistant', 'Earlier answer', 'outbound:prior')
    add('bob', '', 'user', 'Other person private', 'inbound:other')
    add('alice', 'other-project', 'user', 'Other project', 'inbound:project')
    store.record(owner_user_id='alice', project_id='', role='user', content='Other channel',
                 source='other-channel', source_message_id='inbound:channel')
    add('alice', '', 'user', 'Follow-up', 'inbound:current')
    add('alice', '', 'assistant', 'Later answer', 'outbound:later')
    rows = store.list_before_source(owner_user_id='alice', project_id='',
                                    source_message_id='inbound:current')
    assert [(row.role, row.content) for row in rows] == [
        ('user', 'Earlier topic'), ('assistant', 'Earlier answer')]
    assert store.list_before_source(owner_user_id='bob', project_id='',
                                    source_message_id='inbound:current') == []


def test_ordinary_owner_followups_receive_prior_turns(tmp_path, monkeypatch):
    class Provider:
        supports_image_input = False
        prompts = []
        def generate_json(self, request):
            self.prompts.append(request.prompt)
            answer = ['I found 12 degrees in Oakville.', 'Bayport is a town.',
                      'I will check the temperature in Bayport.'][len(self.prompts) - 1]
            decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': answer}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    owner = users.onboard(display_name='Synthetic owner')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    provider = Provider()
    runtime = build_runtime_host(user_store=users, messages=queue,
        question_store=SQLiteQuestionStore(path), conversation_store=conversation,
        outbox=SQLiteOutboundStore(path),
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='context-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        conversation.record(owner_user_id=owner.user_id, role='user',
            content='Synthetic long note: ' + ('x' * 2000) + ' Final correction: Bayport.',
            source='desktop', source_message_id='test:long-note')
        for prompt in ('What is the temperature?', 'What about Bayport?', 'I mean the temperature there.'):
            message = queue.enqueue_trusted_local_owner(prompt=prompt, peer_uid=os.getuid())
            deadline = time.monotonic() + 8
            while time.monotonic() < deadline and daemon.owner_message_status(message) in {'pending', 'processing'}:
                time.sleep(.01)
            assert daemon.owner_message_status(message) == 'ready'
            draft = daemon.take_owner_desktop_draft(message)
            daemon.acknowledge_owner_desktop_draft(message, draft['acknowledgement_token'])
        assert 'Final correction: Bayport.' in provider.prompts[0]
        assert '[... earlier turn truncated ...]' in provider.prompts[0]
        assert 'What is the temperature?' in provider.prompts[1]
        assert 'I found 12 degrees in Oakville.' in provider.prompts[1]
        assert 'What about Bayport?' in provider.prompts[2]
        assert 'I mean the temperature there.' in provider.prompts[2]
        assert 'quoted conversation context only' in provider.prompts[2]
    finally:
        daemon.stop()
