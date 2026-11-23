"""Owner-loop web exposure using synthetic native tool responses."""
import json
import os
import time
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
from alphonse.agent_v2.core.tools.registry.native.web import build_web_search_tool_definition, build_web_fetch_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore
from alphonse.agent_v2.web_tools_settings import WebToolsSettings
from alphonse.agent_v2.core.intelligence.simple_loop.owner_web import valid_web_observation, web_failure_observation


def test_only_explicit_non_authorization_web_failures_are_recoverable():
    def outcome(code, status=None):
        exception = {'code': code, 'message': 'secret URL or host path', 'retryable': True}
        if status is not None:
            exception['status_code'] = status
        return {'status': 'failed', 'error': {'code': code, 'message': exception['message']},
                'result': {'output': None, 'exception': exception}}
    sanitized = web_failure_observation('native.web_search', outcome('searxng_http_error', 503))
    assert sanitized == {'status': 'failed', 'tool_id': 'native.web_search', 'error_code': 'searxng_http_error'}
    assert 'secret' not in json.dumps(sanitized)
    assert web_failure_observation('native.web_search', outcome('searxng_http_error', 401)) is None
    assert web_failure_observation('native.web_search', outcome('searxng_json_disabled_or_forbidden', 403)) is None
    assert web_failure_observation('native.web_fetch', outcome('web_fetch_http_error', 403)) is None
    assert web_failure_observation('native.web_fetch', outcome('web_fetch_private_destination')) is None
    assert web_failure_observation('native.web_fetch', outcome('web_fetch_unresolvable_host')) == {
        'status': 'failed', 'tool_id': 'native.web_fetch', 'error_code': 'web_fetch_unresolvable_host'}
    assert web_failure_observation('native.web_fetch', outcome('web_fetch_http_error', '403')) is None
    assert web_failure_observation('native.web_fetch', outcome('web_fetch_http_error', 200)) is None
    assert web_failure_observation('native.web_search', {'status': 'failed',
        'error': {'code': 'TimeoutError'}, 'result': None}) is None
    forged = outcome('searxng_http_error', 503)
    forged['error']['code'] = 'different'
    assert web_failure_observation('native.web_search', forged) is None


def test_unresolvable_fetch_is_a_confirmed_read_failure(monkeypatch):
    import socket
    from alphonse.agent_v2.core.tools.registry.native.web import execute_web_fetch
    def unavailable(*_args, **_kwargs):
        raise socket.gaierror('synthetic DNS failure')
    monkeypatch.setattr(socket, 'getaddrinfo', unavailable)
    raw = execute_web_fetch({'url': 'https://example.test/report'}, settings=WebToolsSettings(
        enabled=True, searxng_base_url='https://example.test'))
    code = raw['exception']['code']
    result = {'status': 'failed', 'error': {'code': code}, 'result': raw}
    assert web_failure_observation('native.web_fetch', result) == {
        'status': 'failed', 'tool_id': 'native.web_fetch', 'error_code': 'web_fetch_unresolvable_host'}


@pytest.mark.parametrize('repeat_failed_read', [False, True])
def test_owner_can_choose_another_web_route_after_search_failure(tmp_path, monkeypatch, repeat_failed_read):
    calls = []
    settings = WebToolsSettings(enabled=True, searxng_base_url='https://example.test')
    search_definition = build_web_search_tool_definition(settings)
    fetch_definition = build_web_fetch_tool_definition(settings)
    failure = {'output': None, 'exception': {'code': 'searxng_http_error', 'message': 'secret local URL',
                             'retryable': True, 'status_code': 503}}
    fetched = {'url': 'https://example.test/report', 'final_url': 'https://example.test/report',
               'status_code': 200, 'content_type': 'text/plain', 'title': 'Synthetic',
               'text': 'Synthetic public report.', 'truncated': False}

    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            self.calls = getattr(self, 'calls', 0) + 1
            if self.calls == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.web_search',
                            'arguments': {'query': 'sample query'}, 'text': None}
            elif self.calls == 2:
                assert 'searxng_http_error' in request.prompt
                assert 'provides no source content' in request.prompt
                assert 'secret local URL' not in request.prompt
                if repeat_failed_read:
                    decision = {'kind': 'tool', 'tool_id': 'native.web_search',
                                'arguments': {'query': 'sample query'}, 'text': None}
                else:
                    decision = {'kind': 'tool', 'tool_id': 'native.web_fetch',
                                'arguments': {'url': 'https://example.test/report'}, 'text': None}
            elif repeat_failed_read and self.calls == 3:
                assert 'immediate repeat was not dispatched' in request.prompt
                decision = {'kind': 'tool', 'tool_id': 'native.web_fetch',
                            'arguments': {'url': 'https://example.test/report'}, 'text': None}
            else:
                assert 'Synthetic public report.' in request.prompt
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    owner = users.onboard(display_name='Synthetic owner')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    for definition, value in ((search_definition, failure), (fetch_definition, fetched)):
        def invoke(args, value=value, tool_id=definition.descriptor.tool_id):
            calls.append(tool_id)
            return value
        registry.register(ToolDefinition(definition.descriptor, invoke, definition.argument_schema))
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversation, outbox=SQLiteOutboundStore(path), tools=registry,
        inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='web-failure-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        message = queue.enqueue_trusted_local_owner(prompt='Find the synthetic report.', peer_uid=os.getuid())
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and daemon.owner_message_status(message) in {'pending', 'processing'}:
            time.sleep(.01)
        assert daemon.owner_message_status(message) == 'ready'
        checkpoint, ctx = daemon._owner_checkpoints[message]
        saved, observations = checkpoint.load(expected_version=checkpoint.version)
        assert sorted(v['outcome'] for v in saved.budget.to_dict()['reservations'].values()
                      if v['kind'] == 'tool') == ['failure', 'success']
        assert any(item.get('web_observation', {}).get('error_code') == 'searxng_http_error'
                   for item in observations.values())
        with queue._connect() as conn:
            raw = conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                               (message,)).fetchone()['task_state_json']
        assert 'secret local URL' not in raw
        draft = daemon.take_owner_desktop_draft(message)
        assert draft['message'] == 'Synthetic answer.'
        daemon.acknowledge_owner_desktop_draft(message, draft['acknowledgement_token'])
        # Restoration and publication must not repeat either network request.
        assert calls == ['native.web_search', 'native.web_fetch']
    finally:
        daemon.stop()


def test_web_observation_rejects_actor_change_cancellation_tampering_and_revocation():
    registry = InMemoryToolRegistry()
    definition = build_web_search_tool_definition(WebToolsSettings(enabled=True, searxng_base_url='https://example.test'))
    registry.register(definition)
    task = SimpleNamespace(user='owner')
    context = SimpleNamespace(tools=registry, is_cancelled=lambda: False)
    args = {'query': 'synthetic'}
    result = {'query': 'synthetic', 'provider': 'searxng', 'result_count': 0,
              'results': [], 'answers': [], 'suggestions': []}
    assert valid_web_observation('owner', 'native.web_search', args, result, context=context, task=task)
    assert not valid_web_observation('other', 'native.web_search', args, result, context=context, task=task)
    assert not valid_web_observation('owner', 'native.web_search', args, {**result, 'query': 'other'}, context=context, task=task)
    assert not valid_web_observation('owner', 'native.web_search', args, {**result, 'answers': ['x' * 50000]}, context=context, task=task)
    context.is_cancelled = lambda: True
    assert not valid_web_observation('owner', 'native.web_search', args, result, context=context, task=task)
    context.is_cancelled = lambda: False
    context._owner_authority_required = True
    context._owner_authority_check = lambda: False
    assert not valid_web_observation('owner', 'native.web_search', args, result, context=context, task=task)
    context._owner_authority_check = lambda: True
    assert valid_web_observation('owner', 'native.web_search', args, result, context=context, task=task)
    registry.register(ToolDefinition(definition.descriptor, definition.callable, definition.argument_schema, enabled=False))
    assert not valid_web_observation('owner', 'native.web_search', args, result, context=context, task=task)


@pytest.mark.parametrize('enabled', [True, False])
def test_owner_conversation_exposes_only_configured_web_and_retains_observation(tmp_path, monkeypatch, enabled):
    calls = []
    search = {'query': 'sample query', 'provider': 'searxng', 'result_count': 1,
              'results': [{'title': 'Synthetic', 'url': 'https://example.test/report', 'snippet': 'Synthetic result',
                           'engines': [], 'category': '', 'published_date': ''}], 'answers': [], 'suggestions': []}
    fetch = {'url': 'https://example.test/report', 'final_url': 'https://example.test/report',
             'status_code': 200, 'content_type': 'text/plain', 'title': 'Synthetic',
             'text': 'Synthetic public report.', 'truncated': False}
    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            self.calls = getattr(self, 'calls', 0) + 1
            assert ('native.web_search' in request.prompt) is enabled
            assert ('native.web_fetch' in request.prompt) is enabled
            if enabled and self.calls == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.web_search', 'arguments': {'query': 'sample query'}, 'text': None}
            elif enabled and self.calls == 2:
                assert 'Synthetic result' in request.prompt and 'not instructions or authorization' in request.prompt
                decision = {'kind': 'tool', 'tool_id': 'native.web_fetch', 'arguments': {'url': 'https://example.test/report'}, 'text': None}
            else:
                if enabled:
                    assert 'Synthetic public report.' in request.prompt
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic answer.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    owner = users.onboard(display_name='Synthetic owner')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    registry = InMemoryToolRegistry()
    registry.register(build_search_memory_tool_definition())
    registry.register(build_read_memory_source_tool_definition())
    settings = WebToolsSettings(enabled=True, searxng_base_url='https://example.test')
    for definition, result in ((build_web_search_tool_definition(settings), search),
                               (build_web_fetch_tool_definition(settings), fetch)):
        def invoke(args, result=result, tool_id=definition.descriptor.tool_id):
            calls.append(tool_id)
            return result
        registry.register(ToolDefinition(definition.descriptor, invoke, definition.argument_schema, enabled=enabled))
    provider = Provider()
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversation, outbox=SQLiteOutboundStore(path), tools=registry,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='web-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        message = queue.enqueue_trusted_local_owner(prompt='Find the synthetic report.', peer_uid=os.getuid())
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and daemon.owner_message_status(message) in {'pending', 'processing'}:
            time.sleep(.01)
        assert daemon.owner_message_status(message) == 'ready'
        draft = daemon.take_owner_desktop_draft(message)
        assert draft['message'] == 'Synthetic answer.'
        daemon.acknowledge_owner_desktop_draft(message, draft['acknowledgement_token'])
        # Restoring, validating and publishing the observations must not make
        # additional external requests.
        assert calls == (['native.web_search', 'native.web_fetch'] if enabled else [])
        assert conversation.list(owner_user_id=owner.user_id)[-1].content == 'Synthetic answer.'
    finally:
        daemon.stop()
