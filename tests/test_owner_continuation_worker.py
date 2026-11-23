"""A real owner worker pauses, survives restart, and resumes its saved task."""
import json
import os
import time
import tempfile
import threading
from pathlib import Path
import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


@pytest.mark.parametrize('use_bash,crash_after_activation,fail_reclaim,with_progress,with_steering', [(False, False, False, False, False), (True, False, False, False, False), (True, True, False, False, False), (False, False, True, False, False), (True, False, False, True, False), (False, False, False, False, True)])
def test_owner_conversation_checkpoint_after_restart(tmp_path, monkeypatch, use_bash, crash_after_activation, fail_reclaim, with_progress, with_steering):
    output = tmp_path / 'result.txt'
    steering_ready = threading.Event()
    steering_input = {}
    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            if with_steering and self.calls == 1:
                assert steering_ready.wait(5)
                from alphonse.agent_v2.owner_steering import submit
                assert submit(steering_input['queue'], owner=steering_input['owner'], peer_uid=os.getuid(),
                    daemon_id='before-restart', message_id=steering_input['message'],
                    request_id='a' * 32, content='Use the local clock.')['status'] == 'accepted'
                return InferenceResult(json_value={'kind': 'answer', 'tool_id': None, 'arguments': None,
                    'text': 'Stale answer.'}, raw_response='{}', usage={'input_tokens': 100, 'output_tokens': 20})
            if with_steering:
                assert 'Use the local clock.' in request.prompt
            if use_bash and self.calls in (1, 2):
                command = 'printf x >> result.txt' if self.calls == 1 else 'cat result.txt'
                decision = {'kind': 'tool', 'tool_id': 'native.bash',
                            'arguments': {'command': command, 'cwd': str(tmp_path)}, 'text': None}
            elif not use_bash and self.calls == (2 if with_steering else 1):
                decision = {'kind': 'tool', 'tool_id': 'native.owner_clock', 'arguments': {}, 'text': None}
            else:
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'The observed result is x.' if use_bash else 'I observed the clock.'}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    users._set_setting('owner_continuation_checkpoint', json.dumps({'metric': 'actions', 'interval': 1}))
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    provider = Provider()
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=SQLiteConversationStore(path), outbox=SQLiteOutboundStore(path),
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    if with_steering:
        from types import SimpleNamespace
        runtime.core.system_one = SimpleNamespace(review_freeform_effect_claim=lambda **_: True)
    judgments = []
    if with_progress:
        from types import SimpleNamespace
        from alphonse.agent_v2.core.core import CoreLoopContext
        class SyntheticJev:
            settings = SimpleNamespace(no_threshold=.3, yes_threshold=.7)
            def judge_observed_progress(self, *, state):
                judgments.append(state)
                return .9 if len(judgments) == 1 else .1
        runtime.core.system_one = SyntheticJev()
        # Explicit authorization for the disposable test provider only.
        monkeypatch.setattr(CoreLoopContext, 'owner_progress_disclosure_check',
                            lambda self, actor, provider, refs: True, raising=False)
    if use_bash:
        runtime.processor.owner_read_candidate.enable_owner_host_bash(trusted_local_development=True)

    def daemon(name):
        instance = V2Daemon(runtime, daemon_id=name, experimental_owner_only=True)
        instance._continuation_experimental_enabled = True  # isolated fixture only
        monkeypatch.setattr(instance, '_acquire_single_instance_lock', lambda: None)
        monkeypatch.setattr(instance, '_release_single_instance_lock', lambda: None)
        monkeypatch.setattr(instance.ipc, 'start', lambda: None)
        monkeypatch.setattr(instance.ipc, 'stop', lambda: None)
        instance.start()
        return instance

    first = daemon('before-restart')
    message = queue.enqueue_trusted_local_owner(prompt='Report the current clock.', peer_uid=os.getuid())
    steering_input.update(queue=queue, owner=owner.user_id, message=message)
    steering_ready.set()
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and first.owner_message_status(message) != 'continuation_paused':
            time.sleep(.02)
        assert first.owner_message_status(message) == 'continuation_paused'
        assert provider.calls == (2 if with_steering else 1)
        if use_bash:
            assert output.read_text() == 'x'
    finally:
        first.stop()
    second = daemon('after-restart')
    if fail_reclaim:
        from alphonse.agent_v2.core.intelligence.simple_loop import owner_continuation_reclaim as reclaim
        def fail(*_args, **_kwargs):
            raise ValueError('synthetic blocked recovery')
        monkeypatch.setattr(reclaim, 'reclaim_approved_pause', fail)
    if crash_after_activation:
        from alphonse.agent_v2.core.intelligence.simple_loop import owner_continuation_activation as activation
        original_activation = activation.activate_held_same_claim
        def crash_after_commit(*args, **kwargs):
            original_activation(*args, **kwargs)
            raise SystemExit('simulated crash after activation')
        monkeypatch.setattr(activation, 'activate_held_same_claim', crash_after_commit)
    socket_dir = tempfile.TemporaryDirectory(prefix='alp-', dir='/tmp')
    socket_path = Path(socket_dir.name) / 's.sock'
    server = V2DaemonServer(second, socket_path, experimental_owner_ingress=True)
    try:
        server.start()
        client = V2DaemonClient(socket_path)
        items = client.request('experimental_owner_continuation_list')['items']
        assert len(items) == 1 and items[0]['message_id'] == message
        if with_progress:
            assert 'preliminary review found evidence of progress' in items[0]['question']
        assert client.request('experimental_owner_continuation_choose', message_id=message,
            checkpoint_version=items[0]['checkpoint_version'], approve=True)['status'] == 'accepted'
        if fail_reclaim:
            other = queue.enqueue_trusted_local_owner(prompt='Another request', peer_uid=os.getuid())
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and second.owner_message_status(other) != 'ready':
                time.sleep(.02)
            assert second.owner_message_status(other) == 'ready'
            assert second.owner_message_status(message) == 'failed'
            return
        if crash_after_activation:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and second._processor_thread.is_alive():
                time.sleep(.02)
            assert not second._processor_thread.is_alive()
            assert output.read_text() == 'x'
            server.stop()
            second.stop()
            monkeypatch.setattr(activation, 'activate_held_same_claim', original_activation)
            from datetime import datetime, timedelta, timezone
            with queue._connect() as conn:
                conn.execute('UPDATE v2_inbound_messages SET lease_expires_at=? WHERE message_id=?',
                    ((datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat(), message))
            second = daemon('after-activation-crash')
            server = V2DaemonServer(second, socket_path, experimental_owner_ingress=True)
            server.start()
            client = V2DaemonClient(socket_path)
        if use_bash:
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline and not client.request('experimental_owner_continuation_list')['items']:
                time.sleep(.02)
            item = client.request('experimental_owner_continuation_list')['items'][0]
            if with_progress:
                assert 'did not find evidence of progress' in item['question']
            assert client.request('experimental_owner_continuation_choose', message_id=message,
                checkpoint_version=item['checkpoint_version'], approve=True)['status'] == 'accepted'
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and second.owner_message_status(message) not in {'ready', 'failed'}:
            time.sleep(.02)
        assert second.owner_message_status(message) == 'ready'
        assert provider.calls == (3 if use_bash or with_steering else 2)
        draft = second.take_owner_desktop_draft(message)
        assert ('The observed result is x.' if use_bash else 'I observed the clock.') in draft['message']
        if use_bash:
            assert output.read_text() == 'x'
        if with_progress:
            assert len(judgments) == 2
            assert len(judgments[0]['observations']) == len(judgments[1]['observations']) == 1
            assert 'printf x' in judgments[0]['observations'][0]['arguments']['command']
            assert 'cat result.txt' in judgments[1]['observations'][0]['arguments']['command']
    finally:
        server.stop()
        socket_dir.cleanup()
        second.stop()
