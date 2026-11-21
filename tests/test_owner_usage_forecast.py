"""Real owner-loop accounting with a provider reporting more than its forecast."""
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


def test_owner_answer_with_reported_usage_above_forecast_is_charged_and_delivered(tmp_path, monkeypatch):
    class Provider:
        supports_image_input = False
        def generate_json(self, _request):
            value = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'A synthetic answer.'}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 85781, 'output_tokens': 383})

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.set_workspace_root(tmp_path / 'workspace')
    owner = users.onboard(display_name='Synthetic owner')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    runtime = build_runtime_host(user_store=users, messages=queue,
        question_store=SQLiteQuestionStore(path), conversation_store=conversation,
        outbox=SQLiteOutboundStore(path),
        inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='usage-fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        message = queue.enqueue_trusted_local_owner(prompt='Synthetic question', peer_uid=os.getuid())
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline and daemon.owner_message_status(message) in {'pending', 'processing'}:
            time.sleep(.01)
        assert daemon.owner_message_status(message) == 'ready'
        draft = daemon.take_owner_desktop_draft(message)
        assert draft['message'] == 'A synthetic answer.'
        checkpoint, _ = daemon._owner_checkpoints[message]
        saved, _ = checkpoint.load(expected_version=checkpoint.version)
        assert saved.budget.usage()['charged']['input_tokens'] == 85781
        assert saved.budget.usage()['unknown_usage'] == []
        assert saved.budget.within_work_limits()
        daemon.acknowledge_owner_desktop_draft(message, draft['acknowledgement_token'])
    finally:
        daemon.stop()
