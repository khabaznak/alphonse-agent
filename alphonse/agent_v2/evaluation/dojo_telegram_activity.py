"""Disposable Telegram activity Dojo using the real owner loop and a fake provider.

The model is deliberately held while generating an answer so activity can be
observed before completion. The Telegram client is an in-process fake: these
results do not establish delivery through Telegram itself or real-model quality.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import threading
import time
from unittest.mock import patch

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore

CASE_ID = 'telegram-owner-activity-feedback'


def assess(events: list[dict], *, model_was_held: bool, completed: bool) -> dict:
    """Grade externally visible fake-provider calls, not internal intent."""
    if not isinstance(events, list) or any(not isinstance(event, dict) for event in events):
        raise ValueError('telegram_activity_events_invalid')
    replies = [index for index, event in enumerate(events) if event.get('kind') == 'reply']
    reply_index = replies[0] if replies else len(events)
    before = events[:reply_index]
    relevant = [event for event in before if event.get('chat_id') == '123']
    reaction = any(event.get('kind') == 'reaction' and event.get('message_id') == '41'
                   for event in relevant)
    typing = any(event.get('kind') == 'typing' for event in relevant)
    starts = [i for i, event in enumerate(events) if event.get('kind') == 'model_started']
    ends = [i for i, event in enumerate(events) if event.get('kind') == 'model_released']
    window = events[starts[0] + 1:ends[0]] if len(starts) == len(ends) == 1 and starts[0] < ends[0] else []
    while_working = any(event.get('chat_id') == '123' and
                        (event.get('kind') == 'typing' or
                         (event.get('kind') == 'reaction' and event.get('message_id') == '41'))
                        for event in window)
    return {'model_was_held': model_was_held, 'reply_completed': completed and len(replies) == 1,
            'reaction_before_reply': reaction, 'typing_before_reply': typing,
            'visible_activity_before_reply': reaction or typing, 'visible_activity_while_working': while_working,
            'result': ('pass' if model_was_held and completed and len(replies) == 1 and (reaction or typing) and while_working
                       else 'fail' if model_was_held and completed and len(replies) == 1
                       else 'inconclusive')}


def run(*, timeout: float = 12, hold_seconds: float = 6, feedback_failure: bool = False,
        activity_interval: float = 4.0, model_failure: bool = False, cancel_task: bool = False) -> dict:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout <= 60:
        raise ValueError('telegram_activity_timeout_invalid')
    if (isinstance(hold_seconds, bool) or not isinstance(hold_seconds, (int, float))
            or not 0 < hold_seconds < timeout):
        raise ValueError('telegram_activity_hold_invalid')
    entered, release = threading.Event(), threading.Event()
    events = []

    class Provider:
        supports_image_input = False
        def generate_json(self, request):
            events.append({'kind': 'model_started'})
            entered.set()
            if not release.wait(timeout):
                raise TimeoutError('synthetic held model timed out')
            if model_failure:
                raise RuntimeError('synthetic model failure')
            result = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Synthetic reply'}
            return InferenceResult(json_value=result, raw_response=json.dumps(result),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    class Client:
        def get_updates(self, *, offset, timeout):
            if offset is not None and offset > 41:
                return []
            return [{'update_id': 41, 'message': {'date': 1001, 'message_id': 41,
                     'text': 'Please check the synthetic report.', 'from': {'id': 123},
                     'chat': {'id': 123, 'type': 'private'}}}]
        def send_chat_action(self, *, chat_id, action):
            events.append({'kind': 'typing' if action == 'typing' else 'activity', 'chat_id': str(chat_id)})
        def set_message_reaction(self, *, chat_id, message_id, emoji):
            if feedback_failure:
                events.append({'kind': 'feedback_failed'})
                raise RuntimeError('synthetic feedback failure')
            events.append({'kind': 'reaction', 'chat_id': str(chat_id), 'message_id': str(message_id), 'emoji': emoji})
        def send_message(self, *, chat_id, text):
            events.append({'kind': 'reply', 'chat_id': str(chat_id), 'text': text})
            return 'synthetic-provider-ack'

    with tempfile.TemporaryDirectory(prefix='dojo-telegram-activity-') as temp:
        root = Path(temp)
        db = root / 'dojo.sqlite'
        users = V2UserStore(db)
        users.set_workspace_root(root / 'workspace')
        owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
        users.bind_address(user_id=owner.user_id, integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
        store = SQLiteIntegrationStore(db)
        store.upsert(integration_id='telegram-test', provider_key='telegram', display_name='Synthetic', enabled=True,
                     config={'allowed_chat_ids': ['123']})
        queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
        runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
            conversation_store=SQLiteConversationStore(db), outbox=SQLiteOutboundStore(db), integration_store=store,
            inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
            experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        daemon = V2Daemon(runtime, daemon_id='telegram-activity-dojo', experimental_owner_only=True,
                          owner_telegram_enabled=True)
        daemon._acquire_single_instance_lock = lambda: None
        daemon._release_single_instance_lock = lambda: None
        daemon.ipc.start = lambda: None
        daemon.ipc.stop = lambda: None
        from alphonse.agent_v2.core.intelligence.simple_loop import owner_telegram_listener as module
        listener, client = module.TelegramOwnerListener, Client()
        with patch.object(store, 'resolve_credentials', return_value={'bot_token': 'synthetic'}), patch.object(
                module, 'TelegramOwnerListener', lambda q, s, i: listener(q, s, i, client_factory=lambda **_: client,
                    clock=lambda: datetime.fromtimestamp(1000, timezone.utc), poll_interval=.1,
                    activity_interval=activity_interval)):
            daemon.start()
            try:
                held = entered.wait(timeout)
                # Hold the response briefly so independently scheduled activity
                # can be recorded; no provider or user data leaves this process.
                if held:
                    time.sleep(hold_seconds)
                if held and cancel_task:
                    daemon.kill_switch.request_cancel_status()
                    events.append({'kind': 'cancellation_requested'})
                    time.sleep(activity_interval * 2 + .05)
                    events.append({'kind': 'cancellation_settled'})
                events.append({'kind': 'model_released'})
                release.set()
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline and not any(x['kind'] == 'reply' for x in events):
                    with daemon._owner_status_lock:
                        failed = 'failed' in daemon._owner_statuses.values()
                    if failed:
                        events.append({'kind': 'task_failed'})
                        time.sleep(activity_interval * 2 + .05)
                        break
                    time.sleep(.02)
                completed = any(x['kind'] == 'reply' for x in events)
            finally:
                release.set()
                daemon.stop()
        checks = assess(events, model_was_held=held, completed=completed)
        return {'case': CASE_ID, 'checks': checks, 'events': events, 'hold_seconds': hold_seconds,
                'limitations': ['Fake Telegram client; provider delivery is not verified.',
                                'Deterministic blocked model; real-model behavior is not measured.',
                                'One run does not establish reliability or exact timing guarantees.']}


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, sort_keys=True))
