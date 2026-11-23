"""Real-model one-time Telegram reminder using fake Telegram and temporary data.

The owner worker and actual time progression are used. No household account,
real Telegram delivery, or host Bash is enabled by this harness.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore

CASE_ID = 'telegram-real-model-reminder'
PROMPT = 'Please remind me to water the plants in one minute.'


class FakeTelegram:
    def __init__(self):
        self.sent = []

    def get_updates(self, *, offset, timeout):
        if offset is not None and offset > 1:
            return []
        return [{'update_id': 1, 'message': {'date': 1001, 'message_id': 1, 'text': PROMPT,
                 'from': {'id': 123}, 'chat': {'id': 123, 'type': 'private'}}}]

    def send_message(self, *, chat_id, text):
        self.sent.append({'chat_id': chat_id, 'text': text, 'sent_at': datetime.now(timezone.utc).isoformat()})
        return str(100 + len(self.sent))


def run(model, emit, *, timeout=240):
    from alphonse.agent_v2.core.intelligence.simple_loop import owner_telegram_listener
    with tempfile.TemporaryDirectory(prefix='dojo-telegram-reminder-', dir='/private/tmp') as temp:
        root = Path(temp); db = root / 'fixture.sqlite'
        users = V2UserStore(db); users.set_workspace_root(root / 'workspace')
        owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
        users.set_timezone('UTC', owner.user_id)
        users.bind_address(user_id=owner.user_id, integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
        store = SQLiteIntegrationStore(db)
        store.upsert(integration_id='telegram-test', provider_key='telegram', display_name='Synthetic Telegram',
                     enabled=True, config={'allowed_chat_ids': ['123']})
        queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
        conversation = SQLiteConversationStore(db)
        fake = FakeTelegram()
        router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=model))
        router.provider.config = replace(router.provider.config, timeout_seconds=40)
        generate = router.generate_json
        def observe(request):
            try:
                result = generate(request)
            except Exception as exc:
                emit({'event': 'model_failure', 'type': type(exc).__name__})
                raise
            value = result.json_value if isinstance(result.json_value, dict) else {}
            emit({'event': 'model_decision', 'kind': value.get('kind'), 'tool_id': value.get('tool_id'),
                  'arguments': value.get('arguments'), 'text': value.get('text'), 'usage': result.usage})
            return result
        router.generate_json = observe
        runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
            conversation_store=conversation, outbox=SQLiteOutboundStore(db), integration_store=store, inference=router,
            experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        daemon = V2Daemon(runtime, daemon_id='telegram-reminder-dojo', experimental_owner_only=True,
                          owner_telegram_enabled=True)
        daemon._acquire_single_instance_lock = lambda: None
        daemon._release_single_instance_lock = lambda: None
        daemon.ipc.start = lambda: None; daemon.ipc.stop = lambda: None
        real_listener = owner_telegram_listener.TelegramOwnerListener
        with patch.object(store, 'resolve_credentials', return_value={'bot_token': 'synthetic'}), patch.object(
                owner_telegram_listener, 'TelegramOwnerListener',
                lambda q, s, i: real_listener(q, s, i, client_factory=lambda **_: fake,
                    clock=lambda: datetime.fromtimestamp(1000, timezone.utc), poll_interval=.1)):
            daemon.start()
            try:
                deadline = time.monotonic() + timeout
                schedules = []; last_state = None
                while time.monotonic() < deadline:
                    schedules = runtime.owner_schedule_store.list_tasks(owner_user_id=owner.user_id)
                    with queue._connect() as conn:
                        table = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='v2_owner_schedule_occurrences'").fetchone()
                        occurrences = [dict(row) for row in conn.execute('SELECT * FROM v2_owner_schedule_occurrences').fetchall()] if table else []
                    state = (len(schedules), len(fake.sent), tuple(item['state'] for item in occurrences))
                    if state != last_state:
                        emit({'event': 'progress', 'schedules': len(schedules), 'sent': len(fake.sent),
                              'occurrence_states': [item['state'] for item in occurrences]})
                        last_state = state
                    if len(fake.sent) >= 2 and any(item['state'] == 'completed' for item in occurrences):
                        break
                    if fake.sent and not schedules:
                        break
                    if schedules and 'failed' in daemon._owner_statuses.values():
                        break
                    if not fake.sent and 'failed' in daemon._owner_statuses.values():
                        break
                    time.sleep(.2)
                task = schedules[0] if len(schedules) == 1 else None
                due = getattr(task, 'next_run_at', None) if task else None
                # Completed schedules clear next_run_at; use the immutable schedule JSON.
                if task:
                    with queue._connect() as conn:
                        row = conn.execute('SELECT schedule_json FROM v2_scheduled_tasks WHERE scheduled_task_id=?', (task.scheduled_task_id,)).fetchone()
                    due = json.loads(row['schedule_json']).get('run_at') if row else due
                sent = list(fake.sent)
                second_at = datetime.fromisoformat(sent[1]['sent_at']) if len(sent) >= 2 else None
                due_time = datetime.fromisoformat(due) if due else None
                return {'case': CASE_ID, 'prompt': PROMPT, 'schedule_count': len(schedules),
                        'schedule_due_at': due, 'fake_telegram_sends': sent,
                        'due_response_not_early': bool(second_at and due_time and second_at >= due_time),
                        'occurrence_states': [item['state'] for item in occurrences],
                        'worker_statuses': list(daemon._owner_statuses.values()),
                        'response_semantics': 'requires_manual_review',
                        'limitations': ['Fake Telegram does not establish real provider delivery.',
                                        'One model attempt is not a reliability estimate.',
                                        'Response content requires manual review.']}
            finally:
                daemon.stop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--journal', required=True, type=Path)
    args = parser.parse_args()
    with args.journal.open('x', encoding='utf-8') as journal:
        def emit(event):
            journal.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + '\n')
            journal.flush(); os.fsync(journal.fileno())
        emit({'event': 'started', 'case': CASE_ID, 'model': args.model, 'prompt': PROMPT})
        try:
            result = run(args.model, emit)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'runner_error', 'error_type': type(exc).__name__})
            raise
        emit({'event': 'finished', **result})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
