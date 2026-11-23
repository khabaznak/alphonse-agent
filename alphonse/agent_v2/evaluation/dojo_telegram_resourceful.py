"""Synthetic Telegram owner conversation with real model and disposable host data.

The Telegram client is fake, but the model and Bash may access the local host
as the current user. No household data or real Telegram account is used.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sqlite3
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

CASE_ID = 'telegram-lead-followup'
LEADS = 'name,status\nMira,Follow up\nEmil,Closed\nCato,Follow up\n'


class FakeTelegramClient:
    def __init__(self, prompt):
        self.prompt, self.sent = prompt, []
    def get_updates(self, *, offset, timeout):
        if offset is not None and offset > 1:
            return []
        return [{'update_id': 1, 'message': {'date': 1001, 'message_id': 1, 'text': self.prompt,
                    'from': {'id': 123}, 'chat': {'id': 123, 'type': 'private'}}}]
    def send_message(self, *, chat_id, text):
        self.sent.append({'chat_id': chat_id, 'text': text})
        return 'synthetic-provider-ack'


def seed(folder):
    folder.mkdir(mode=0o700)
    (folder / 'leads.csv').write_text(LEADS, encoding='utf-8')
    return (f'I keep my sales leads in {folder}. Please create a concise follow-up report '
            f'at {folder / "follow-up.md"} listing the leads whose status calls for follow-up '
            'and their total count. Leave the original data unchanged.')


def grade(folder, reply, bash_records):
    output = folder / 'follow-up.md'
    try:
        contents = output.read_text(encoding='utf-8')
        source_unchanged = (folder / 'leads.csv').read_text(encoding='utf-8') == LEADS
    except (OSError, UnicodeError):
        return {'report_exists': output.is_file(), 'source_unchanged': None, 'response_semantics': 'requires_manual_review'}
    return {'report_exists': True, 'source_unchanged': source_unchanged,
            'report_content': contents[:4096], 'report_truncated': len(contents) > 4096,
            'follow_up_names_present': all(name in contents for name in ('Mira', 'Cato')),
            'closed_name_mentioned': 'Emil' in contents, 'count_two_mentioned': '2' in contents,
            'bash_calls_observed': len(bash_records), 'reply_available': bool(reply),
            'response_semantics': 'requires_manual_review'}


def run(model, timeout=240):
    from datetime import datetime, timezone
    from alphonse.agent_v2.core.intelligence.simple_loop import owner_telegram_listener
    calls = {'started': 0, 'returned': 0, 'bash_decisions': 0, 'failures': []}
    with tempfile.TemporaryDirectory(prefix='dojo-telegram-host-', dir='/private/tmp') as temp:
        root = Path(temp)
        db = root / 'dojo.sqlite'
        users = V2UserStore(db)
        users.set_workspace_root(root / 'workspace')
        owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
        users.bind_address(user_id=owner.user_id, integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
        folder = users.ordinary_workspace(owner.user_id) / 'prospects'
        prompt = seed(folder)
        store = SQLiteIntegrationStore(db)
        store.upsert(integration_id='telegram-test', provider_key='telegram', display_name='Synthetic', enabled=True,
                     config={'allowed_chat_ids': ['123']})
        fake = FakeTelegramClient(prompt)
        queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
        conversation = SQLiteConversationStore(db)
        router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=model))
        router.provider.config = replace(router.provider.config, timeout_seconds=40)
        generate = router.generate_json
        def observe(request):
            calls['started'] += 1
            calls['bash_advertised'] = 'native.bash' in request.prompt
            try:
                result = generate(request)
            except Exception as exc:
                calls['failures'].append(type(exc).__name__)
                raise
            calls['returned'] += 1
            if isinstance(result.json_value, dict) and result.json_value.get('tool_id') == 'native.bash':
                calls['bash_decisions'] += 1
            return result
        router.generate_json = observe
        runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
            conversation_store=conversation, outbox=SQLiteOutboundStore(db), integration_store=store, inference=router,
            experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        runtime.processor.owner_read_candidate.enable_owner_host_bash(trusted_local_development=True)
        real_listener = owner_telegram_listener.TelegramOwnerListener
        daemon = V2Daemon(runtime, daemon_id='telegram-resourcefulness-dojo', experimental_owner_only=True,
                          owner_telegram_enabled=True)
        daemon._acquire_single_instance_lock = lambda: None
        daemon._release_single_instance_lock = lambda: None
        daemon.ipc.start = lambda: None
        daemon.ipc.stop = lambda: None
        with patch.object(store, 'resolve_credentials', return_value={'bot_token': 'synthetic'}), patch.object(
                owner_telegram_listener, 'TelegramOwnerListener',
                lambda q, s, i: real_listener(q, s, i, client_factory=lambda **_: fake,
                    clock=lambda: datetime.fromtimestamp(1000, timezone.utc), poll_interval=.1)):
            daemon.start()
            try:
                deadline = time.monotonic() + timeout
                status = None
                while time.monotonic() < deadline:
                    with sqlite3.connect(db) as conn:
                        row = conn.execute('SELECT status FROM v2_owner_telegram_replies').fetchone()
                        status = row[0] if row else None
                    if status in {'finalized', 'started_unconfirmed'}:
                        break
                    with daemon._owner_status_lock:
                        if 'failed' in daemon._owner_statuses.values():
                            status = 'failed'
                            break
                    time.sleep(.1)
            finally:
                daemon.stop()
        records = []
        with sqlite3.connect(db) as conn:
            rows = conn.execute('SELECT task_state_json FROM v2_task_checkpoints').fetchall()
            for (raw,) in rows:
                metadata = json.loads(raw).get('metadata', {})
                for item in metadata.get('owner_read_observations', {}).values():
                    if isinstance(item, dict) and isinstance(item.get('host_bash_observation'), dict):
                        evidence = item['host_bash_observation']
                        records.append({'exit_code': evidence.get('exit_code'), 'cancelled': evidence.get('cancelled'),
                                        'timed_out': evidence.get('timed_out')})
        answer = fake.sent[0]['text'] if fake.sent else None
        return {'case': CASE_ID, 'status': status or 'unresolved', 'calls': calls, 'bash_processes': records,
                'fake_telegram_sends': len(fake.sent), 'answer': answer, 'checks': grade(folder, answer, records),
                'limitations': ['Real model and unrestricted host Bash; fixture is disposable, not a sandbox.',
                    'Telegram client is fake; no real provider delivery is established.',
                    'Report and response semantics require manual review.', 'One attempt is not a reliability estimate.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--accept-trusted-host', action='store_true')
    args = parser.parse_args()
    if not args.accept_trusted_host:
        parser.error('unrestricted host Bash requires --accept-trusted-host')
    with args.journal.open('x', encoding='utf-8') as stream:
        json.dump({'case': CASE_ID, 'status': 'running', 'model': args.model}, stream)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        try:
            result = run(args.model)
        except Exception as exc:
            result = {'case': CASE_ID, 'status': 'runner_error', 'error_type': type(exc).__name__}
            stream.seek(0); stream.truncate(); json.dump(result, stream); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
            raise
        stream.seek(0); stream.truncate(); json.dump(result, stream, ensure_ascii=False, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
