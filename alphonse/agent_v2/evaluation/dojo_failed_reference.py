"""Exact failed-request reference exercise using disposable owner-worker IPC."""
from __future__ import annotations

import json
from pathlib import Path
import sqlite3
import tempfile
import time

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.evaluation.dojo_failure_diagnosis import _wait
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore

GOAL = 'Please review the imaginary garden plan.'
FOLLOWUP = 'What happened to this request?'


def run(*, timeout=10):
    """Use two distinct injected failures and inspect the selected model context."""
    start = time.monotonic()
    class Provider:
        supports_image_input = False
        def __init__(self): self.prompts = []
        def generate_json(self, request):
            self.prompts.append(request.prompt)
            if len(self.prompts) == 1:
                raise ValueError('openai_codex_timeout')
            if len(self.prompts) == 2:
                raise ValueError('private synthetic provider failure')
            return InferenceResult(json_value={'kind': 'answer', 'tool_id': None, 'arguments': None,
                                   'text': 'I will check the available record.'}, usage={'input_tokens': 100, 'output_tokens': 20})
    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'scripted', 'fixture'))
    with tempfile.TemporaryDirectory(prefix='dojo-reference-', dir='/tmp') as directory:
        root = Path(directory)
        db = root / 'owner.sqlite'
        users = V2UserStore(db)
        owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
        foreign = users.create_user(display_name='Other synthetic owner', role='admin')
        queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
        conversation = SQLiteConversationStore(db)
        runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
            conversation_store=conversation, outbox=SQLiteOutboundStore(db), inference=router,
            experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        daemon = V2Daemon(runtime, daemon_id='dojo-reference', experimental_owner_only=True)
        daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
        daemon._acquire_single_instance_lock = lambda: None
        daemon._release_single_instance_lock = lambda: None
        daemon.start()
        try:
            client = V2DaemonClient(root / 'owner.sock')
            first = client.request('experimental_owner_ingress', prompt=GOAL)['message_id']
            first_status = _wait(client, first, timeout)
            second = client.request('experimental_owner_ingress', prompt=GOAL)['message_id']
            second_status = _wait(client, second, timeout)
            follow = client.request('experimental_owner_ingress', prompt=FOLLOWUP,
                                    failed_request_message_id=first)['message_id']
            follow_status = _wait(client, follow, timeout)
            marker = '"explicitly_referenced_failed_request": '
            prompt = provider.prompts[-1]
            projection = None
            if marker in prompt:
                projection, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1])
            def count():
                with sqlite3.connect(db) as conn:
                    return conn.execute('SELECT count(*) FROM v2_inbound_messages').fetchone()[0]
            before = count()
            with sqlite3.connect(db) as conn:
                conn.execute("UPDATE v2_user_settings SET value=? WHERE key='admin_user_id'", (foreign.user_id,))
            try:
                client.request('experimental_owner_ingress', prompt=FOLLOWUP, failed_request_message_id=first)
                foreign_error = None
            except Exception as exc:
                foreign_error = str(exc)
            finally:
                with sqlite3.connect(db) as conn:
                    conn.execute("UPDATE v2_user_settings SET value=? WHERE key='admin_user_id'", (owner.user_id,))
            foreign_count = count()
            with sqlite3.connect(db) as conn:
                conn.execute('DELETE FROM v2_conversation_events WHERE source_message_id=?', ('inbound:' + first,))
            try:
                client.request('experimental_owner_ingress', prompt=FOLLOWUP, failed_request_message_id=first)
                stale_error = None
            except Exception as exc:
                stale_error = str(exc)
            after = count()
            return {'scenario': 'exact-failed-request-reference',
                    'first_status': first_status, 'second_status': second_status, 'followup_status': follow_status,
                    'identical_prompts': True, 'different_request_ids': first != second,
                    'projection': projection, 'selected_message_id': first, 'other_message_id': second,
                    'matches_selected_first': isinstance(projection, dict) and projection.get('message_id') == first,
                    'matches_first_timeout': isinstance(projection, dict) and projection.get('description') == 'The model provider timed out.',
                    'not_other_failure': isinstance(projection, dict) and projection.get('message_id') != second,
                    'foreign_rejection': foreign_error, 'stale_rejection': stale_error,
                    'inbound_counts': {'before_rejections': before, 'after_foreign': foreign_count, 'after_stale': after},
                    'original_status_after_followup': client.request('experimental_owner_status', message_id=first),
                    'model_call_count': len(provider.prompts), 'model_cost_usd': 0,
                    'elapsed_seconds': round(time.monotonic() - start, 3),
                    'model_provider': 'scripted fixture', 'semantic_answer_verdict': 'not_assessed',
                    'limitations': ['Disposable owner-worker IPC and synthetic scripted provider; no real model or Desktop rendering.',
                        'The selected context is independently inspected, not evidence the model understood or accurately answered.',
                        'Foreign and stale references are tested at ingress; later revocation after enqueue is not assessed.',
                        'No external effects or retries are authorized; equal inbound counts show only that rejected sends were not enqueued.',
                        'One attempt does not establish general reliability or accessibility.']}
        finally:
            daemon.stop()


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
