"""Disposable multi-step activity and recovery exercise; Critic reviews UX separately."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import threading
import time

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.evaluation.dojo_failure_diagnosis import _wait
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore

GOAL = 'What was the latest plan for the imaginary neighborhood picnic, and what remains uncertain?'


def run(router, *, timeout=30):
    """Record one attempt. A fault follows two model-selected tool calls, if reached."""
    started = time.monotonic()
    calls = []
    original = router.generate_json
    at_fault = threading.Event()
    release = threading.Event()
    selected = 0
    injected = False

    def provider(request):
        nonlocal selected, injected
        if selected >= 2 and not injected:
            at_fault.set()
            release.wait(timeout)
            injected = True
            calls.append({'status': 'injected_failure'})
            raise RuntimeError('synthetic provider fault')
        try:
            result = original(request)
        except Exception:
            calls.append({'status': 'provider_error'})
            raise
        value = result.json_value if isinstance(result.json_value, dict) else {}
        if value.get('kind') == 'tool':
            selected += 1
        calls.append({'status': 'returned', 'kind': value.get('kind'),
                      'tool_id': value.get('tool_id'), 'usage': dict(result.usage or {})})
        return result

    router.generate_json = provider
    try:
        with tempfile.TemporaryDirectory(prefix='dojo-activity-', dir='/tmp') as directory:
            root = Path(directory)
            db = root / 'owner.sqlite'
            users = V2UserStore(db)
            owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
            queue = SQLiteMessageQueue(db, experimental_claim_fencing=True)
            conversation = SQLiteConversationStore(db)
            for index, content in enumerate((
                'For the imaginary neighborhood picnic, we considered the riverside lawn.',
                'The latest imaginary picnic plan is the community garden on Saturday, weather permitting.',
                'We have not confirmed whether the garden is available.',
            )):
                conversation.record(owner_user_id=owner.user_id, role='user', content=content,
                                    source='desktop', source_message_id=f'dojo:plan:{index}')
            for index in range(18):
                conversation.record(owner_user_id=owner.user_id, role='user' if index % 2 == 0 else 'assistant',
                                    content=f'Unrelated synthetic note {index}.', source='desktop',
                                    source_message_id=f'dojo:filler:{index}')
            runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(db),
                conversation_store=conversation, outbox=SQLiteOutboundStore(db), inference=router,
                experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
            daemon = V2Daemon(runtime, daemon_id='dojo-activity', experimental_owner_only=True)
            daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
            daemon._acquire_single_instance_lock = lambda: None
            daemon._release_single_instance_lock = lambda: None
            daemon.start()
            try:
                client = V2DaemonClient(root / 'owner.sock')
                message_id = client.request('experimental_owner_ingress', prompt=GOAL)['message_id']
                deadline = time.monotonic() + timeout
                live = None
                while time.monotonic() < deadline:
                    if at_fault.wait(.05):
                        live = client.request('experimental_owner_work_log', message_id=message_id)
                        break
                    status = client.request('experimental_owner_status', message_id=message_id)
                    if status.get('status') not in {'pending', 'processing'}:
                        break
                release.set()
                status = _wait(client, message_id, max(0.1, deadline - time.monotonic()))
                log = client.request('experimental_owner_work_log', message_id=message_id)
                recovery = client.request('experimental_owner_recovery_list')
                return {
                    'scenario': 'activity-ux', 'goal': GOAL, 'initial_ipc': status,
                    'live_work_log': live, 'terminal_work_log': log, 'recovery': recovery,
                    'fault': 'injected_after_two_selected_tools' if injected else 'not_triggered',
                    'calls': calls, 'elapsed_seconds': round(time.monotonic() - started, 3),
                    'product_path': 'disposable owner worker and IPC, not Desktop rendering',
                    'critic_verdict': 'not_assessed', 'passed': None,
                    'limitations': [
                        'Synthetic conversation and injected model failure; no household data or real-world outcome is established.',
                        'The trigger counts model selections; the work log independently shows what was actually recorded.',
                        'Desktop rendering, accessibility, recovery execution and human comprehension are not observed by this runner.',
                        'Scripted providers test plumbing only; provider-reported token usage is retained when available, not a dollar-cost estimate.',
                        'A single attempt does not establish general reliability, grounding, privacy or language coverage.',
                    ],
                }
            finally:
                release.set()
                daemon.stop()
    finally:
        router.generate_json = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=args.model))
    router.provider.config = replace(router.provider.config, timeout_seconds=40.0)
    with args.journal.open('x', encoding='utf-8') as journal:
        try:
            result = run(router, timeout=90)
        except Exception as exc:
            json.dump({'scenario': 'activity-ux', 'status': 'infrastructure_error',
                       'error_type': type(exc).__name__, 'passed': None, 'critic_verdict': 'not_assessed'}, journal)
            journal.write('\n')
            journal.flush()
            raise
        json.dump(result, journal, ensure_ascii=False, sort_keys=True)
        journal.write('\n')


if __name__ == '__main__':
    main()
