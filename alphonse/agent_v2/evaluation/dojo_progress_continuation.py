"""Real-model continuation exercise with fake Jev and disposable host data.

The model may execute unrestricted host Bash. Only the synthetic fixture is
disposable. The injected progress judgment and disclosure check are fake and
do not establish real Jev accuracy or production provider authorization.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def run(model, *, timeout=180):
    events, judgments = [], []
    with tempfile.TemporaryDirectory(prefix='dojo-progress-', dir='/private/tmp') as temp:
        root = Path(temp)
        source = root / 'reading.txt'
        source.write_text('Valve reading: 72.4 psi\n', encoding='utf-8')
        users = V2UserStore(root / 'dojo.sqlite')
        users.set_workspace_root(root / 'workspace')
        users.onboard(display_name='Synthetic owner', users_root=root / 'users')
        users._set_setting('owner_continuation_checkpoint', json.dumps({'metric': 'actions', 'interval': 1}))
        queue = SQLiteMessageQueue(root / 'dojo.sqlite', experimental_claim_fencing=True)
        router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=model))
        router.provider.config = replace(router.provider.config, timeout_seconds=45)
        original_generate = router.generate_json
        def observe(request):
            result = original_generate(request)
            decision = result.json_value
            if isinstance(decision, dict):
                events.append({'event': 'model_decision', 'kind': decision.get('kind'),
                               'tool_id': decision.get('tool_id'), 'arguments': decision.get('arguments'),
                               'text': decision.get('text')})
            return result
        router.generate_json = observe
        runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(root / 'dojo.sqlite'),
            conversation_store=SQLiteConversationStore(root / 'dojo.sqlite'), outbox=SQLiteOutboundStore(root / 'dojo.sqlite'),
            inference=router, experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        class FakeJev:
            settings = SimpleNamespace(no_threshold=.3, yes_threshold=.7)
            def judge_observed_progress(self, *, state):
                judgments.append(state)
                return .9
        runtime.core.system_one = FakeJev()
        runtime.processor.owner_read_candidate.enable_owner_host_bash(trusted_local_development=True)
        daemon = V2Daemon(runtime, daemon_id='synthetic-progress-dojo', experimental_owner_only=True)
        daemon._continuation_experimental_enabled = True
        daemon._acquire_single_instance_lock = lambda: None
        daemon._release_single_instance_lock = lambda: None
        daemon.ipc.start = lambda: None
        daemon.ipc.stop = lambda: None
        prompt = f'Read the latest valve reading in {source} and tell me the value and unit. Leave the file unchanged.'
        with patch.object(CoreLoopContext, 'owner_progress_disclosure_check',
                          lambda self, actor, provider, refs: True, create=True):
            daemon.start()
            try:
                message = queue.enqueue_trusted_local_owner(prompt=prompt, peer_uid=os.getuid())
                deadline = time.monotonic() + timeout
                answered = False
                while time.monotonic() < deadline:
                    status = daemon.owner_message_status(message)
                    if status == 'continuation_paused':
                        items = daemon.owner_continuation_list()
                        if items:
                            item = next((item for item in items if item['message_id'] == message), None)
                            if item:
                                events.append({'event': 'checkpoint', 'question': item['question']})
                                answer = daemon.owner_continuation_choose(message, item['checkpoint_version'], True)
                                events.append({'event': 'synthetic_owner_choice', 'status': answer['status']})
                    elif status == 'ready':
                        draft = daemon.take_owner_desktop_draft(message)
                        events.append({'event': 'draft', 'text': draft['message']})
                        answered = True
                        break
                    elif status in {'failed', 'unavailable'}:
                        events.append({'event': 'stopped', 'status': status})
                        break
                    time.sleep(.1)
            finally:
                daemon.stop()
        drafts = [row['text'] for row in events if row['event'] == 'draft']
        checkpoints = [row for row in events if row['event'] == 'checkpoint']
        tool_ids = [row.get('tool_id') for row in events if row['event'] == 'model_decision']
        return {'case': 'real-model-continuation-fake-jev', 'events': events,
                'fake_judgment_count': len(judgments),
                'fake_judgment_observation_counts': [len(j['observations']) for j in judgments],
                'checks': {'model_used_bash': 'native.bash' in tool_ids,
                           'checkpoint_seen': bool(checkpoints), 'fake_judgment_received_evidence': bool(judgments),
                           'progress_question_seen': any('preliminary review found evidence' in row['question'] for row in checkpoints),
                           'answer_contains_value_and_unit': bool(drafts and '72.4' in drafts[-1] and 'psi' in drafts[-1].lower()),
                           'answered': answered, 'source_unchanged': source.read_text(encoding='utf-8') == 'Valve reading: 72.4 psi\n'},
                'limitations': ['Fake Jev and synthetic disclosure authorization; no real Jev assessment was made.',
                                'Synthetic Desktop choice; no packaged Desktop or real user interaction.',
                                'Trusted host Bash is unrestricted; only the file is disposable.',
                                'One real-model attempt is not a reliability estimate.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--accept-trusted-host', action='store_true')
    args = parser.parse_args()
    if not args.accept_trusted_host:
        parser.error('unrestricted host Bash requires --accept-trusted-host')
    result = run(args.model)
    with args.journal.open('x', encoding='utf-8') as out:
        json.dump(result, out, ensure_ascii=False, sort_keys=True)
        out.write('\n')
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
