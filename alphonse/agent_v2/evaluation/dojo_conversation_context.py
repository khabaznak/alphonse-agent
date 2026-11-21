"""Disposable Dojo observation for a multilingual conversational follow-up.

No weather or network is required: the final question asks the model to
identify the location from earlier synthetic conversation turns. The answer
is retained for review; a matching place name alone is not a semantic grade.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import tempfile
import time
from pathlib import Path

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def run(router: InferenceRouter, *, timeout: float = 90) -> dict:
    with tempfile.TemporaryDirectory(prefix='dojo-context-', dir='/tmp') as directory:
        root = Path(directory)
        path = root / 'owner.sqlite'
        users = V2UserStore(path)
        owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
        queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
        conversation = SQLiteConversationStore(path)
        context_seen = []
        original_generate = router.generate_json
        def inspect_request(request):
            context_seen.append({'prior_location_visible': 'Campeche' in request.prompt,
                                 'prior_topic_visible': 'temperatura' in request.prompt})
            return original_generate(request)
        router.generate_json = inspect_request
        for n, (role, content) in enumerate((
            ('user', '¿Qué temperatura hay en Mérida?'),
            ('assistant', 'El informe anterior indicó 25 grados en Mérida.'),
            ('user', '¿Y en Campeche?'),
            ('assistant', 'Campeche es una ciudad de México.'),
        )):
            conversation.record(owner_user_id=owner.user_id, role=role, content=content,
                                source='desktop', source_message_id=f'dojo:{n}')
        runtime = build_runtime_host(user_store=users, messages=queue,
            question_store=SQLiteQuestionStore(path), conversation_store=conversation,
            outbox=SQLiteOutboundStore(path), inference=router,
            experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        daemon = V2Daemon(runtime, daemon_id='dojo-context', experimental_owner_only=True)
        daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
        daemon._acquire_single_instance_lock = lambda: None
        daemon._release_single_instance_lock = lambda: None
        daemon.start()
        try:
            client = V2DaemonClient(root / 'owner.sock')
            message = client.request('experimental_owner_ingress', prompt=(
                'Me refiero a la temperatura allí. ¿A qué ciudad me refería? '
                'No consultes fuentes externas; solo identifica la ciudad.'))['message_id']
            deadline = time.monotonic() + timeout
            while time.monotonic() < deadline:
                result = client.request('experimental_owner_status', message_id=message)
                if result['status'] not in {'pending', 'processing'}:
                    break
                time.sleep(.05)
            else:
                return {'status': 'timeout', 'passed': None}
            if result['status'] != 'ready':
                return {'status': result['status'], 'failure_code': result.get('failure_code'), 'passed': False}
            draft = client.request('experimental_owner_take_draft', message_id=message)
            client.request('experimental_owner_ack_draft', message_id=message,
                           acknowledgement_token=draft['acknowledgement_token'])
            return {'status': 'ready', 'answer': draft['message'],
                    'expected_location_mentioned': 'campeche' in draft['message'].casefold(),
                    'context_visible_to_provider': context_seen,
                    'passed': None, 'response_semantics': 'requires_manual_review',
                    'limitations': ['Synthetic prior turns; no live weather or Desktop UI.',
                                    'A location match alone does not prove the answer is correct.',
                                    'Provider behavior outside this fixture is not assessed.']}
        finally:
            daemon.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=args.model))
    router.provider.config = replace(router.provider.config, timeout_seconds=40.0)
    with args.journal.open('x', encoding='utf-8') as journal:
        json.dump(run(router), journal, ensure_ascii=False, sort_keys=True)
        journal.write('\n')


if __name__ == '__main__':
    main()
