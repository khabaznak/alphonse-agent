"""Disposable Dojo observation for a multilingual conversational follow-up.

No weather or network is required: the final question asks the model to
identify the location from earlier synthetic conversation turns. The answer
is retained for review; a matching place name alone is not a semantic grade.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
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


# A deliberately different domain from the incident. The latest recap is
# misleading; the older owner statement is the independent fixture evidence.
HISTORY = (
    ('user', 'The dining room paint we chose is Harbor Blue, finish eggshell.'),
    ('assistant', 'Noted: Harbor Blue in an eggshell finish for the dining room.'),
    *tuple(turn for topic in ('garden', 'books', 'bicycles', 'pantry', 'curtains',
                               'lighting', 'plants', 'storage', 'hallway')
           for turn in (('user', f'We can discuss the {topic} later.'),
                        ('assistant', f'Understood, we can revisit the {topic}.'))),
    ('assistant', 'We still need to choose the dining room paint color.'),
)
HISTORY_GOAL = 'Had we already chosen the dining room paint?'


def run_historical_correction(router: InferenceRouter, *, timeout: float = 90) -> dict:
    """Observe recovery of an older fact after an erroneous recent recap.

    Markers are diagnostics only. A human must assess the entire answer and
    whether it corrects the recap without overstating what the fixture proves.
    """
    started = time.monotonic()
    decisions = []
    original_generate = router.generate_json

    def inspect_request(request):
        try:
            result = original_generate(request)
        except Exception:
            decisions.append({'status': 'provider_error'})
            raise
        value = result.json_value if isinstance(result.json_value, dict) else {}
        decisions.append({'status': 'returned', 'kind': value.get('kind'),
                          'tool_id': value.get('tool_id'), 'arguments': value.get('arguments'),
                          'historical_fact_visible_in_prompt': 'Harbor Blue' in request.prompt,
                          'usage': dict(result.usage or {})})
        return result

    router.generate_json = inspect_request
    try:
        with tempfile.TemporaryDirectory(prefix='dojo-history-', dir='/tmp') as directory:
            root = Path(directory)
            path = root / 'owner.sqlite'
            users = V2UserStore(path)
            owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
            queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
            conversation = SQLiteConversationStore(path)
            now = datetime.now(timezone.utc)
            for n, (role, content) in enumerate(HISTORY):
                age = timedelta(days=8 if n < 2 else 1, minutes=len(HISTORY) - n)
                conversation.record(owner_user_id=owner.user_id, role=role, content=content,
                                    source='desktop', source_message_id=f'dojo-history:{n}',
                                    created_at=(now - age).isoformat())
            runtime = build_runtime_host(user_store=users, messages=queue,
                question_store=SQLiteQuestionStore(path), conversation_store=conversation,
                outbox=SQLiteOutboundStore(path), inference=router,
                experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
            daemon = V2Daemon(runtime, daemon_id='dojo-history', experimental_owner_only=True)
            daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
            daemon._acquire_single_instance_lock = lambda: None
            daemon._release_single_instance_lock = lambda: None
            daemon.start()
            try:
                client = V2DaemonClient(root / 'owner.sock')
                message = client.request('experimental_owner_ingress', prompt=HISTORY_GOAL)['message_id']
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    status = client.request('experimental_owner_status', message_id=message)
                    if status['status'] not in {'pending', 'processing'}:
                        break
                    time.sleep(.05)
                else:
                    status = {'status': 'timeout'}
                report = {'case': 'historical-correction', 'status': status['status'],
                          'failure_code': status.get('failure_code'), 'answer': None,
                          'acknowledgement': None, 'expected_fact_mentioned': None,
                          'response_semantics': 'requires_manual_review', 'passed': None,
                          'source_completeness': 'not independently observed',
                          'decisions': decisions,
                          'limitations': ['Synthetic owner and historical turns; not the installed Desktop UI.',
                              'Literal mentions do not establish correction, grounding, or semantic accuracy.',
                              'Source completeness is not independently assessed; provider activity outside the host is not observed.',
                              'Client acknowledgement is not proof of human receipt.',
                              'One attempt is not a reliability estimate.']}
                if status['status'] == 'ready':
                    draft = client.request('experimental_owner_take_draft', message_id=message)
                    report['answer'] = draft['message']
                    report['answer_kind'] = draft.get('kind')
                    report['expected_fact_mentioned'] = 'harbor blue' in draft['message'].casefold()
                    ack = client.request('experimental_owner_ack_draft', message_id=message,
                                         acknowledgement_token=draft['acknowledgement_token'])
                    report['acknowledgement'] = ack.get('status')
                report['elapsed_seconds'] = round(time.monotonic() - started, 3)
                return report
            finally:
                daemon.stop()
    finally:
        router.generate_json = original_generate


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
    parser.add_argument('--case', choices=('context', 'historical-correction'), default='context')
    args = parser.parse_args()
    router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=args.model))
    router.provider.config = replace(router.provider.config, timeout_seconds=40.0)
    with args.journal.open('x', encoding='utf-8') as journal:
        try:
            result = run_historical_correction(router) if args.case == 'historical-correction' else run(router)
        except Exception as exc:
            json.dump({'case': args.case, 'status': 'infrastructure_error',
                       'error_type': type(exc).__name__, 'passed': None}, journal)
            journal.write('\n')
            journal.flush()
            raise
        json.dump(result, journal, ensure_ascii=False, sort_keys=True)
        journal.write('\n')


if __name__ == '__main__':
    main()
