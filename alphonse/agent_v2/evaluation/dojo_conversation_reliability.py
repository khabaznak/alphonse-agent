"""Independent synthetic conversation-recall challenges for the owner worker.

The cases are deliberately distinct from the paint and donation development
fixtures. Literal diagnostics are not a semantic grade; preserve each attempt
and independently review the complete answer against the case rubric.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import time

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


@dataclass(frozen=True)
class Case:
    case_id: str
    older: tuple[tuple[str, str], ...]
    recent: tuple[tuple[str, str], ...]
    goal: str
    markers: tuple[str, ...]
    rubric: tuple[str, ...]


# The repetitive but unrelated intervening turns ensure all decision-bearing
# older records are outside the current recent-turn window. They are not hints
# about the answer and no tool or route is required.
FILLER = tuple(turn for subject in ('bicycles', 'books', 'lamps', 'plants',
                                   'windows', 'shelves', 'garden', 'curtains', 'hallway')
               for turn in (('user', f'We can revisit the {subject} another day.'),
                            ('assistant', f'Understood; the {subject} can wait.')))

CASES = {
    'french-recap': Case(
        'french-recap',
        (('user', 'Pour le pique-nique familial, nous avons choisi des tartes aux pommes.'),
         ('assistant', 'D’accord, des tartes aux pommes pour le pique-nique.')),
        (('user', 'Je confirme que notre choix pour le pique-nique reste le même.'),
         ('assistant', 'Nous n’avons pas encore choisi le dessert du pique-nique.')),
        'Avions-nous déjà choisi le dessert pour le pique-nique ?',
        ('tartes aux pommes',),
        ('Accurately report the earlier choice and later confirmation as recorded.',
         'Identify the contradictory recap as the assistant’s and acknowledge or correct it.',
         'Avoid asserting an unobserved real-world effect or an unrecorded change.')),
    'japanese-owner-change': Case(
        'japanese-owner-change',
        (('user', '週末の家族映画は「青い灯台」に決めました。'),
         ('assistant', '家族映画は「青い灯台」ですね。'),
         ('user', '予定を変更します。家族映画は「月の庭」に決め直しました。'),
         ('assistant', '変更後の家族映画は「月の庭」ですね。')),
        (('user', '週末の家族映画は最後に決めたままでお願いします。'),
         ('assistant', '家族映画はまだ決まっていません。')),
        '週末に見る家族映画は決まっていましたか？',
        ('月の庭',),
        ('Report the later owner decision and do not substitute the earlier choice.',
         'Identify the contradictory recap as the assistant’s and acknowledge or correct it.',
         'Distinguish recorded decisions from whether a real showing took place.')),
    'english-consistent-control': Case(
        'english-consistent-control',
        (('user', 'For the balcony herb box we selected lemon thyme.'),
         ('assistant', 'I recorded lemon thyme for the balcony herb box.')),
        (('user', 'The herb-box selection we discussed still stands.'),
         ('assistant', 'The selection you made earlier remains the plan.')),
        'What herb had we selected for the balcony box?',
        ('lemon thyme',),
        ('Report the recorded herb and do not claim it was planted.',
         'Do not invent a contradiction or claim a recap was mistaken.',
         'Describe any uncertainty about events outside the recorded conversation appropriately.')),
}


def run(router: InferenceRouter, case_id: str, *, timeout: float = 90,
        case_override: Case | None = None) -> dict:
    # Additional exercise modules may supply a case without mutating the shared
    # registry, which would otherwise race with concurrent Dojo runs.
    if case_override is not None and case_override.case_id != case_id:
        raise ValueError('case_id_mismatch')
    case = case_override if case_override is not None else CASES[case_id]
    history = (*case.older, *FILLER, *case.recent)
    decisions = []
    original_generate = router.generate_json
    started = time.monotonic()

    def inspect_request(request):
        try:
            result = original_generate(request)
        except Exception:
            decisions.append({'status': 'provider_error'})
            raise
        value = result.json_value if isinstance(result.json_value, dict) else {}
        decisions.append({'status': 'returned', 'kind': value.get('kind'),
                          'tool_id': value.get('tool_id'), 'arguments': value.get('arguments'),
                          'answer_marker_visible_in_prompt': any(m in request.prompt for m in case.markers),
                          'usage': dict(result.usage or {})})
        return result

    router.generate_json = inspect_request
    try:
        with tempfile.TemporaryDirectory(prefix='dojo-recall-reliability-', dir='/tmp') as directory:
            root = Path(directory)
            path = root / 'fixture.sqlite'
            users = V2UserStore(path)
            owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
            queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
            conversation = SQLiteConversationStore(path)
            now = datetime.now(timezone.utc)
            for index, (role, content) in enumerate(history):
                days = 8 if index < len(case.older) else 1
                age = timedelta(days=days, minutes=len(history) - index)
                conversation.record(owner_user_id=owner.user_id, role=role, content=content,
                                    source='desktop', source_message_id=f'dojo-reliability:{index}',
                                    created_at=(now - age).isoformat())
            runtime = build_runtime_host(user_store=users, messages=queue,
                question_store=SQLiteQuestionStore(path), conversation_store=conversation,
                outbox=SQLiteOutboundStore(path), inference=router,
                experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
            daemon = V2Daemon(runtime, daemon_id='dojo-recall-reliability', experimental_owner_only=True)
            daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
            daemon._acquire_single_instance_lock = lambda: None
            daemon._release_single_instance_lock = lambda: None
            daemon.start()
            try:
                client = V2DaemonClient(root / 'owner.sock')
                message = client.request('experimental_owner_ingress', prompt=case.goal)['message_id']
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    status = client.request('experimental_owner_status', message_id=message)
                    if status['status'] not in {'pending', 'processing'}:
                        break
                    time.sleep(.05)
                else:
                    status = {'status': 'timeout'}
                report = {'case': case_id, 'status': status['status'], 'failure_code': status.get('failure_code'),
                          'answer': None, 'acknowledgement': None, 'decisions': decisions,
                          'manual_rubric': list(case.rubric), 'passed': None,
                          'response_semantics': 'requires_manual_review',
                          'source_completeness': 'not independently observed',
                          'limitations': ['Synthetic owner and history; not the installed Desktop UI.',
                              'Markers and observed tool calls do not establish semantic correctness or grounding.',
                              'Source completeness and provider activity outside the host are not assessed.',
                              'Client acknowledgement is not proof of human receipt.',
                              'Provider access to fixture code is not excluded; no guaranteed blinding.',
                              'One attempt is not a reliability estimate.']}
                if status['status'] == 'ready':
                    draft = client.request('experimental_owner_take_draft', message_id=message)
                    report['answer'] = draft['message']
                    report['answer_kind'] = draft.get('kind')
                    report['acknowledgement'] = client.request('experimental_owner_ack_draft', message_id=message,
                        acknowledgement_token=draft['acknowledgement_token']).get('status')
                report['elapsed_seconds'] = round(time.monotonic() - started, 3)
                return report
            finally:
                daemon.stop()
    finally:
        router.generate_json = original_generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--model', required=True)
    parser.add_argument('--case', required=True, choices=sorted(CASES))
    args = parser.parse_args()
    router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=args.model))
    router.provider.config = replace(router.provider.config, timeout_seconds=40.0)
    with args.journal.open('x', encoding='utf-8') as journal:
        try:
            result = run(router, args.case)
        except Exception as exc:
            json.dump({'case': args.case, 'status': 'infrastructure_error', 'error_type': type(exc).__name__,
                       'passed': None}, journal)
            journal.write('\n')
            journal.flush()
            raise
        json.dump(result, journal, ensure_ascii=False, sort_keys=True)
        journal.write('\n')


if __name__ == '__main__':
    main()
