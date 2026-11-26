"""Run the owner worker on disposable mail and observe the configured model."""
import argparse
import json
from pathlib import Path
import tempfile
import time

from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider, OpenAICodexProviderConfig
from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail_search import OwnerGmailSearchService
from test_owner_calendar import calendar_worker, worker_status
from test_owner_gmail_search_conversation import BODY, GOAL, SUBJECT, f, inbox_fixture, seed_mail


LABEL = 'Family/Spring workshop'
LABEL_ID = 'Label_2042'
LABEL_GOAL = (
    'Busca en la etiqueta de Gmail Family/Spring workshop el aviso del taller '
    'de cerámica. Dime a qué hora empieza y qué tengo que llevar.'
)


class ObservedProvider:
    supports_image_input = True

    def __init__(self):
        self.provider = OpenAICodexProvider(OpenAICodexProviderConfig(timeout_seconds=50))
        self.decisions = []

    def generate_json(self, request):
        row = {'mail_excerpt_in_prompt': BODY in request.prompt,
               'other_mail_tool_in_prompt': 'native.owner_gmail_unread' in request.prompt}
        self.decisions.append(row)
        result = self.provider.generate_json(request)
        arguments = result.json_value.get('arguments') or {}
        row.update(kind=result.json_value.get('kind'), tool_id=result.json_value.get('tool_id'),
                   requested_label=arguments.get('label'),
                   usage=dict(result.usage or {}))
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('case', choices=('authorized', 'count-only', 'label-authorized', 'label-denied'))
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    provider = ObservedProvider()
    wire = inbox_fixture()
    label_case = args.case in {'label-authorized', 'label-denied'}
    goal = LABEL_GOAL if label_case else GOAL
    if label_case:
        wire.responses[-1]['labelIds'] = [LABEL_ID]
    report = {'model': args.model, 'case': args.case, 'request': goal, 'status': 'started'}
    started = time.monotonic()
    with args.receipt.open('x', encoding='utf-8') as evidence:
        evidence.write(json.dumps(report) + '\n')
        evidence.flush()
        try:
            with tempfile.TemporaryDirectory(prefix='dojo-mail-', dir='/tmp') as directory:
                root = Path(directory)
                fixture = f.__wrapped__(root)
                seed_mail(fixture, content_grant=args.case != 'count-only')
                if args.case == 'label-authorized':
                    fixture.accounts.save_grant(
                        fixture.owner, grant_id='workshop-label', connection_id='one',
                        expected_revision=0, beneficiary_user_id='owner',
                        operations=['mail_content_read'], resources=[LABEL_ID],
                        gmail_label_name=LABEL, recipient_user_ids=['owner'],
                        expires_at=None,
                    )
                with calendar_worker(fixture, root, wire, provider, model=args.model, socket=True) as (
                        daemon, _, conversation, client):
                    daemon.runtime.owner_gmail_search_service = OwnerGmailSearchService(
                        fixture.accounts, lambda _: fixture.service, transport=wire)
                    mid = client.request('experimental_owner_ingress', prompt=goal)['message_id']
                    report['status'] = worker_status(lambda: client.request('experimental_owner_status',
                        message_id=mid)['status'], timeout=120)
                    if report['status'] == 'ready':
                        draft = client.request('experimental_owner_take_draft', message_id=mid)
                        report.update(answer=draft['message'], private_reply=draft.get('private_gmail'))
                        report['ack'] = client.request('experimental_owner_ack_draft', message_id=mid,
                            acknowledgement_token=draft['acknowledgement_token'])['status']
                        report['mail_in_history'] = any(SUBJECT in item.content or '11:30' in item.content
                            for item in conversation.list(owner_user_id='owner'))
        except Exception as exc:
            report.update(status='infrastructure_error', error_type=type(exc).__name__)
        report.update(decisions=provider.decisions, duration=round(time.monotonic()-started, 3),
                      gmail_paths=[path for path, _ in wire.calls],
                      gmail_list_scopes=[request['params']['labelIds'] for path, request in wire.calls
                                         if path == 'messages'],
                      limitation='Disposable owner and Gmail data. Real inference, Unix IPC test client. No native Desktop or live Google verification.')
        evidence.write(json.dumps(report, ensure_ascii=False) + '\n')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
