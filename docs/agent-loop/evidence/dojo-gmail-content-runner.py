"""Run the owner worker on disposable mail and observe the configured model."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import tempfile
import time

from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider, OpenAICodexProviderConfig
from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail_search import OwnerGmailSearchService
from test_gmail_adapter import Wire, message
from test_owner_calendar import calendar_worker, worker_status
from test_owner_gmail_search_conversation import BODY, GOAL, SUBJECT, f, inbox_fixture, seed_mail


LABEL = 'Family/Spring workshop'
LABEL_ID = 'Label_2042'
LABEL_GOAL = (
    'Busca en la etiqueta de Gmail Family/Spring workshop el aviso del taller '
    'de cerámica. Dime a qué hora empieza y qué tengo que llevar.'
)
EXPANDED_GOAL = (
    'Revisa los avisos de Gmail de los últimos 30 días en Family/Spring workshop. '
    '¿Cuál es la fecha límite vigente del registro y qué tareas necesito hacer?'
)
REVISED_DEADLINE = 'Registration now closes December 3 at 17:00 Mexico City time.'
SCHOOL_LABEL = 'Family/School notices'
SCHOOL_LABEL_ID = 'Label_3220'
SENDER_GOAL = '¿Tengo algún correo de Miss Maribel de la escuela de mis hijos?'
SENDER_SUBJECT = 'Actualización de la excursión'
SENDER_BODY = 'La excursión sale a las 8:15 por la puerta norte.'
MENTION_SUBJECT = 'Minuta del comité de familias'


class SenderMailbox:
    """Small Gmail stand-in for sender terms, plain terms, label and epoch bounds."""

    def __init__(self, *, empty_first):
        self.calls = []
        self.queries = []
        self.empty_first = empty_first
        today = datetime.now(timezone.utc)
        self.mail = {}
        for identifier, sender, subject, body, age in (
                ('trip', 'Maribel Ortiz <office-notices@example.invalid>', SENDER_SUBJECT, SENDER_BODY, 8),
                ('committee', 'Eva Costa <committee@example.invalid>', MENTION_SUBJECT,
                 'Maribel fue mencionada durante la reunión.', 2)):
            item = message(identifier, labels=(SCHOOL_LABEL_ID,), text=body)
            item['internalDate'] = str(int((today - timedelta(days=age)).timestamp() * 1000))
            item['payload']['headers'] = [{'name': 'Subject', 'value': subject}, {'name': 'From', 'value': sender}]
            self.mail[identifier] = (item, sender.casefold(), f'{sender} {subject} {body}'.casefold())

    def get(self, path, **kwargs):
        self.calls.append((path, kwargs))
        params = kwargs['params']
        if path == 'messages':
            raw_query = params['q']
            query, after, before = re.fullmatch(r'(.*?)after:(\d+) before:(\d+)', raw_query).groups()
            query = query.strip().removeprefix('(').removesuffix(')').strip()
            self.queries.append({'label_id': params['labelIds'], 'query': query})
            ids = []
            if not (self.empty_first and len(self.queries) == 1):
                for identifier, (item, sender, content) in self.mail.items():
                    received = int(item['internalDate']) / 1000
                    if (params['labelIds'] in item['labelIds'] and int(after) < received < int(before)
                            and self._matches(query, sender, content)):
                        ids.append({'id': identifier})
            result = {'messages': ids}
        else:
            result = self.mail[path.removeprefix('messages/')][0]
        return result, len(json.dumps(result).encode())

    @staticmethod
    def _matches(query, sender, content):
        query = re.sub(r'\bfrom:\s*\(([^()]*)\)', r'from:\1', query, flags=re.IGNORECASE)
        for alternative in re.split(r'\s+OR\s+', query):
            terms = re.findall(r'(from:\s*)?(?:"([^"]+)"|([\w@.\-]+))', alternative, flags=re.IGNORECASE)
            if all((phrase or word).casefold() in (sender if field else content)
                   for field, phrase, word in terms):
                return True
        return False


def expanded_fixture():
    bodies = {
        2: 'The original registration deadline was November 29 at 09:00.',
        7: 'Bring a plain apron. A separate deadline for this task was not provided.',
        8: REVISED_DEADLINE + ' This replaces the November 29 deadline.',
        11: 'Please bring proof of registration. The office opens at 09:00, timezone unspecified.',
    }
    messages = []
    now = datetime.now(timezone.utc)
    for index in range(1, 13):
        item = message(f'workshop-{index}', labels=(LABEL_ID,), text=bodies.get(index, 'General workshop notice.'))
        received = now - timedelta(hours=12 if index == 2 else 3)
        item['internalDate'] = str(int(received.timestamp() * 1000))
        item['payload']['headers'] = [{'name': 'Subject', 'value': f'Synthetic Workshop notice {index}'}]
        messages.append(item)
    return Wire({'messages': [{'id': item['id']} for item in messages], 'nextPageToken': 'unreviewed'}, *messages)


class ObservedProvider:
    supports_image_input = True

    def __init__(self):
        self.provider = OpenAICodexProvider(OpenAICodexProviderConfig(timeout_seconds=50))
        self.decisions = []

    def generate_json(self, request):
        row = {'mail_excerpt_in_prompt': BODY in request.prompt,
               'source_beyond_five_in_prompt': REVISED_DEADLINE in request.prompt,
               'other_mail_tool_in_prompt': 'native.owner_gmail_unread' in request.prompt}
        catalog_text = request.prompt.split('Alphonse has the following host tools available: ', 1)[1]
        tools, _ = json.JSONDecoder().raw_decode(catalog_text)
        row.update(school_label_available=SCHOOL_LABEL in request.prompt,
                   sender_excerpt_in_prompt=SENDER_BODY in request.prompt,
                   gmail_search_tool_available=any(tool['tool_id'] == 'native.owner_gmail_search' for tool in tools),
                   available_tool_count=len(tools))
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
    parser.add_argument('case', choices=('authorized', 'count-only', 'label-authorized', 'label-denied', 'label-expanded',
                                         'sender', 'sender-empty-first'))
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    provider = ObservedProvider()
    expanded = args.case == 'label-expanded'
    wire = expanded_fixture() if expanded else inbox_fixture()
    label_case = args.case.startswith('label-')
    goal = EXPANDED_GOAL if expanded else LABEL_GOAL if label_case else GOAL
    sender_case = args.case.startswith('sender')
    if sender_case:
        goal = SENDER_GOAL
        wire = SenderMailbox(empty_first=args.case == 'sender-empty-first')
    if label_case and not expanded:
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
                if args.case in {'label-authorized', 'label-expanded'}:
                    fixture.accounts.save_grant(
                        fixture.owner, grant_id='workshop-label', connection_id='one',
                        expected_revision=0, beneficiary_user_id='owner',
                        operations=['mail_content_read'], resources=[LABEL_ID],
                        gmail_label_name=LABEL, recipient_user_ids=['owner'],
                        expires_at=None,
                    )
                if sender_case:
                    fixture.accounts.save_grant(
                        fixture.owner, grant_id='school-label', connection_id='one', expected_revision=0,
                        beneficiary_user_id='owner', operations=['mail_content_read'], resources=[SCHOOL_LABEL_ID],
                        gmail_label_name=SCHOOL_LABEL, recipient_user_ids=['owner'], expires_at=None,
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
                        if expanded:
                            _, update = draft['private_a2ui']
                            components = update['updateComponents']['components']
                            report['a2ui_source_rows'] = sum(item['id'].startswith('source_') and
                                item['component'] == 'Column' for item in components)
                        report['ack'] = client.request('experimental_owner_ack_draft', message_id=mid,
                            acknowledgement_token=draft['acknowledgement_token'])['status']
                        report['mail_in_history'] = any(SUBJECT in item.content or '11:30' in item.content
                            or 'Synthetic Workshop notice' in item.content or REVISED_DEADLINE in item.content
                            for item in conversation.list(owner_user_id='owner'))
                        if sender_case:
                            components = [item for envelope in draft.get('private_a2ui', [])
                                          for item in envelope.get('updateComponents', {}).get('components', [])]
                            report['sender_source_in_card'] = any(item['id'].startswith('source_')
                                and item.get('text', '').endswith(f'] {SENDER_SUBJECT}') for item in components)
                            report['mail_in_history'] = any(SENDER_SUBJECT in item.content or SENDER_BODY in item.content
                                or SCHOOL_LABEL in item.content for item in conversation.list(owner_user_id='owner'))
        except Exception as exc:
            report.update(status='infrastructure_error', error_type=type(exc).__name__)
        report.update(decisions=provider.decisions, duration=round(time.monotonic()-started, 3),
                      gmail_paths=[path for path, _ in wire.calls],
                      gmail_list_scopes=[request['params']['labelIds'] for path, request in wire.calls
                                         if path == 'messages'],
                      limitation='Disposable owner and Gmail data. Real inference, Unix IPC test client. No native Desktop or live Google verification.')
        if sender_case:
            report['gmail_queries'] = wire.queries
        evidence.write(json.dumps(report, ensure_ascii=False) + '\n')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
