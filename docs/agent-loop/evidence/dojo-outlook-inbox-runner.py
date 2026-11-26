"""Observe owner Outlook Inbox requests with real model decisions and simulated Microsoft."""
import argparse
import json
from pathlib import Path
import tempfile
import time
from urllib.parse import urlsplit

from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider, OpenAICodexProviderConfig
from test_graph_mail import Transport
from test_oauth_lifecycle import expire
from test_owner_calendar import calendar_worker, worker_status
from test_owner_outlook import COUNT, PRIVATE_MARKER, PROMPT, connect_inbox, f, grant_inbox, graph_inbox, outlook_service


GOALS = {
    'authorized': 'Combien de messages non lus ai-je dans ma boîte de réception Hotmail ?',
    'missing-grant': PROMPT,
    'missing-grant-fr': 'Peux-tu vérifier le nombre de messages non lus dans ma boîte de réception Outlook ?',
    'expired-token': 'Could you tell me how many messages are still unread in my Outlook.com inbox?',
    'outside-inbox': 'How many unread messages are in my Outlook Archive folder?',
}
PRIVATE_VALUES = (str(COUNT), PRIVATE_MARKER, 'disposable-access', 'disposable-refresh', 'grant-one')


class ObservedModel:
    supports_image_input = True

    def __init__(self):
        config = OpenAICodexProviderConfig(timeout_seconds=45, require_explicit_model=True)
        self.provider = OpenAICodexProvider(config)
        self.decisions = []
        self.private_value_in_prompt = False

    def generate_json(self, request):
        self.private_value_in_prompt |= any(value in request.prompt for value in PRIVATE_VALUES)
        try:
            result = self.provider.generate_json(request)
        except Exception as exc:
            self.decisions.append({'status': 'error', 'type': type(exc).__name__})
            raise
        value = result.json_value if isinstance(result.json_value, dict) else {}
        self.decisions.append({'kind': value.get('kind'), 'tool_id': value.get('tool_id'),
                               'usage': dict(result.usage or {})})
        return result


def run(case, model):
    goal = GOALS[case]
    started = time.monotonic()
    wire = Transport(graph_inbox()) if case == 'authorized' else Transport()
    provider = ObservedModel()
    report = {'case': case, 'model': model, 'prompt': goal,
              'fixture_unread_messages': COUNT if case == 'authorized' else None}
    try:
        with tempfile.TemporaryDirectory(prefix='dojo-outlook-', dir='/tmp') as directory:
            root = Path(directory)
            fixture = f.__wrapped__(root)
            fixture.accounts.users.set_workspace_root(root / 'workspace')
            connect_inbox(fixture)
            if case not in {'missing-grant', 'missing-grant-fr'}:
                grant_inbox(fixture)
            if case == 'expired-token':
                expire(fixture)
            credentials_before = dict(fixture.backend.data)
            with calendar_worker(fixture, root, wire, provider, model=model, socket=True) as (daemon, _, conversation, client):
                daemon.runtime.owner_outlook_service = outlook_service(fixture, wire)
                message_id = client.request('experimental_owner_ingress', prompt=goal)['message_id']
                status = worker_status(lambda: client.request('experimental_owner_status', message_id=message_id)['status'],
                                       timeout=100)
                report['status'] = status
                if status == 'ready':
                    draft = client.request('experimental_owner_take_draft', message_id=message_id)
                    report['answer'] = draft['message']
                    report['outlook_inbox_status'] = draft.get('outlook_inbox_status')
                    report['outlook_inbox_result'] = draft.get('outlook_inbox_result')
                    report['acknowledgement'] = client.request('experimental_owner_ack_draft', message_id=message_id,
                        acknowledgement_token=draft['acknowledgement_token'])['status']
                    history = '\n'.join(row.content for row in conversation.list(owner_user_id='owner'))
                    report['history_contains_private_value'] = any(value in history for value in PRIVATE_VALUES)
            report['oauth_refreshes'] = fixture.service.clients['microsoft'].refresh_count
            report['credentials_unchanged'] = fixture.backend.data == credentials_before
    except Exception as exc:
        report.update(status='infrastructure_error', error_type=type(exc).__name__)
    report.update(elapsed_seconds=round(time.monotonic() - started, 3), decisions=provider.decisions,
                  private_value_in_model_prompt=provider.private_value_in_prompt,
                  graph_requests=[{'host': urlsplit(url).hostname, 'path': urlsplit(url).path,
                                   'query': urlsplit(url).query} for url, _ in wire.calls])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model')
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--case', choices=tuple(GOALS), action='append')
    arguments = parser.parse_args()
    with arguments.receipt.open('x', encoding='utf-8') as receipt:
        for case in arguments.case or GOALS:
            result = run(case, arguments.model)
            line = json.dumps(result, ensure_ascii=False)
            receipt.write(line + '\n')
            receipt.flush()
            print(line, flush=True)


if __name__ == '__main__':
    main()
