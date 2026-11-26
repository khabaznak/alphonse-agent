import argparse
from pathlib import Path
import json
import tempfile
import time

from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider, OpenAICodexProviderConfig
from test_owner_calendar import calendar_worker, connect_calendar, f, GOAL, PAGE, TITLE, Wire, worker_status

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('case', choices=('authorized', 'no-grant'))
parser.add_argument('receipt', type=Path)
args = parser.parse_args()

class ObservedProvider:
    supports_image_input = True

    def __init__(self):
        self.provider = OpenAICodexProvider(OpenAICodexProviderConfig(timeout_seconds=45))
        self.decisions = []
        self.title_visible = False

    def generate_json(self, request):
        self.title_visible |= TITLE in request.prompt
        read = self.provider.generate_json(request)
        value = read.json_value
        self.decisions.append(dict(kind=value.get('kind'), tool_id=value.get('tool_id'), arguments=value.get('arguments'),
                                   usage=dict(read.usage or {})))
        return read

with args.receipt.open('x') as evidence:
    started = time.monotonic()
    report = {'case': args.case, 'model': args.model, 'prompt': GOAL, 'status': 'started',
              'scope': 'Real Codex model, fake Google Calendar, disposable owner database and Desktop IPC client.'}
    evidence.write(json.dumps(report, ensure_ascii=False) + '\n')
    evidence.flush()
    provider = ObservedProvider()
    wire = Wire(PAGE)
    try:
        with tempfile.TemporaryDirectory(prefix='dojo-calendar-', dir='/tmp') as directory:
            root = Path(directory)
            fixture = f.__wrapped__(root)
            connect_calendar(fixture, grant=args.case == 'authorized')
            with calendar_worker(fixture, root, wire, provider, model=args.model, socket=True) as (daemon, _, conversation, client):
                mid = client.request('experimental_owner_ingress', prompt=GOAL)['message_id']
                status = worker_status(lambda: client.request('experimental_owner_status', message_id=mid)['status'], timeout=130)
                report['status'] = status
                if status == 'ready':
                    draft = client.request('experimental_owner_take_draft', message_id=mid)
                    report['answer'] = draft['message']
                    report['private_calendar'] = draft.get('private_calendar')
                    report['acknowledgement'] = client.request('experimental_owner_ack_draft', message_id=mid,
                                                               acknowledgement_token=draft['acknowledgement_token'])['status']
                    report['history_has_title'] = any(TITLE in e.content for e in conversation.list(owner_user_id='owner'))
    except Exception as exc:
        report.update(status='infrastructure_error', error_type=type(exc).__name__)
    report.update(elapsed_seconds=round(time.monotonic()-started, 3), title_visible_to_model=provider.title_visible,
                  provider_reads=[{'path': path, 'query': attrs['params']} for path, attrs in wire.calls], decisions=provider.decisions,
                  limitations=['Synthetic provider data. Real Google, rendering in native Desktop, account scope changes and alternate providers were not assessed.',
                               'One attempt per case; model decisions do not prove grounding. IPC acknowledgement is not proof of human receipt.'])
    evidence.write(json.dumps(report, ensure_ascii=False) + '\n')
    print(json.dumps(report, ensure_ascii=False))
