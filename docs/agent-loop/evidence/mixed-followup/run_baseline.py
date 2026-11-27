"""Disposable owner-worker baseline for mixed notes and delegated work."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
from tempfile import TemporaryDirectory
import time
from zoneinfo import ZoneInfo

from alphonse.agent_v2.core.inference.models import InferenceResult
from alphonse.agent_v2.core.inference.openai_codex import OpenAICodexProvider, OpenAICodexProviderConfig
from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail_search import OwnerGmailSearchService
from alphonse.agent_v2.core.intelligence.simple_loop.owner_work_log import owner_work_log
from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders
from codex_worker import CodexProposalReceiver
from dojo_mixed_followup_fixture import DisposableReceiver, attach_disposable_receiver
from test_gmail_adapter import message
from test_owner_calendar import calendar_worker, worker_status
from test_owner_gmail_search_conversation import f, seed_mail


FIRST = 'What in my Gmail inbox needs my attention? Give me a short digest of the important emails.'
PRIVATE_TITLE = 'Blue Lantern Workshop'
MAIL_MARKER = 'PRINT-NOTE-2741'
REPO = 'print-preview-fixture'
REMINDER = 'run the test print'
MAIL = {
    'hosting': ('Hosting renewal due soon', 'Your hosting renewal invoice is due. Check payment status.'),
    'workshop': ('Workshop naming decision', f'The provisional event name is {PRIVATE_TITLE}. It is private until the unveiling.'),
    'preview': ('Print preview cuts off the subtitle',
                f'The caption ends midway through the subtitle in print preview. Internal mail reference {MAIL_MARKER}.'),
}


class SyntheticMailbox:
    def __init__(self, now):
        self.calls = []
        self.items = {}
        for index, (key, (subject, body)) in enumerate(MAIL.items()):
            item = message(key, labels=('INBOX', 'UNREAD', 'IMPORTANT'), text=body)
            item['internalDate'] = str(int((now - timedelta(hours=index + 2)).timestamp() * 1000))
            item['payload']['headers'] = [{'name': 'Subject', 'value': subject},
                {'name': 'From', 'value': 'Synthetic sender <fixture@example.invalid>'}]
            self.items[key] = item

    def get(self, path, **kwargs):
        params = kwargs.get('params', {})
        self.calls.append({'path': path, 'params': params})
        if path == 'messages':
            scoped = re.fullmatch(r'(.*?)after:(\d+) before:(\d+)', params['q'])
            if not scoped:
                raise ValueError('fixture_expected_scoped_query')
            query, start, end = scoped.groups()
            query = query.strip().removeprefix('(').removesuffix(')').strip()
            terms = re.findall(r'"([^"]+)"|([^\s{}()]+)', query)
            alternatives = [[]]
            for phrase, word in terms:
                term = (phrase or word).lower()
                if term == 'or':
                    alternatives.append([])
                else:
                    alternatives[-1].append(term)

            def matches(key):
                subject, body = MAIL[key]
                content = f'{subject} {body}'.lower()
                return any(all(term in {'is:important', 'is:unread', 'in:inbox'}
                               or (term.removeprefix('subject:') in (subject.lower() if term.startswith('subject:') else content))
                               for term in group) for group in alternatives)

            result = {'messages': [{'id': key} for key, item in self.items.items()
                if params['labelIds'] in item['labelIds']
                and int(start) * 1000 < int(item['internalDate']) < int(end) * 1000 and matches(key)]}
        else:
            result = self.items[path.removeprefix('messages/')]
        return result, len(json.dumps(result).encode())


class ObservedProvider:
    supports_image_input = False

    def __init__(self, scripted, *, handoff=False, cli='codex'):
        self.live = None if scripted else OpenAICodexProvider(OpenAICodexProviderConfig(cli_bin=cli, timeout_seconds=50))
        self.turn = None
        self.decisions = []
        self.handoff = handoff
        self.followup_prompt = ''
        self.due_at = ''

    def generate_json(self, request):
        catalog_text = request.prompt.split('Alphonse has the following host tools available: ', 1)[1]
        tools, _ = json.JSONDecoder().raw_decode(catalog_text)
        row = {'turn': self.turn, 'tools': [tool['tool_id'] for tool in tools],
               'prior_gmail_receipt_visible': 'Gmail Inbox matches were displayed privately in Desktop.' in request.prompt,
               'mail_text_marker_visible': MAIL_MARKER in request.prompt,
               'private_note_marker_visible': PRIVATE_TITLE in request.prompt}
        self.decisions.append(row)
        try:
            result = self.live.generate_json(request) if self.live else self.scripted(row)
        except Exception as exc:
            row.update(error_type=type(exc).__name__, error_code=error_code(exc))
            raise
        row.update(decision=result.json_value, usage=dict(result.usage or {}))
        return result

    def scripted(self, row):
        if self.handoff and self.turn == 2:
            actions = (
                ('native.owner_clock', {}),
                ('native.owner_reminder_create', {
                    'message': REMINDER, 'due_at': self.due_at, 'request_quote': self.followup_prompt,
                }),
                ('native.owner_disposable_handoff', {'target': REPO, 'worker': 'codex', 'operation': 'propose_patch'}),
            )
            call_index = sum(decision['turn'] == 2 for decision in self.decisions) - 1
            if call_index < len(actions):
                tool_id, arguments = actions[call_index]
                assert tool_id in row['tools'], f'Missing scripted Dojo capability {tool_id}'
                decision = {'kind': 'tool', 'tool_id': tool_id, 'arguments': arguments, 'text': None}
            else:
                decision = {'kind': 'answer', 'tool_id': None, 'arguments': None,
                            'text': 'The reminder was blocked. Codex accepted the handoff. A result is pending.'}
        elif 'native.owner_gmail_search' in row['tools']:
            decision = {'kind': 'tool', 'tool_id': 'native.owner_gmail_search',
                        'arguments': {'query': '' if self.turn == 1 else 'subtitle'}, 'text': None}
        else:
            response = ('Check the hosting renewal [1], keep the workshop name private [2], and look into the clipping report [3].'
                        if self.turn == 1 else 'The email reports subtitle clipping [1].')
            decision = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': response}
        return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                               usage={'input_tokens': 100, 'output_tokens': 45})


def error_code(exc):
    text = str(exc)
    if re.fullmatch(r'[a-z][a-z0-9_]{0,100}', text):
        return text
    return f'errno_{exc.errno}' if isinstance(exc, OSError) else None


def serial_turn(daemon, queue, client, prompt):
    start = time.monotonic()
    if client:
        message_id = client.request('experimental_owner_ingress', prompt=prompt)['message_id']
        status = lambda: client.request('experimental_owner_status', message_id=message_id)['status']
    else:
        message_id = queue.enqueue_trusted_local_owner(prompt=prompt, peer_uid=os.getuid())
        status = lambda: daemon.owner_message_status(message_id)
    outcome = worker_status(status, timeout=125)
    row = {'prompt': prompt, 'message_id': message_id, 'status': outcome}
    if outcome == 'ready':
        draft = client.request('experimental_owner_take_draft', message_id=message_id) if client else daemon.take_owner_desktop_draft(message_id)
        row.update(answer=draft['message'], answer_kind=draft.get('kind'), private_gmail=draft.get('private_gmail', False),
                   private_surface=draft.get('private_a2ui'), committed_effect=draft.get('committed_effect'))
        token = draft['acknowledgement_token']
        if client:
            ack = client.request('experimental_owner_ack_draft', message_id=message_id, acknowledgement_token=token)['status']
        else:
            daemon.acknowledge_owner_desktop_draft(message_id, token)
            ack = status()
        row['ack'] = ack
    elif client and outcome != 'timeout':
        row['failure_code'] = client.request('experimental_owner_status', message_id=message_id).get('failure_code')
    try:
        row['work_log'] = (client.request('experimental_owner_work_log', message_id=message_id)
                           if client else owner_work_log(queue, message_id))
    except Exception as exc:
        row['work_log_error'] = error_code(exc) or type(exc).__name__
    row['elapsed_seconds'] = round(time.monotonic() - start, 3)
    return row


def snapshot(repo):
    return {str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(repo.rglob('*')) if path.is_file() and '.git' not in path.parts}


def run(args):
    provider = ObservedProvider(args.scripted, handoff=args.handoff, cli=args.parent_cli)
    now = datetime.now(timezone.utc)
    zone = ZoneInfo('America/Mexico_City')
    local_now = now.astimezone(zone)
    thursday = local_now + timedelta(days=(3 - local_now.weekday()) % 7 or 7)
    due = thursday.replace(hour=15, minute=0, second=0, microsecond=0).isoformat()
    notes = ('The hosting invoice is already paid. The workshop name should stay between us. We print Friday, '
             'so the subtitle problem is the priority.')
    handoff_request = f'Ask Codex to check the print-preview complaint in {REPO} and propose a patch if it reproduces.'
    reminder_request = f'Please remind me to {REMINDER} at {due}.'
    requests = (reminder_request, handoff_request) if args.reminder_first else (handoff_request, reminder_request)
    second = ' '.join((notes, *requests))
    provider.followup_prompt, provider.due_at = second, due
    wire = SyntheticMailbox(now)
    report = {'case': 'mixed-notes-and-delegation-handoff' if args.handoff else 'mixed-notes-and-delegation-baseline',
              'model': 'scripted' if args.scripted else args.model,
              'reminder_mentioned_first': args.reminder_first,
              'transport': 'direct owner harness' if args.direct else 'Unix IPC test client',
              'turns': [], 'second_prompt': second, 'reminder_expected': {'message': REMINDER, 'due_at': due},
              'limitations': ['Synthetic inbox, owner, and repository; no live Google traffic.',
                              ('The separate live Codex CLI receives only the pinned synthetic file snapshot and fixed brief. '
                               'It proposes a change without applying it to the original. An independent local fixture test verifies the proposal.'
                               if args.real_worker else
                               'A separate disposable receiving process stands in for the named Codex worker. '
                               'Acceptance is independently recorded; review, proposal generation and real Codex execution are unassessed.'
                               if args.handoff else
                               'No Codex worker adapter, repository access or dispatch enabled for the owner loop.'),
                              'This does not render installed Desktop or establish human receipt.',
                              'The standalone reminder grammar currently refuses multi-clause requests.',
                              'One exploratory attempt does not establish model reliability.']}
    start = time.monotonic()
    receiver = None
    try:
        with TemporaryDirectory(prefix='dojo-mixed-', dir='/tmp') as folder:
            root = Path(folder)
            repo = root / REPO
            repo.mkdir()
            subprocess.run(['git', 'init', '--quiet', str(repo)], check=True, capture_output=True)
            (repo / 'preview.py').write_text('def render_subtitle(text):\n    return text[:16]\n', encoding='utf-8')
            subtitle = 'Illustration proofs for Friday'
            (repo / 'check_preview.py').write_text(f'from preview import render_subtitle\nassert render_subtitle({subtitle!r}) == {subtitle!r}\n', encoding='utf-8')
            verify = lambda: subprocess.run([os.sys.executable, '-B', str(repo / 'check_preview.py')], cwd=repo,
                                             capture_output=True, check=False).returncode
            report['repo_before'] = snapshot(repo)
            report['repo_test_before'] = verify()
            configure_runtime = None
            if args.handoff:
                subprocess.run(['git', '-C', str(repo), 'add', 'preview.py', 'check_preview.py'],
                               check=True, capture_output=True)
                subprocess.run(['git', '-C', str(repo), '-c', 'user.name=Dojo fixture',
                                '-c', 'user.email=fixture@example.invalid', '-c', 'commit.gpgsign=false',
                                '-c', 'core.hooksPath=/dev/null', 'commit', '-qm', 'Seed disposable preview'],
                               check=True, capture_output=True)
                revision = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
                report['authorized_revision'] = revision
                receiver = (CodexProposalReceiver(root / 'receiver', repo=repo, target=REPO, revision=revision,
                                                   model=args.worker_model, cli=args.worker_cli)
                            if args.real_worker else DisposableReceiver(root / 'receiver'))

                def configure_runtime(runtime):
                    attach_disposable_receiver(runtime, actor='owner', target=REPO, revision=revision,
                                               prompt=second, receiver=receiver)
            fixture = f.__wrapped__(root)
            fixture.accounts.users.set_timezone('America/Mexico_City', 'owner')
            seed_mail(fixture)
            with calendar_worker(fixture, root, wire, provider, model=args.model, socket=not args.direct,
                                 configure_runtime=configure_runtime) as (
                    daemon, queue, conversation, client):
                daemon.runtime.owner_gmail_search_service = OwnerGmailSearchService(
                    fixture.accounts, lambda _: fixture.service, transport=wire)
                for turn, prompt in enumerate((FIRST, second), 1):
                    provider.turn = turn
                    row = serial_turn(daemon, queue, client, prompt)
                    report['turns'].append(row)
                    if turn == 1 and row.get('ack') != 'client_acknowledged':
                        break
                report['reminders_after'] = OwnerDesktopReminders(fixture.accounts.users.db_path).list_open('owner')
                report['history'] = [{'role': item.role, 'content': item.content} for item in conversation.list(owner_user_id='owner')]
                with sqlite3.connect(fixture.accounts.users.db_path) as conn:
                    exists = conn.execute("select 1 from sqlite_schema where type='table' and name='coding_jobs'").fetchone()
                    report['coding_job_rows'] = conn.execute('select count(*) from coding_jobs').fetchone()[0] if exists else 0
                    report['coding_jobs_table_present'] = bool(exists)
                if receiver is not None:
                    report['receiver_receipts'] = receiver.receipts()
            report['repo_after'] = snapshot(repo)
            report['repo_test_after'] = verify()
            report['status'] = 'followup_observed' if len(report['turns']) == 2 else 'first_turn_blocked'
    except Exception as exc:
        report.update(status='infrastructure_error', error_type=type(exc).__name__, error_code=error_code(exc))
    if args.real_worker and isinstance(receiver, CodexProposalReceiver):
        report['codex_worker'] = receiver.evidence
    report.update(model_decisions=provider.decisions, gmail_calls=wire.calls,
                  elapsed_seconds=round(time.monotonic() - start, 3))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('--model', default='gpt-6.1-terra')
    parser.add_argument('--parent-cli', default='codex', help='Optional explicit local Codex CLI for the parent model.')
    parser.add_argument('--direct', action='store_true')
    parser.add_argument('--scripted', action='store_true')
    parser.add_argument('--handoff', action='store_true', help='Enable only the bound disposable receiving process.')
    parser.add_argument('--real-worker', action='store_true', help='Ask a separate native Codex CLI to propose a patch.')
    parser.add_argument('--worker-model', default='gpt-6-luna')
    parser.add_argument('--worker-cli', type=Path, help='Path to the native macOS Codex executable.')
    parser.add_argument('--reminder-first', action='store_true', help='Place the reminder request before the handoff.')
    args = parser.parse_args()
    if args.real_worker and (not args.handoff or args.worker_cli is None):
        parser.error('--real-worker requires --handoff and --worker-cli')
    with args.receipt.open('x', encoding='utf-8') as journal:
        result = run(args)
        json.dump(result, journal, ensure_ascii=False, indent=2)
        journal.write('\n')
    print(json.dumps({'status': result['status'], 'receipt': str(args.receipt),
                      'elapsed_seconds': result['elapsed_seconds'], 'turns': len(result['turns']),
                      'error_type': result.get('error_type'), 'error_code': result.get('error_code')}))
