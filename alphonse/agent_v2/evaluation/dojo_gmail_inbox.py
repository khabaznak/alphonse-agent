"""Observe Gmail Inbox requests through the owner daemon with a fake provider."""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from alphonse.agent_v2.accounts import AccountService, AccountSession
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail import OwnerGmailService
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord, build_inference_router_from_settings
from alphonse.agent_v2.integrations.credentials import CredentialVault
from alphonse.agent_v2.integrations.gmail import SCOPE
from alphonse.agent_v2.integrations.oauth import ClientConfig, OAuthService
from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


CASES = ('missing-grant', 'authorized', 'expired-token')
GOALS = {
    'missing-grant': 'How many unread messages are in my Gmail inbox right now?',
    'authorized': 'Peux-tu me dire combien de messages non lus sont dans ma boîte de réception Gmail ?',
    'expired-token': 'Could you check how many unread messages I have in my Gmail inbox?',
}
PRIVATE_MARKERS = ('fake-oauth-access', 'fake-oauth-refresh', 'FAKE_GMAIL_UNUSED_FIELD')


class MemoryBackend:
    def __init__(self):
        self.values = {}

    def put(self, reference, payload):
        self.values[reference] = payload

    def get(self, reference):
        return self.values[reference]

    def delete(self, reference):
        self.values.pop(reference, None)


class FakeGoogle:
    def __init__(self):
        self.gmail_reads = []
        self.oauth_calls = []
        self.label = {'id': 'INBOX', 'messagesUnread': 7, 'unused': PRIVATE_MARKERS[2]}

    def get(self, path, *, params, token, deadline, max_bytes):
        self.gmail_reads.append({'path': path, 'params': params, 'token_matches': token == PRIVATE_MARKERS[0]})
        if path != 'labels/INBOX' or params or token != PRIVATE_MARKERS[0]:
            raise AssertionError('unexpected_fake_gmail_request')
        return self.label.copy(), len(json.dumps(self.label).encode())

    def refresh(self, *_):
        self.oauth_calls.append('refresh')
        raise AssertionError('unexpected_fixture_oauth_refresh')


def seed(accounts, session, case):
    account = accounts.create_connection(session, connection_id='dojo-gmail', provider_key='google',
        provider_subject='invented-subject', display_name='Synthetic Gmail')
    seconds = -1 if case == 'expired-token' else 3600
    accounts.integrations.install_credentials('dojo-gmail',
        {'access_token': PRIVATE_MARKERS[0], 'refresh_token': PRIVATE_MARKERS[1],
         'expires_at': str(time.time() + seconds)}, expected_generation=0)
    accounts.save_connection(session, connection_id='dojo-gmail', expected_revision=account['revision'],
        provider_subject='invented-subject', capabilities=['mail_read'], granted_scopes=[SCOPE])
    if case != 'missing-grant':
        accounts.save_grant(session, grant_id='dojo-only-inbox', connection_id='dojo-gmail', expected_revision=0,
            beneficiary_user_id=session.actor_user_id, operations=['mail_read'], resources=['INBOX'],
            recipient_user_ids=[session.actor_user_id],
            expires_at=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat())


def run(router, case, *, timeout=100):
    goal = GOALS[case]
    started = time.monotonic()
    decisions = []
    prompt_markers_present = False
    original_generate = router.generate_json

    def observe_decision(request):
        nonlocal prompt_markers_present
        prompt_markers_present |= any(marker in request.prompt for marker in PRIVATE_MARKERS)
        try:
            result = original_generate(request)
        except Exception as exc:
            decisions.append({'status': 'provider_error', 'error_type': type(exc).__name__})
            raise
        value = result.json_value if isinstance(result.json_value, dict) else {}
        decisions.append({'status': 'returned', 'kind': value.get('kind'), 'tool_id': value.get('tool_id'),
                          'usage': dict(result.usage or {})})
        return result

    router.generate_json = observe_decision
    try:
        with tempfile.TemporaryDirectory(prefix='dojo-gmail-', dir='/tmp') as directory:
            root = Path(directory)
            db = root / 'owner.sqlite'
            users = V2UserStore(db)
            owner = users.onboard(display_name='Synthetic owner', users_root=root / 'users')
            users.set_workspace_root(root / 'workspace')
            backend, google = MemoryBackend(), FakeGoogle()
            store = SQLiteIntegrationStore(db, credential_vault=CredentialVault(backend))
            accounts = AccountService(store, users)
            seed(accounts, AccountSession(owner.user_id), case)
            credentials_before = dict(backend.values)
            oauth = OAuthService(accounts, [ClientConfig('google', 'fake-app', 'http://127.0.0.1:54321/callback')],
                                 client_factory=lambda _: google)
            runtime = build_runtime_host(user_store=users, messages=SQLiteMessageQueue(db, experimental_claim_fencing=True),
                question_store=SQLiteQuestionStore(db), conversation_store=SQLiteConversationStore(db),
                integration_store=store, outbox=SQLiteOutboundStore(db), inference=router,
                experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
            runtime.owner_gmail_service = OwnerGmailService(accounts, lambda _: oauth, transport=google)
            daemon = V2Daemon(runtime, daemon_id='dojo-gmail-inbox', experimental_owner_only=True)
            socket_path = root / 'owner.sock'
            daemon.ipc = V2DaemonServer(daemon, socket_path=socket_path, experimental_owner_ingress=True)
            with patch('alphonse.agent_v2.daemon.default_socket_path', return_value=socket_path):
                daemon.start()
            try:
                client = V2DaemonClient(socket_path)
                message_id = client.request('experimental_owner_ingress', prompt=goal)['message_id']
                deadline = time.monotonic() + timeout
                while time.monotonic() < deadline:
                    status = client.request('experimental_owner_status', message_id=message_id)
                    if status['status'] not in {'pending', 'processing'}:
                        break
                    time.sleep(.1)
                else:
                    status = {'status': 'timeout'}
                answer, kind, acknowledged = None, None, None
                if status['status'] == 'ready':
                    draft = client.request('experimental_owner_take_draft', message_id=message_id)
                    answer, kind = draft['message'], draft['kind']
                    acknowledged = client.request('experimental_owner_ack_draft', message_id=message_id,
                        acknowledgement_token=draft['acknowledgement_token']).get('status')
                return {'case': case, 'goal': goal, 'ipc': status, 'answer': answer, 'kind': kind,
                        'client_acknowledgement': acknowledged, 'decisions': decisions,
                        'elapsed_seconds': round(time.monotonic() - started, 3),
                        'fixture_unread_messages': google.label['messagesUnread'] if case == 'authorized' else None,
                        'fake_gmail_calls': google.gmail_reads, 'fake_oauth_calls': google.oauth_calls,
                        'credentials_unchanged': backend.values == credentials_before,
                        'private_markers_absent': not prompt_markers_present and not any(
                            marker in (answer or '') for marker in PRIVATE_MARKERS),
                        'response_review': 'requires_manual_review',
                        'limitations': ['Disposable Google fixture and Unix socket client. No live Google account or installed Desktop UI.',
                            'Host observations do not cover inference-provider behavior outside Alphonse.',
                            'Client acknowledgement does not establish human receipt.',
                            'The run does not assess other mail capabilities, languages generally or reliability over repeated attempts.']}
            finally:
                daemon.stop()
    finally:
        router.generate_json = original_generate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--case', required=True, choices=CASES)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    with args.journal.open('x', encoding='utf-8') as journal:
        report = {'model': args.model, 'case': args.case, 'started_at': datetime.now(timezone.utc).isoformat()}
        try:
            router = build_inference_router_from_settings(InferenceSettingsRecord(model_id=args.model))
            router.provider.config = replace(router.provider.config, timeout_seconds=40.0)
            report.update(run(router, args.case))
        except Exception as exc:
            report.update(status='infrastructure_error', error_type=type(exc).__name__)
            raise
        finally:
            json.dump(report, journal, ensure_ascii=False, indent=2, sort_keys=True)
            journal.write('\n')


if __name__ == '__main__':
    main()
