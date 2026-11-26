"""Direct Desktop Gmail search with disposable Inbox messages and authorization."""
from datetime import datetime, timedelta, timezone
import json
import os

import pytest

from alphonse.agent_v2.core.inference.models import InferenceResult
from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail_search import OWNER_GMAIL_SEARCH_TOOL_ID, OwnerGmailSearchService
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable
from test_gmail_adapter import Wire, message
from test_oauth_lifecycle import connect
from test_owner_calendar import calendar_worker, worker_status
from test_owner_gmail import f


GOAL = 'Find the last email about the pottery workshop and summarize it.'
BODY = 'The pottery workshop begins at 11:30. Bring a plain apron.'
SUBJECT = 'Synthetic pottery workshop update'
SUMMARY = 'The workshop starts at 11:30. Bring a plain apron.'


def seed_mail(f, *, content_grant=True):
    connect(f)
    if content_grant:
        f.accounts.save_grant(f.owner, grant_id='content', connection_id='one', expected_revision=0,
            beneficiary_user_id='owner', operations=['mail_content_read'], resources=['INBOX'],
            recipient_user_ids=['owner'], expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())
    else:
        f.accounts.save_grant(f.owner, grant_id='count', connection_id='one', expected_revision=0,
            beneficiary_user_id='owner', operations=['mail_read'], resources=['INBOX'],
            recipient_user_ids=['owner'], expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())


def inbox_fixture():
    item = message('fixture-workshop', labels=('INBOX',), text=BODY)
    item['internalDate'] = str(int((datetime.now(timezone.utc) - timedelta(hours=4)).timestamp() * 1000))
    item['payload']['headers'] = [{'name': 'Subject', 'value': SUBJECT},
                                  {'name': 'From', 'value': 'Workshop <workshop@example.invalid>'}]
    return Wire({'messages': [{'id': item['id']}]}, item)


class ScriptedSummary:
    supports_image_input = False

    def __init__(self, *, request_tool_after_mail=False, question_after_mail=False):
        self.prompts = []
        self.request_tool_after_mail = request_tool_after_mail
        self.question_after_mail = question_after_mail

    def generate_json(self, request):
        self.prompts.append(request.prompt)
        if len(self.prompts) == 1:
            action = {'kind': 'tool', 'tool_id': OWNER_GMAIL_SEARCH_TOOL_ID,
                      'arguments': {'query': 'pottery workshop'}, 'text': None}
        elif self.request_tool_after_mail:
            action = {'kind': 'tool', 'tool_id': 'native.owner_gmail_unread', 'arguments': {}, 'text': None}
        elif self.question_after_mail:
            action = {'kind': 'question', 'tool_id': None, 'arguments': None, 'text': 'Can you grant access?'}
        else:
            action = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': SUMMARY}
        return InferenceResult(json_value=action, raw_response=json.dumps(action),
                               usage={'input_tokens': 100, 'output_tokens': 45})


@pytest.mark.parametrize('case', ['authorized', 'count_only', 'revoked', 'tool_after_mail'])
def test_search_evidence_and_summary_stay_in_private_desktop_path(f, tmp_path, case):
    seed_mail(f, content_grant=case != 'count_only')
    wire = inbox_fixture()
    provider = ScriptedSummary(request_tool_after_mail=case == 'tool_after_mail', question_after_mail=case == 'count_only')
    with calendar_worker(f, tmp_path, wire, provider) as (daemon, queue, conversation, _):
        daemon.runtime.owner_gmail_search_service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
        mid = queue.enqueue_trusted_local_owner(prompt=GOAL, peer_uid=os.getuid())
        status = worker_status(lambda: daemon.owner_message_status(mid))
        assert SUBJECT not in provider.prompts[0]
        if case == 'tool_after_mail':
            assert status == 'failed'
        else:
            assert status == 'ready'
        if case == 'count_only':
            assert wire.calls == []
            draft = daemon.take_owner_desktop_draft(mid)
            assert not draft.get('private_gmail') and SUMMARY not in draft['message']
            return
        assert len(wire.calls) == 2 and wire.calls[0][1]['params']['labelIds'] == 'INBOX'
        assert BODY in provider.prompts[1]
        assert 'native.owner_gmail_unread' not in provider.prompts[1]
        if case == 'revoked':
            f.accounts.revoke_grant(f.owner, grant_id='content', expected_revision=1)
        if case in {'revoked', 'tool_after_mail'}:
            with pytest.raises((OwnerRecoveryUnavailable, ValueError, RuntimeError)):
                daemon.take_owner_desktop_draft(mid)
            return
        draft = daemon.take_owner_desktop_draft(mid)
        assert draft['private_gmail'] is True and '11:30' in draft['message'] and SUBJECT in draft['message']
        create, update = draft['private_a2ui']
        assert create['createSurface']['surfaceId'] == f'private-gmail:{mid}'
        assert create['createSurface']['sendDataModel'] is False
        assert any(c.get('component') == 'Disclosure' for c in update['updateComponents']['components'])
        assert BODY not in json.dumps(draft['private_a2ui'])
        daemon.acknowledge_owner_desktop_draft(mid, draft['acknowledgement_token'])
        assert all('11:30' not in item.content and SUBJECT not in item.content
                   for item in conversation.list(owner_user_id='owner'))
        assert all(marker not in repr(daemon.runtime.core.telemetry_events if hasattr(daemon.runtime.core, 'telemetry_events')
                                      else daemon.runtime.telemetry_events) for marker in (SUBJECT, BODY, SUMMARY))
        assert len(wire.calls) == 2


@pytest.mark.parametrize('revoked_before_pickup', [False, True])
def test_content_grant_covers_only_a_label_read_for_desktop_unread_count(f, tmp_path, revoked_before_pickup):
    seed_mail(f)
    wire = Wire({'id': 'INBOX', 'messagesUnread': 17})
    provider = ScriptedSummary()
    original_generate = provider.generate_json

    def count_decision(request):
        if not provider.prompts:
            provider.prompts.append(request.prompt)
            action = {'kind': 'tool', 'tool_id': 'native.owner_gmail_unread', 'arguments': {}, 'text': None}
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 45})
        return original_generate(request)

    provider.generate_json = count_decision
    with calendar_worker(f, tmp_path, wire, provider) as (daemon, queue, _, _):
        daemon.runtime.owner_gmail_search_service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
        mid = queue.enqueue_trusted_local_owner(prompt='How many unread emails are in my Inbox?', peer_uid=os.getuid())
        assert worker_status(lambda: daemon.owner_message_status(mid)) == 'ready'
        assert [path for path, _ in wire.calls] == ['labels/INBOX']
        if revoked_before_pickup:
            f.accounts.revoke_grant(f.owner, grant_id='content', expected_revision=1)
            with pytest.raises((OwnerRecoveryUnavailable, ValueError, RuntimeError)):
                daemon.take_owner_desktop_draft(mid)
        else:
            draft = daemon.take_owner_desktop_draft(mid)
            assert 'Gmail INBOX unread messages: 17' in draft['message']
            daemon.acknowledge_owner_desktop_draft(mid, draft['acknowledgement_token'])
            assert daemon.owner_message_status(mid) == 'client_acknowledged'
