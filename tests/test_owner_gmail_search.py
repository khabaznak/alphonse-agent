"""Exact Gmail label content grants and projection with disposable provider data."""
from datetime import datetime, timedelta, timezone
import json
import os
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.accounts import GMAIL_CONTENT_OPERATION
from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_gmail_search import (
    OwnerGmailSearchService,
    execute_owner_gmail_search,
    format_gmail_search_observation,
    gmail_search_arguments,
    gmail_search_prompt,
    private_gmail_search_result,
    valid_owner_gmail_search,
    valid_owner_gmail_search_locked,
)
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.integrations.gmail import GmailHTTPError
from alphonse.agent_v2.services.project_check_worker import CheckStopped
from test_gmail_adapter import Wire, encoded
from test_gmail_unread import grant
from test_oauth_lifecycle import connect, expire
from test_owner_gmail import f


def grant_content(f, *, connection_id='one', grant_id='content', resource='INBOX', label_name=None, standing=False):
    expires = None if standing else (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    f.accounts.save_grant(f.owner, grant_id=grant_id, connection_id=connection_id, expected_revision=0,
        beneficiary_user_id=f.owner.actor_user_id, operations=[GMAIL_CONTENT_OPERATION], resources=[resource],
        recipient_user_ids=[f.owner.actor_user_id], expires_at=expires, gmail_label_name=label_name)


def mail(id='retained', *, age_days=1, labels=('INBOX',)):
    received = datetime.now(timezone.utc) - timedelta(days=age_days)
    return {'id': id, 'historyId': '1', 'internalDate': str(int(received.timestamp() * 1000)), 'labelIds': list(labels),
        'payload': {'mimeType': 'multipart/mixed',
            'headers': [{'name': 'Subject', 'value': 'Shipping update'}, {'name': 'From', 'value': 'Parcel shop <p@example.invalid>'}],
            'parts': [{'mimeType': 'text/html', 'body': {'data': encoded('<p>Ships on Friday.</p>'
                '<img src="https://mail-tracker.invalid"><script>HIDDEN_SCRIPT</script>')}},
                {'mimeType': 'text/plain', 'filename': 'invoice.txt',
                 'body': {'attachmentId': 'fixture_attachment', 'data': encoded('PRIVATE_ATTACHMENT')}}]}}


def test_inbox_content_counts_and_content_permissions_are_distinct(f):
    connect(f)
    grant(f, 'one', 'count')
    wire = Wire()
    service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
    result = service.observe('owner', {'query': 'shipping'}, guard=lambda: None)
    assert result == {'status': 'unavailable', 'reason': 'inbox_content_access_required'}
    assert not wire.calls


def test_scoped_inbox_search_filters_label_and_time_and_bounds_model_projection(f):
    connect(f)
    grant_content(f)
    wrong_labels = mail('wronglabel', labels=('SENT',))
    wrong_age = mail('wrongage', age_days=120)
    wire = Wire({'messages': [{'id': 'retained'}, {'id': 'wronglabel'}, {'id': 'wrongage'}], 'nextPageToken': 'PRIVATE_CURSOR'},
                mail(), wrong_labels, wrong_age)
    service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
    result = service.observe('owner', {'query': 'from:p@example.invalid'}, guard=lambda: None)
    assert result['status'] == 'partial' and result['lookback_days'] == 90
    assert private_gmail_search_result(result)
    match, = result['messages']
    assert match['subject'] == 'Shipping update' and match['excerpt'] == 'Ships on Friday.' and match['attachments_omitted'] is True
    assert [path for path, _ in wire.calls] == ['messages', 'messages/retained', 'messages/wronglabel', 'messages/wrongage']
    query = wire.calls[0][1]['params']
    assert query['q'].startswith('(from:p@example.invalid) after:') and ' before:' in query['q']
    assert query['labelIds'] == 'INBOX' and query['maxResults'] == 5 and query['includeSpamTrash'] == 'false'
    assert {attrs['token'] for _, attrs in wire.calls} == {'disposable-access'}
    prompt = gmail_search_prompt(result)
    entry, = json.loads(prompt)['messages']
    assert entry['source_number'] == 1 and entry['excerpt'] == 'Ships on Friday.'
    forbidden = ('disposable-access', 'PRIVATE_CURSOR', 'PRIVATE_ATTACHMENT', 'HIDDEN_SCRIPT', 'mail-tracker',
                 'wronglabel', 'wrongage')
    assert not any(word in prompt or word in json.dumps(result) for word in forbidden)
    assert not any(word in prompt for word in ('connection_id', 'grant_id', 'credential_generation'))
    history = format_gmail_search_observation(result, 'Ships on Friday.')
    answer = format_gmail_search_observation(result, 'Ships on Friday [1].', private_delivery=True)
    assert history == 'Gmail Inbox matches were displayed privately in Desktop.'
    assert 'Shipping update' in answer and '[1]' in answer and 'Search coverage is incomplete' in answer


def test_search_stops_without_provider_access_when_account_ambiguous(f):
    connect(f)
    grant_content(f)
    connect(f, connection='another')
    grant_content(f, connection_id='another', grant_id='another-content')
    wire = Wire()
    service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
    assert service.observe('owner', {'query': ''}, guard=lambda: None) == {
        'status': 'unavailable', 'reason': 'account_ambiguous'}
    assert not wire.calls and f.service.clients['google'].refresh_count == 0


def test_standing_label_grant_renews_expired_oauth_once_and_uses_only_current_credentials(f):
    connect(f)
    grant_content(f, resource='Label_42', label_name='Projects/Pottery', standing=True)
    expire(f)
    old = f.store.get('one')
    wire = Wire({'messages': [{'id': 'retained'}]}, mail(labels=('Label_42',)))
    service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
    unavailable = service.observe('owner', {'query': 'pottery'}, guard=lambda: None)
    assert unavailable == {'status': 'unavailable', 'reason': 'inbox_content_access_required'}
    assert not wire.calls and f.service.clients['google'].refresh_count == 0

    result = service.observe('owner', {'query': 'pottery', 'label': 'Projects/Pottery'}, guard=lambda: None)
    assert result['status'] == 'complete' and len(result['messages']) == 1
    assert f.service.clients['google'].refresh_count == 1
    assert result['authority']['connection_revision'] == old.connection['revision']
    assert result['authority']['credential_generation'] == old.credential_generation + 1
    assert {attrs['token'] for _, attrs in wire.calls} == {'rotated-access'}
    assert 'rotated-' not in gmail_search_prompt(result)


def test_revocation_during_managed_renewal_prevents_gmail_access(f):
    connect(f)
    grant_content(f, resource='Label_42', label_name='Projects/Pottery', standing=True)
    expire(f)
    f.service.clients['google'].on_refresh = lambda: f.accounts.revoke_grant(f.owner, grant_id='content', expected_revision=1)
    wire = Wire()
    service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
    with pytest.raises(CheckStopped, match='authority_changed'):
        service.observe('owner', {'query': '', 'label': 'Projects/Pottery'}, guard=lambda: None)
    assert not wire.calls


@pytest.mark.parametrize('selector', ['work/cafe\u0301', 'Label_42'])
def test_named_or_exact_id_search_selects_granted_label_and_keeps_scope_private(f, selector):
    connect(f)
    grant_content(f, resource='Label_42', label_name='Work/Café')
    grant_content(f, grant_id='inbox')
    wire = Wire({'messages': [{'id': 'retained'}, {'id': 'moved'}, {'id': 'trashed'}]},
                mail(labels=('Label_42',)), mail('moved', labels=('INBOX',)), mail('trashed', labels=('TRASH', 'Label_42')))
    service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
    result = service.observe('owner', {'query': 'shipment', 'label': selector, 'lookback_days': 7}, guard=lambda: None)
    assert result['status'] == 'partial' and len(result['messages']) == 1
    assert wire.calls[0][1]['params']['labelIds'] == 'Label_42'
    assert wire.calls[0][1]['params']['includeSpamTrash'] == 'false'
    prompt = gmail_search_prompt(result)
    private = format_gmail_search_observation(result, 'Ships on Friday [1].', private_delivery=True)
    assert json.loads(prompt)['source'] == 'Gmail label Work/Café'
    assert private.startswith('Gmail label Work/Café · Last 7 days') and 'coverage is incomplete' in private
    assert 'Label_42' not in prompt and 'Label_42' not in private
    assert 'Work/Café' not in format_gmail_search_observation(result)


@pytest.mark.parametrize('resource', ['SPAM', 'TRASH'])
def test_search_includes_spam_and_trash_only_for_explicit_grants(f, resource):
    connect(f)
    grant_content(f, resource=resource, label_name=resource.title())
    wire = Wire({'messages': [{'id': 'retained'}, {'id': 'moved'}]},
                mail(labels=(resource,)), mail('moved', labels=('INBOX',)))
    service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
    result = service.observe('owner', {'query': '', 'label': resource.title()}, guard=lambda: None)
    assert result['status'] == 'partial' and len(result['messages']) == 1
    assert wire.calls[0][1]['params']['labelIds'] == resource
    assert wire.calls[0][1]['params']['includeSpamTrash'] == 'true'


@pytest.mark.parametrize(('label', 'reason'), [('Café', 'label_content_access_required'),
    ('Work/Unknown', 'label_content_access_required'), ('Work/Café', 'account_ambiguous')])
def test_non_exact_or_ambiguous_label_never_falls_back_to_inbox(f, label, reason):
    connect(f)
    grant_content(f, resource='Label_42', label_name='Work/Café')
    grant_content(f, grant_id='inbox')
    grant_content(f, grant_id='another', resource='Label_123', label_name='WORK/CAFÉ')
    wire = Wire()
    service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
    result = service.observe('owner', {'query': '', 'label': label}, guard=lambda: None)
    assert result == {'status': 'unavailable', 'reason': reason}
    assert not wire.calls and f.service.clients['google'].refresh_count == 0


@pytest.mark.parametrize('partial', [False, True])
def test_unreadable_matches_cannot_be_reported_as_no_results(f, partial):
    connect(f)
    grant_content(f)
    wire = Wire({'messages': [{'id': 'removed'}]}, GmailHTTPError(404)) if partial else Wire({'messages': []})
    result = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire).observe('owner', {'query': ''}, guard=lambda: None)
    assert not private_gmail_search_result(result)
    answer = format_gmail_search_observation(result)
    assert ('no matching messages' in answer) is (not partial)
    assert ('coverage is incomplete' in answer) is partial


def test_revocation_during_list_prevents_reading_message_content(f):
    connect(f)
    grant_content(f)
    wire = Wire({'messages': [{'id': 'retained'}]})
    wire.after = lambda: f.accounts.revoke_grant(f.owner, grant_id='content', expected_revision=1)
    service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
    with pytest.raises(CheckStopped, match='authority_changed'):
        service.observe('owner', {'query': 'shipping'}, guard=lambda: None)
    assert [path for path, _ in wire.calls] == ['messages']


@pytest.mark.parametrize('label', [None, 'Projects/Pottery'])
def test_saved_excerpt_revalidation_honors_desktop_claim_and_revoked_grant(f, label):
    connect(f)
    resource = 'Label_42' if label else 'INBOX'
    grant_content(f, resource=resource, label_name=label)
    wire = Wire({'messages': [{'id': 'retained'}]}, mail(labels=(resource,)))
    service = OwnerGmailSearchService(f.accounts, lambda _: f.service, transport=wire)
    queue = SQLiteMessageQueue(f.store.db_path, experimental_claim_fencing=True)
    SQLiteConversationStore(f.store.db_path)
    message_id = queue.enqueue_trusted_local_owner(prompt='Find and summarize my shipping email.', peer_uid=os.getuid())
    acquired = queue.claim_next()
    task = TaskState.from_message(acquired.message, message_id=message_id)
    task.task_id = message_id
    context = SimpleNamespace(task=task, owner_gmail_search_service=service, messages=queue,
                              claim_receipt=acquired.claim_receipt, cancellation_checker=lambda: False)
    arguments = {'query': 'shipping'}
    if label:
        arguments['label'] = label
    observation = execute_owner_gmail_search(arguments, context=context)
    assert valid_owner_gmail_search('owner', arguments, observation, context=context)
    assert not valid_owner_gmail_search('other', arguments, observation, context=context)
    with queue.trusted_local_owner_claim_transaction(acquired.claim_receipt, task) as conn:
        assert valid_owner_gmail_search_locked(conn, 'owner', arguments, observation, context=context)
        assert not valid_owner_gmail_search_locked(conn, 'owner', {**arguments, 'query': 'new'}, observation, context=context)
        assert not valid_owner_gmail_search_locked(conn, 'owner', {**arguments, 'label': 'Other'}, observation, context=context)
        assert not valid_owner_gmail_search_locked(conn, 'other', arguments, observation, context=context)
    f.accounts.revoke_grant(f.owner, grant_id='content', expected_revision=1)
    assert not valid_owner_gmail_search('owner', arguments, observation, context=context)
    with queue.trusted_local_owner_claim_transaction(acquired.claim_receipt, task) as conn:
        assert not valid_owner_gmail_search_locked(conn, 'owner', arguments, observation, context=context)
    assert len(wire.calls) == 2


@pytest.mark.parametrize('arguments', [{'query': 'line\nbreak'}, {'query': '', 'lookback_days': True},
    {'query': 'x' * 257}, {'query': '', 'lookback_days': 366}, {'query': '', 'label': ''},
    {'query': '', 'label': 'label\nname'}, {'query': '', 'label': 'x' * 257}, {'query': '', 'label': None}])
def test_search_arguments_reject_unbounded_or_wrong_scope_inputs(arguments):
    with pytest.raises(ValueError):
        gmail_search_arguments(arguments)
