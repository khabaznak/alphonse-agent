"""Owner Desktop Outlook Inbox checks with a fake Microsoft provider and Graph transport."""
from datetime import datetime, timedelta, timezone
import json
import re
import socket
from types import SimpleNamespace
from urllib.parse import parse_qs, urlencode, urlsplit

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult
from alphonse.agent_v2.core.intelligence.simple_loop.owner_outlook import (
    OWNER_OUTLOOK_TOOL_ID, OwnerOutlookService, execute_owner_outlook,
)
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable
from alphonse.agent_v2.core.intelligence.simple_loop.owner_telegram_ingress import enqueue_telegram_owner, verify_telegram_owner_claim
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.ipc import PROTOCOL_VERSION
from alphonse.agent_v2.services.project_check_worker import CheckStopped
from test_graph_mail import Transport, response
from test_oauth_lifecycle import connect, expire
from test_owner_calendar import calendar_worker, worker_status
from test_owner_gmail import f
from test_owner_telegram_ingress import update


CAPABILITY = 'outlook_inbox_unread'
RESOURCE = 'outlook:inbox'
READ_BASIC = 'https://graph.microsoft.com/Mail.ReadBasic'
COUNT = 7149
PRIVATE_MARKER = 'OUTLOOK_PRIVATE_SYNTHETIC_FIELD'
PROMPT = 'How many unread messages are in my Hotmail Inbox?'


def complete_connection(fixture):
    client = fixture.service.clients['microsoft']
    return fixture.service.complete(fixture.owner, client.config.redirect_uri + '?' + urlencode({
        'state': client.state, 'code': 'disposable-code',
    }))


def connect_inbox(fixture, connection_id='one'):
    fixture.service.begin(fixture.owner, connection_id=connection_id, provider='microsoft',
        capabilities=[CAPABILITY], expected_revision=0)
    return complete_connection(fixture)


def grant_values(connection_id='one'):
    return {'grant_id': 'grant-' + connection_id, 'connection_id': connection_id, 'expected_revision': 0,
        'beneficiary_user_id': 'owner', 'operations': [CAPABILITY], 'resources': [RESOURCE],
        'recipient_user_ids': ['owner'], 'expires_at': (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()}


def grant_inbox(fixture, connection_id='one'):
    return fixture.accounts.save_grant(fixture.owner, **grant_values(connection_id))


def graph_inbox(count=COUNT):
    return response({'id': 'disposable-graph-inbox-id', 'unreadItemCount': count, 'extra': PRIVATE_MARKER})


def outlook_service(fixture, wire):
    return OwnerOutlookService(fixture.accounts, lambda _: fixture.service, transport=wire)


def observe(fixture, wire, *, actor='owner', guard=lambda: None):
    return outlook_service(fixture, wire).observe(actor, guard=guard)


def assert_bounded_inbox_request(wire):
    assert len(wire.calls) == 1
    url, kwargs = wire.calls[0]
    endpoint = urlsplit(url)
    assert (endpoint.scheme, endpoint.netloc, endpoint.path) == (
        'https', 'graph.microsoft.com', '/v1.0/me/mailFolders/inbox')
    selected = parse_qs(endpoint.query)
    assert set(selected) == {'$select'}
    assert len(selected['$select']) == 1
    assert set(selected['$select'][0].split(',')) == {'id', 'unreadItemCount'}
    assert kwargs['token'] == 'disposable-access'
    assert kwargs['max_bytes'] > 0
    assert timedelta(0) < kwargs['deadline'] - datetime.now(timezone.utc) < timedelta(minutes=1)


@pytest.mark.parametrize('count', [0, COUNT])
def test_one_narrow_graph_get_returns_verified_inbox_count(f, count):
    health = connect_inbox(f)
    grant_inbox(f)
    assert health['capabilities'] == [CAPABILITY]
    assert set(f.service.clients['microsoft'].scopes) == {'openid', 'offline_access', READ_BASIC}
    wire = Transport(graph_inbox(count))
    result = observe(f, wire)
    assert result['status'] == 'complete' and result['count'] == count
    assert result['resource'] == RESOURCE and result['authority']['connection_id'] == 'one'
    assert PRIVATE_MARKER not in json.dumps(result)
    assert datetime.fromisoformat(result['observed_at']) <= datetime.now(timezone.utc)
    assert_bounded_inbox_request(wire)
    assert f.service.clients['microsoft'].refresh_count == 0


@pytest.mark.parametrize('scenario', ['no_account', 'no_grant', 'legacy_mail_grant', 'wrong_actor', 'grant_expired'])
def test_read_authority_cannot_be_inferred_from_other_accounts_or_capabilities(f, scenario):
    if scenario == 'legacy_mail_grant':
        connect(f, 'microsoft')
        f.accounts.save_grant(f.owner, **{**grant_values(), 'operations': ['mail_read'], 'resources': ['legacy-folder']})
    elif scenario != 'no_account':
        connect_inbox(f)
        if scenario != 'no_grant':
            grant_inbox(f)
    if scenario == 'grant_expired':
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
        with f.store._connect() as db:
            db.execute("UPDATE v2_account_grants SET definition=json_set(definition, '$.expires_at', ?) WHERE grant_id=?",
                       (past, 'grant-one'))

    wire = Transport()
    if scenario == 'wrong_actor':
        result = observe(f, wire, actor='other')
    else:
        result = observe(f, wire)
    assert result['status'] == 'unavailable' and result['reason'] == 'inbox_access_required'
    assert 'count' not in result and not wire.calls


def test_switching_between_connected_accounts_requires_revoking_the_first_grant(f):
    connect_inbox(f)
    grant_inbox(f)
    f.service.clients['microsoft'].identity = 'disposable-second-subject'
    connect_inbox(f, 'two')
    with pytest.raises(ValueError, match='outlook_inbox_grant_exists'):
        grant_inbox(f, 'two')
    wire = Transport(graph_inbox())
    assert observe(f, wire)['authority']['connection_id'] == 'one'
    f.accounts.revoke_grant(f.owner, grant_id='grant-one', expected_revision=1)
    grant_inbox(f, 'two')
    wire = Transport(graph_inbox())
    assert observe(f, wire)['authority']['connection_id'] == 'two'


def test_duplicate_live_grants_fail_closed_even_for_the_same_account(f):
    connect_inbox(f)
    grant_inbox(f)
    with f.store._connect() as db:
        db.execute("""INSERT INTO v2_account_grants (grant_id, connection_id, revision, definition)
            SELECT 'duplicate', connection_id, revision, json_set(definition, '$.grant_id', 'duplicate')
            FROM v2_account_grants WHERE grant_id='grant-one'""")
    wire = Transport()
    assert observe(f, wire)['reason'] == 'account_ambiguous'
    assert not wire.calls


@pytest.mark.parametrize('changes', [
    {'resources': ['inbox']},
    {'resources': [RESOURCE, 'other-folder']},
    {'operations': [CAPABILITY, 'mail_read']},
    {'beneficiary_user_id': 'other'},
    {'recipient_user_ids': ['owner', 'other']},
    {'project_id': 'other-project'},
])
def test_inbox_grant_enforces_exact_scope_at_the_account_boundary(f, changes):
    connect_inbox(f)
    with pytest.raises((PermissionError, ValueError)):
        f.accounts.save_grant(f.owner, **{**grant_values(), **changes})


@pytest.mark.parametrize('page, reason', [
    (response({'error': {'message': PRIVATE_MARKER}}, 401), 'reconnect_required'),
    (response({'error': {'message': PRIVATE_MARKER}}, 403), 'provider_denied'),
    (response({'error': {'message': PRIVATE_MARKER}}, 429, {'Retry-After': '5'}), 'provider_busy'),
    (response({'error': {'message': PRIVATE_MARKER}}, 503), 'provider_busy'),
    (response({'id': 'disposable-graph-inbox-id'}), 'read_unavailable'),
    (response({'id': '  ', 'unreadItemCount': COUNT}), 'read_unavailable'),
    (graph_inbox(True), 'read_unavailable'),
    (graph_inbox(-1), 'read_unavailable'),
])
def test_provider_failures_never_become_zero_or_leak_provider_details(f, page, reason):
    connect_inbox(f)
    grant_inbox(f)
    wire = Transport(page)
    result = observe(f, wire)
    assert result['status'] == 'unavailable' and result['reason'] == reason
    assert 'count' not in result and PRIVATE_MARKER not in json.dumps(result)
    assert_bounded_inbox_request(wire)
    assert f.store.get('one').connection['status'] == ('reauth_required' if reason == 'reconnect_required' else 'connected')


def test_expired_access_token_makes_no_graph_or_oauth_request(f):
    connect_inbox(f)
    grant_inbox(f)
    expire(f)
    wire = Transport()
    result = observe(f, wire)
    assert result['status'] == 'unavailable' and result['reason'] == 'refresh_required'
    assert 'count' not in result and not wire.calls and f.service.clients['microsoft'].refresh_count == 0


@pytest.mark.parametrize('during_graph', [False, True])
def test_cancellation_fences_inbox_observation(f, during_graph):
    connect_inbox(f)
    grant_inbox(f)
    wire = Transport(graph_inbox())
    cancelled = not during_graph

    def cancel():
        nonlocal cancelled
        cancelled = True

    def guard():
        if cancelled:
            raise CheckStopped('cancelled', 'fixture_cancelled')

    wire.after = cancel
    with pytest.raises(CheckStopped, match='fixture_cancelled'):
        observe(f, wire, guard=guard)
    assert len(wire.calls) == int(during_graph)


def test_grant_revoked_while_graph_is_in_flight_cannot_release_count(f):
    connect_inbox(f)
    grant_inbox(f)
    wire = Transport(graph_inbox())
    wire.after = lambda: f.accounts.revoke_grant(f.owner, grant_id='grant-one', expected_revision=1)
    with pytest.raises((CheckStopped, PermissionError)):
        observe(f, wire)
    assert_bounded_inbox_request(wire)


def test_even_a_verified_owner_telegram_claim_cannot_use_the_desktop_outlook_tool(f):
    connect_inbox(f)
    grant_inbox(f)
    users = f.accounts.users
    users.bind_address(user_id='owner', integration_id='telegram-test', provider_key='telegram', provider_user_id='123')
    f.store.upsert(integration_id='telegram-test', provider_key='telegram', display_name='Telegram', enabled=True)
    SQLiteConversationStore(users.db_path)
    queue = SQLiteMessageQueue(users.db_path, experimental_claim_fencing=True)
    mid = enqueue_telegram_owner(queue, integration_id='telegram-test', update=update(PROMPT))
    claimed = queue.claim_next()
    task = TaskState.from_message(claimed.message, message_id=mid)
    task.task_id = mid
    assert verify_telegram_owner_claim(queue, claimed.claim_receipt, task)
    wire = Transport()
    context = SimpleNamespace(messages=queue, claim_receipt=claimed.claim_receipt, task=task,
                              owner_outlook_service=outlook_service(f, wire), cancellation_checker=lambda: False)
    with pytest.raises(CheckStopped, match='owner_outlook_claim_unavailable'):
        execute_owner_outlook({}, context=context)
    assert not wire.calls


class ScriptedOutlookModel:
    supports_image_input = False

    def __init__(self, reply='There are 9999 unread messages.'):
        self.requests = []
        self.reply = reply

    def generate_json(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            assert OWNER_OUTLOOK_TOOL_ID in request.prompt
            action = {'kind': 'tool', 'tool_id': OWNER_OUTLOOK_TOOL_ID, 'arguments': {}, 'text': None}
        else:
            action = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': self.reply}
        return InferenceResult(json_value=action, raw_response=json.dumps(action),
                               usage={'input_tokens': 100, 'output_tokens': 20})


def ipc_request(**params):
    return {'version': PROTOCOL_VERSION, 'params': params}


@pytest.mark.parametrize('scenario', ['authorized', 'restored', 'no_grant', 'revoked_before_pickup'])
def test_owner_desktop_handlers_and_worker_pick_up_only_current_inbox_observations(f, tmp_path, scenario):
    connect_inbox(f)
    if scenario != 'no_grant':
        grant_inbox(f)
    wire = Transport(graph_inbox())
    model = ScriptedOutlookModel()
    with calendar_worker(f, tmp_path, wire, model) as (daemon, _, conversation, _):
        daemon.runtime.owner_outlook_service = outlook_service(f, wire)
        # Exercise the real authenticated handlers without binding a listening socket.
        server, client = socket.socketpair()
        with server, client:
            message_id = daemon.ipc._owner_ingress(server, ipc_request(prompt=PROMPT))['message_id']
            assert worker_status(lambda: daemon.owner_message_status(message_id)) == 'ready'
            if scenario == 'restored':
                daemon._lock_file = object()
                daemon._owner_checkpoints.pop(message_id)
                daemon._owner_statuses.pop(message_id)
                assert daemon._restore_owner_ready(message_id)
            if scenario == 'revoked_before_pickup':
                f.accounts.revoke_grant(f.owner, grant_id='grant-one', expected_revision=1)
                with pytest.raises((OwnerRecoveryUnavailable, ValueError, RuntimeError)):
                    daemon.ipc._owner_take_draft(server, ipc_request(message_id=message_id))
                assert_bounded_inbox_request(wire)
                return
            draft = daemon.ipc._owner_take_draft(server, ipc_request(message_id=message_id))
            text = draft['message']
            assert 'outlook' in text.lower() or 'hotmail' in text.lower()
            assert 'inbox' in text.lower() and '9999' not in text and PRIVATE_MARKER not in text
            if scenario in {'authorized', 'restored'}:
                assert str(COUNT) in text.replace(',', '')
                assert_bounded_inbox_request(wire)
            else:
                assert 'grant' in text.lower() and str(COUNT) not in text and not wire.calls
            acknowledged = daemon.ipc._owner_ack_draft(server, ipc_request(
                message_id=message_id, acknowledgement_token=draft['acknowledgement_token']))
            assert acknowledged['status'] == 'client_acknowledged'

        prompts = '\n'.join(request.prompt for request in model.requests)
        assert not re.search(rf'(?<![\w-]){COUNT}(?![\w-])', prompts)
        assert all(value not in prompts for value in (PRIVATE_MARKER, 'disposable-access', 'grant-one'))
        history = '\n'.join(item.content for item in conversation.list(owner_user_id='owner'))
        assert PROMPT in history and str(COUNT) not in history and PRIVATE_MARKER not in history


@pytest.mark.parametrize('has_grant, model_answer, expected_locale', [
    (True, 'fr-FR', 'fr-FR'),
    (True, 'Gmail Archive has {count} unread messages. I deleted them at {time}.', None),
    (False, 'Messages deleted successfully', None),
])
def test_private_pickup_uses_typed_facts_and_treats_model_language_as_presentation_hint(
        f, tmp_path, has_grant, model_answer, expected_locale):
    connect_inbox(f)
    f.accounts.users.set_timezone('America/Mexico_City', f.owner.actor_user_id)
    if has_grant:
        grant_inbox(f)
    wire = Transport(graph_inbox())
    model = ScriptedOutlookModel(reply=model_answer)
    with calendar_worker(f, tmp_path, wire, model) as (daemon, _, _, _):
        daemon.runtime.owner_outlook_service = outlook_service(f, wire)
        server, client = socket.socketpair()
        with server, client:
            request = ipc_request(prompt='Combien de messages non lus ai-je dans ma boîte de réception Hotmail ?')
            message_id = daemon.ipc._owner_ingress(server, request)['message_id']
            assert worker_status(lambda: daemon.owner_message_status(message_id)) == 'ready'
            draft = daemon.ipc._owner_take_draft(server, ipc_request(message_id=message_id))
            assert model_answer not in draft['message']
            assert 'Outlook' in draft['message'] and 'Inbox' in draft['message']
            assert not {'committed_effect', 'private_a2ui', 'private_gmail', 'private_calendar'} & draft.keys()
            assert draft['outlook_inbox_status'] == ('complete' if has_grant else 'inbox_access_required')
            if has_grant:
                result = draft['outlook_inbox_result']
                assert set(result) == {'count', 'observed_at', 'timezone'} | ({'locale'} if expected_locale else set())
                assert result['count'] == COUNT
                assert datetime.fromisoformat(result['observed_at']).utcoffset() is not None
                assert result['timezone'] == 'America/Mexico_City'
                assert result.get('locale') == expected_locale
            else:
                assert 'outlook_inbox_result' not in draft and 'grant Inbox access' in draft['message']
    assert len(wire.calls) == int(has_grant)
    assert all(not re.search(rf'(?<![\w-]){COUNT}(?![\w-])', request.prompt) for request in model.requests)
