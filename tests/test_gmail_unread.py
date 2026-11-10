"""Synthetic Gmail count and access tests; never contact an account/provider."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.accounts import AccountSession
from alphonse.agent_v2.integrations.gmail import ALL_MAIL, GmailHTTPError
from alphonse.agent_v2.integrations.gmail_unread import GmailUnreadService
from alphonse.agent_v2.services.project_check_worker import CheckStopped
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry.native.gmail_unread import build_gmail_unread_tool_definition
from test_gmail_adapter import Wire
from test_oauth_lifecycle import f, connect


def grant(f, connection, grant_id, *, actor='owner', resources=('INBOX',), project_id=None):
    f.accounts.save_grant(f.owner, grant_id=grant_id, connection_id=connection, expected_revision=0,
        beneficiary_user_id=actor, operations=['mail_read'], resources=list(resources), recipient_user_ids=[],
        expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat(), project_id=project_id)


def service(f, *response):
    wire = Wire(*response)
    return GmailUnreadService(f.accounts, f.service, transport=wire), wire


@pytest.mark.parametrize('unit,field,count', [('messages', 'messagesUnread', 0), ('threads', 'threadsUnread', 27)])
def test_scoped_count_is_one_label_request_without_message_content(f, unit, field, count):
    connect(f); grant(f, 'one', 'grant')
    svc, wire = service(f, {'id': 'INBOX', field: count, 'name': 'not returned', 'unexpected': 'private'})
    result = svc.count(f.owner, unit=unit)
    assert result['status'] == 'complete' and result['count'] == count
    assert result['label_id'] == 'INBOX' and result['unit'] == unit and result['observed_at']
    assert result['coverage'] == 'provider_label_count'
    assert len(wire.calls) == 1 and wire.calls[0][0] == 'labels/INBOX'
    assert 'private' not in str(result) and 'synthetic' not in str(result)


def test_no_grant_or_project_only_grant_cannot_read_one_off(f):
    connect(f); grant(f, 'one', 'project-grant', project_id='school')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 3})
    assert svc.count(f.owner)['status'] == 'blocked'
    assert not wire.calls


def test_label_scope_narrow_and_all_mail_grant(f):
    connect(f); grant(f, 'one', 'school', resources=('Label_12',))
    svc, wire = service(f, {'id': 'Label_12', 'messagesUnread': 2})
    assert svc.count(f.owner)['status'] == 'blocked' and not wire.calls
    result = svc.count(f.owner, label_id='Label_12')
    assert result['count'] == 2 and wire.calls[0][0] == 'labels/Label_12'
    grant(f, 'one', 'all', resources=(ALL_MAIL,))
    svc2, _ = service(f, {'id': 'INBOX', 'messagesUnread': 5})
    assert svc2.count(f.owner)['count'] == 5
    with pytest.raises(ValueError):
        svc2.count(f.owner, label_id='../messages')


def test_multiple_accounts_are_choice_not_implicit_selection(f):
    connect(f, connection='one'); connect(f, connection='two')
    grant(f, 'one', 'grant-one'); grant(f, 'two', 'grant-two')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 9})
    result = svc.count(f.owner)
    assert result['status'] == 'account_choice_required'
    assert {x['connection_id'] for x in result['accounts']} == {'one', 'two'} and not wire.calls
    assert svc.count(f.owner, connection_id='two')['status'] == 'account_choice_required'
    assert not wire.calls


def test_actor_cannot_use_someone_elses_grant_or_invent_account(f):
    connect(f); grant(f, 'one', 'grant')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 1})
    assert svc.count(f.other)['status'] == 'blocked'
    assert svc.count(f.other, connection_id='one')['status'] == 'blocked'
    assert not wire.calls


def test_revocation_during_io_and_cancellation_fence_response(f):
    connect(f); grant(f, 'one', 'grant')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 2})
    wire.after = lambda: f.accounts.revoke_grant(f.owner, grant_id='grant', expected_revision=1)
    assert svc.count(f.owner)['status'] == 'blocked'
    assert len(wire.calls) == 1

    grant(f, 'one', 'newgrant')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 2})
    cancelled = [False]
    wire.after = lambda: cancelled.__setitem__(0, True)
    def guard():
        if cancelled[0]: raise CheckStopped('cancelled', 'task_cancelled')
    assert svc.count(f.owner, guard=guard)['status'] == 'cancelled'
    assert len(wire.calls) == 1


@pytest.mark.parametrize('response,expected', [
    (GmailHTTPError(401), 'blocked'), (GmailHTTPError(403), 'blocked'),
    (GmailHTTPError(429), 'partial'), (GmailHTTPError(503), 'partial'),
    (GmailHTTPError(404), 'unknown'), ({'id': 'INBOX'}, 'unknown'),
    ({'id': 'INBOX', 'messagesUnread': True}, 'unknown'),
    ({'id': 'OTHER', 'messagesUnread': 0}, 'unknown')])
def test_provider_failures_do_not_fabricate_zero(f, response, expected):
    connect(f); grant(f, 'one', 'grant')
    svc, _ = service(f, response)
    result = svc.count(f.owner)
    assert result['status'] == expected and 'count' not in result


def test_native_tool_uses_authenticated_task_not_arguments(f):
    connect(f); grant(f, 'one', 'grant')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 4})
    tool = build_gmail_unread_tool_definition(svc, delivery_is_private=lambda context: True)
    context = SimpleNamespace(task=SimpleNamespace(user='owner', status='running'), cancellation_checker=None)
    result = tool.callable({'unit': 'messages'}, context=context)
    assert result['count'] == 4 and len(wire.calls) == 1
    assert 'user_id' not in tool.argument_schema['properties']


def test_grant_beneficiary_can_read_but_other_accounts_are_not_exposed(f):
    connect(f); grant(f, 'one', 'delegated', actor='other')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 3})
    assert svc.count(f.owner)['status'] == 'blocked' and not wire.calls
    assert svc.count(f.other)['count'] == 3


def test_inflight_byte_budget_never_returns_a_count(f):
    connect(f); grant(f, 'one', 'grant')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 3, 'padding': 'x' * 500})
    result = svc.count(f.owner, max_bytes=64)
    assert result['status'] == 'partial' and 'count' not in result
    assert len(wire.calls) == 1


def test_tool_cancellation_and_missing_service_are_safe(f):
    tool = build_gmail_unread_tool_definition(None, delivery_is_private=lambda context: True)
    context = SimpleNamespace(task=SimpleNamespace(user='owner', status='running'), cancellation_checker=None)
    assert tool.callable({}, context=context)['status'] == 'blocked'
    context.cancellation_checker = lambda: True
    with pytest.raises(CheckStopped):
        tool.callable({}, context=context)


def test_fixture_registered_gmail_tool_curation_reveal_and_tactical_execution(f, monkeypatch):
    """Synthetic Spanish request through real registry, curation and V3 Do.

    The model responses are fixtures; this does not establish live intent accuracy.
    """
    from alphonse.agent_v2.core.core import CoreLoopContext
    from alphonse.agent_v2.core.intelligence.v3 import (
        PhaseExecutor, PhasePlan, PhaseStatus, PhaseSubgoal, SideEffectClass,
        CompletionCondition, ToolRevealPolicy, new_tactical_state,
    )
    from alphonse.agent_v2.core.intelligence.v3.tool_curation import curate_request_tools
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
    from alphonse.agent_v2.integrations.gmail import GmailTransport
    import alphonse.agent_v2.system_one as system_one
    from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings, SystemOneTacticalReview

    connect(f); grant(f, 'one', 'grant')
    wire = Wire({'id': 'INBOX', 'messagesUnread': 7})
    monkeypatch.setattr(GmailTransport, 'get', lambda self, path, **kw: wire.get(path, **kw))
    management = SimpleNamespace(accounts=f.accounts, oauth_factory=lambda session: f.service, oauth=None)
    runtime = SimpleNamespace(integration_management=management)
    registry = build_native_tool_registry(management_runtime=runtime)
    # Production registration is intentionally disabled until an authenticated
    # private-channel audience contract and account-choice binding exist.
    assert registry.get('native.gmail_unread_count') is None
    registry.register(build_gmail_unread_tool_definition(runtime=runtime, delivery_is_private=lambda context: True))
    assert registry.get('native.gmail_unread_count').metadata['v3_capabilities'] == ['mail_read']
    requests = []
    def transport(url, key, payload, timeout):
        requests.append(payload)
        return {'answers': {q: {'type': 'noul', 'noul': .95} for q in payload['questions']}}
    # Fixture-only relevance question; production registration remains disabled.
    native_templates = dict(system_one._load_jev_native_tool_registry())
    native_templates['native.gmail_unread_count'] = {'type': 'noul',
        'instructions': 'Does `request.text` ask for a current Gmail unread count?',
        'criteria': {'true': 'The requester wants an unread Gmail count.',
                     'false': 'The requester does not ask for an unread Gmail count.'}}
    monkeypatch.setattr(system_one, '_load_jev_native_tool_registry', lambda: native_templates)
    jev = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture-secret'), transport=transport)
    task = TaskState(task_id='gmail-test', user='owner', goal='Hola Alphonse, puedes ver mi gmail? cuántos correos nuevos tengo sin revisar?')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=jev)
    selected = curate_request_tools(task, registry.list(), context)
    assert 'native.gmail_unread_count' in {row.tool_id for row in selected}
    assert len(requests) == 1 and any('Gmail unread count' in q['instructions'] for q in requests[0]['questions'].values())
    assert task.metadata['v3_request_tool_curation']['status'] == 'used'
    stage = PhaseSubgoal('count', 'Count unread inbox messages', 'gmail_count',
        allowed_capabilities=('mail_read',), allowed_side_effects=(SideEffectClass.READ_ONLY,),
        completion=CompletionCondition('field_equals', field='status', expected='complete'))
    state = new_tactical_state(PhasePlan('phase', 'Get current Gmail count', (stage,), authorized_capabilities=('mail_read',)))
    context.system_one = SimpleNamespace(evaluate_tactical_progress=lambda **kw: SystemOneTacticalReview(
        complete=True, confidence=.99, confident=True, answers={'runtime_declared_subgoal_complete': .99}))
    selected_actions = []
    def select(state, subgoal, tools):
        selected_actions.append([t.tool_id for t in tools])
        return {'tool_id': 'native.gmail_unread_count', 'arguments': {}}
    outcome = PhaseExecutor(action_selector=select, reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.PHASE_COMPLETE, (outcome, state.to_dict())
    assert len(selected_actions) == 1 and 'native.gmail_unread_count' in selected_actions[0]
    assert len(wire.calls) == 1 and wire.calls[0][0] == 'labels/INBOX'
    assert state.actions[0].result['count'] == 7


def test_choice_candidates_are_reauthorized_before_disclosure(f, monkeypatch):
    connect(f, connection='one'); connect(f, connection='two')
    grant(f, 'one', 'grant-one'); grant(f, 'two', 'grant-two')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 9})
    original = svc._candidates
    calls = []
    def revoke_after_discovery(*args):
        result = original(*args)
        calls.append(1)
        if len(calls) == 1:
            f.accounts.revoke_grant(f.owner, grant_id='grant-two', expected_revision=1)
        return result
    monkeypatch.setattr(svc, '_candidates', revoke_after_discovery)
    result = svc.count(f.owner)
    assert result == {'status': 'blocked', 'reason': 'gmail_account_choices_changed'}
    assert 'two' not in str(result) and not wire.calls


@pytest.mark.parametrize('provider', ['discord', 'telegram'])
def test_group_channel_and_unproven_audience_cannot_read_mail(f, provider):
    connect(f); grant(f, 'one', 'grant')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 9})
    # An authenticated actor and an asserted private metadata flag do not prove
    # that the delivery destination is private.
    task = SimpleNamespace(user='owner', status='running', metadata={'channel': provider, 'is_private': True})
    context = SimpleNamespace(task=task, cancellation_checker=None)
    result = build_gmail_unread_tool_definition(svc).callable({}, context=context)
    assert result['status'] == 'blocked' and result['reason'] == 'gmail_private_delivery_unverified'
    assert not wire.calls


def test_only_server_issued_authenticated_choice_selects_between_accounts(f):
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.integrations.gmail_choice import issue_choice
    connect(f, connection='one'); connect(f, connection='two')
    grant(f, 'one', 'grant-one'); grant(f, 'two', 'grant-two')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 9})
    store = SQLiteQuestionStore()
    task = TaskState(task_id='mail-choice-task', user='owner', status='running', goal='Count mail')
    task.metadata['channel'] = {'integration_id': 'desktop', 'provider_key': 'tui',
        'channel_target': 'owner', 'alphonse_user_id': 'owner'}
    initial = svc.count(f.owner)
    assert initial['status'] == 'account_choice_required'
    question = issue_choice(store, task=task, accounts=initial['accounts'], label_id='INBOX', unit='messages')
    assert svc.count(f.owner, connection_id='two', choice_question_id=question.question_id,
                     choice_store=store, task=task)['status'] == 'account_choice_required'
    assert not wire.calls
    assert store.route_answer(respondent_user_id='owner', question_id=question.question_id,
                              payload={'choice_id': 'two'}).handled
    result = svc.count(f.owner, connection_id='two', choice_question_id=question.question_id,
                       choice_store=store, task=task)
    assert result['status'] == 'complete' and result['connection_id'] == 'two' and result['count'] == 9
    assert len(wire.calls) == 1


def test_account_choice_is_invalidated_by_grant_revocation(f):
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.integrations.gmail_choice import issue_choice
    connect(f, connection='one'); connect(f, connection='two')
    grant(f, 'one', 'grant-one'); grant(f, 'two', 'grant-two')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 9})
    store = SQLiteQuestionStore()
    task = TaskState(task_id='mail-choice-task', user='owner', status='running', goal='Count mail')
    task.metadata['channel'] = {'integration_id': 'desktop', 'provider_key': 'tui',
        'channel_target': 'owner', 'alphonse_user_id': 'owner'}
    initial = svc.count(f.owner)
    question = issue_choice(store, task=task, accounts=initial['accounts'], label_id='INBOX', unit='messages')
    store.route_answer(respondent_user_id='owner', question_id=question.question_id, payload={'choice_id': 'two'})
    f.accounts.revoke_grant(f.owner, grant_id='grant-two', expected_revision=1)
    result = svc.count(f.owner, connection_id='two', choice_question_id=question.question_id,
                       choice_store=store, task=task)
    assert result['status'] == 'blocked' and not wire.calls
    # A model cannot evade the prior choice by omitting its ID after the other
    # account disappears; that would silently select a different mailbox.
    result = svc.count(f.owner, choice_store=store, task=task)
    assert result['status'] == 'blocked' and not wire.calls


def test_private_audience_revoked_during_io_withholds_count(f):
    connect(f); grant(f, 'one', 'grant')
    svc, wire = service(f, {'id': 'INBOX', 'messagesUnread': 9})
    state = {'private': True}
    wire.after = lambda: state.update(private=False)
    context = SimpleNamespace(task=SimpleNamespace(user='owner', status='running'), cancellation_checker=None)
    tool = build_gmail_unread_tool_definition(svc, delivery_is_private=lambda context: state['private'])
    result = tool.callable({}, context=context)
    assert result == {'status': 'blocked', 'reason': 'gmail_private_delivery_unverified'}
    assert len(wire.calls) == 1 and 'count' not in result
