"""Only synthetic Google responses; no household programs, records or network."""
import copy
import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from alphonse.agent_v2.integrations.store import SQLiteIntegrationStore
from alphonse.agent_v2.skill_bindings import SkillBindings
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.users import V2UserStore

ROOT = Path(__file__).parents[1] / 'alphonse/agent_v2/builtin_skills/family-calendar'
spec = importlib.util.spec_from_file_location('family_calendar_client', ROOT / 'scripts/google_calendar.py')
client = importlib.util.module_from_spec(spec)
spec.loader.exec_module(client)
NOW = datetime(2030, 1, 1, tzinfo=timezone.utc)


def event():
    return {'summary': 'Synthetic test',
            'start': {'dateTime': '2030-01-10T10:00:00-06:00', 'timeZone': 'America/Mexico_City'},
            'end': {'dateTime': '2030-01-10T10:30:00-06:00', 'timeZone': 'America/Mexico_City'}}


def create():
    return {'operation': 'create', 'calendar_id': 'synthetic@invalid',
            'operation_id': 'synthetic-operation-0001', 'send_updates': 'none', 'event': event()}


def bindings(request=None, scopes='calendar'):
    result = {'calendar': {'kind': 'account', 'provider': 'google-calendar',
                          'credentials': {'refresh_token': 'synthetic-secret'},
                          'configuration': {'granted_scopes': client.SCOPE + scopes,
                                            'calendars': {'synthetic@invalid': ['list', 'get', 'availability', 'create', 'update']}}}}
    if request:
        result['authorization'] = {'kind': 'configuration', 'configuration': {
            'request_sha256': client.digest(request), 'expires_at': '2030-01-02T00:00:00Z'}}
    return result


class Google:
    def __init__(self):
        self.events, self.calls = {}, []
        self.fail_after_write = False
        self.bad_readback = False
        self.conflict = False

    def __call__(self, method, url, body, headers):
        self.calls.append((method, url, copy.deepcopy(body), headers))
        assert headers['Authorization'] == 'Bearer synthetic-access'
        path = urlsplit(url).path
        if path.endswith('/freeBusy'):
            return {'calendars': {'synthetic@invalid': {'busy': []}}}
        if method == 'GET':
            event_id = path.rsplit('/', 1)[-1]
            if event_id not in self.events:
                raise client.ProviderError(404)
            result = copy.deepcopy(self.events[event_id])
            if self.bad_readback:
                result['summary'] = 'wrong'
            return result
        if method == 'POST':
            if body['id'] in self.events:
                raise client.ProviderError(409)
            self.events[body['id']] = {**copy.deepcopy(body), 'etag': 'etag-1'}
        elif method == 'PATCH':
            if self.conflict:
                raise client.ProviderError(412)
            event_id = path.rsplit('/', 1)[-1]
            assert headers['If-Match'] == self.events[event_id]['etag']
            self.events[event_id].update(copy.deepcopy(body), etag='etag-2')
        else:
            pytest.fail('unexpected HTTP operation')
        assert parse_qs(urlsplit(url).query)['sendUpdates'] == ['none']
        assert 'attendees' not in body
        if self.fail_after_write:
            self.fail_after_write = False
            raise client.CalendarError('uncertain')
        return {}


def run(request, google=None, bound=None, renew=None):
    return client.Calendar(bound if bound is not None else bindings(request),
                           send=google or Google(), renew=renew or (lambda _: {'access_token': 'synthetic-access'}),
                           now=lambda: NOW).run(request)


def test_create_readback_and_resume_after_uncertain_network_write():
    request, google = create(), Google()
    google.fail_after_write = True
    with pytest.raises(client.CalendarError, match='uncertain'):
        run(request, google)
    result = run(request, google)
    assert result['verified']
    assert result['event']['id'] == hashlib.sha256(request['operation_id'].encode()).hexdigest()
    assert len(google.events) == 1
    assert [call[0] for call in google.calls].count('POST') == 1
    assert run(request, google)['verified']


def test_create_conflicting_id_never_claims_success_or_duplicates():
    request, google = create(), Google()
    run(request, google)
    request['event']['summary'] = 'another intention'
    with pytest.raises(client.CalendarError, match='readback_mismatch'):
        run(request, google)
    assert len(google.events) == 1


def update(google):
    result = run(create(), google)['event']
    return {'operation': 'update', 'calendar_id': 'synthetic@invalid',
            'operation_id': 'synthetic-update-0001', 'send_updates': 'none',
            'event_id': result['id'], 'etag': result['etag'], 'recurrence_scope': 'event',
            'event': {**event(), 'summary': 'Changed'}}


def test_update_if_match_and_resume_without_second_patch():
    google = Google()
    request = update(google)
    google.fail_after_write = True
    with pytest.raises(client.CalendarError, match='uncertain'):
        run(request, google)
    assert run(request, google)['verified']
    assert [call[0] for call in google.calls].count('PATCH') == 1


@pytest.mark.parametrize('change', ['etag', 'race', 'attendees', 'scope', 'readback'])
def test_update_rejects_conflicts_scope_guests_and_failed_readback(change):
    google = Google()
    request = update(google)
    if change == 'etag':
        request['etag'] = 'stale'
    if change == 'race':
        google.conflict = True
    if change == 'attendees':
        google.events[request['event_id']]['attendees'] = [{'email': 'synthetic@invalid'}]
    if change == 'scope':
        request['recurrence_scope'] = 'series'
    if change == 'readback':
        google.bad_readback = True
    with pytest.raises(client.CalendarError):
        run(request, google)
    if change in {'etag', 'attendees', 'scope'}:
        assert all(call[0] != 'PATCH' for call in google.calls)


@pytest.mark.parametrize('change', ['no_grant', 'hash', 'expired', 'calendar', 'operation', 'provider', 'invites', 'extra', 'id'])
def test_deterministic_access_and_exact_effect_denials_before_network(change):
    request, google = create(), Google()
    bound = bindings(request)
    if change == 'no_grant':
        bound.pop('authorization')
    elif change == 'hash':
        request['event']['summary'] = 'unapproved'
    elif change == 'expired':
        bound['authorization']['configuration']['expires_at'] = '2029-12-31T00:00:00Z'
    elif change == 'calendar':
        request['calendar_id'] = 'someone-else@invalid'
    elif change == 'operation':
        bound['calendar']['configuration']['calendars']['synthetic@invalid'] = ['get']
    elif change == 'provider':
        bound['calendar']['provider'] = 'other'
    elif change == 'invites':
        request['send_updates'] = 'all'
        bound = bindings(request)
    elif change == 'extra':
        request['event']['attendees'] = [{'email': 'synthetic@invalid'}]
        bound = bindings(request)
    elif change == 'id':
        request['operation_id'] = ''
        bound = bindings(request)
    with pytest.raises(client.CalendarError):
        run(request, google, bound)
    assert not google.calls


@pytest.mark.parametrize('scopes,op,success', [('calendar.readonly', 'create', False),
    ('calendar.events', 'availability', False), ('calendar.freebusy', 'availability', True),
    ('calendar.events', 'get', True), ('', 'create', False)])
def test_scope_checks(scopes, op, success):
    request, google = create(), Google()
    if op != 'create':
        run(request, google)
        request = {'operation': op, 'calendar_id': 'synthetic@invalid',
                   **({'event_id': next(iter(google.events))} if op == 'get' else
                      {'time_min': '2030-01-10T00:00:00Z', 'time_max': '2030-01-11T00:00:00Z'})}
    if success:
        assert run(request, google, bindings(request, scopes))
    else:
        before = len(google.calls)
        with pytest.raises(client.CalendarError, match='missing_scopes'):
            run(request, google, bindings(request, scopes))
        assert len(google.calls) == before


def test_fresh_scope_evidence_overrides_stored_grant_and_revocation_fails_closed():
    request, google = create(), Google()
    with pytest.raises(client.CalendarError, match='missing_scopes'):
        run(request, google, renew=lambda _: {'access_token': 'synthetic-access', 'scope': client.SCOPE + 'calendar.readonly'})
    def revoked(_):
        raise client.CalendarError('account_revoked_reconnect')
    with pytest.raises(client.CalendarError, match='reconnect'):
        run(request, google, renew=revoked)
    assert not google.calls


@pytest.mark.parametrize('change', ['ambiguous', 'nonexistent', 'offset', 'reverse', 'unbounded', 'attendees'])
def test_ambiguous_dates_dst_and_unsupported_fields_fail_before_network(change):
    request, google = create(), Google()
    if change == 'ambiguous':
        request['event']['start']['dateTime'] = 'next Friday'
    elif change == 'nonexistent':
        request['event']['start'] = {'dateTime': '2030-03-10T02:30:00-05:00', 'timeZone': 'America/New_York'}
    elif change == 'offset':
        request['event']['start']['dateTime'] = '2030-01-10T10:00:00Z'
    elif change == 'reverse':
        request['event']['end'] = copy.deepcopy(request['event']['start'])
    elif change == 'unbounded':
        request['event']['recurrence'] = ['RRULE:FREQ=WEEKLY']
    elif change == 'attendees':
        request['event']['attendees'] = []
    with pytest.raises(client.CalendarError):
        run(request, google)
    assert not google.calls


def test_all_day_and_recurring_series_and_instance_updates():
    request, google = create(), Google()
    request['event'].update(start={'date': '2030-01-10'}, end={'date': '2030-01-11'},
                            recurrence=['RRULE:FREQ=WEEKLY;COUNT=2'])
    saved = run(request, google)['event']
    patch = {'operation': 'update', 'calendar_id': request['calendar_id'], 'operation_id': 'synthetic-series-001',
             'send_updates': 'none', 'event_id': saved['id'], 'etag': saved['etag'], 'recurrence_scope': 'series',
             'event': {**request['event'], 'summary': 'Series title'}}
    assert run(patch, google)['verified']
    instance_id = 'syntheticinstance'
    google.events[instance_id] = {**event(), 'id': instance_id, 'etag': 'instance-etag', 'recurringEventId': saved['id']}
    patch.update(event_id=instance_id, etag='instance-etag', recurrence_scope='instance', event={**event(), 'summary': 'One occurrence'})
    assert run(patch, google)['verified']
    assert google.events[saved['id']]['summary'] == 'Series title'


def test_listing_expands_recurrences_and_follows_pages():
    calls = []
    def send(method, url, body, headers):
        query = parse_qs(urlsplit(url).query)
        assert query['singleEvents'] == ['true']
        calls.append(query)
        return {'items': [{'id': 'second'}]} if 'pageToken' in query else {'items': [{'id': 'first'}], 'nextPageToken': 'page-2'}
    request = {'operation': 'list', 'calendar_id': 'synthetic@invalid',
               'time_min': '2030-01-01T00:00:00Z', 'time_max': '2030-01-02T00:00:00Z'}
    assert run(request, send)['items'] == [{'id': 'first'}, {'id': 'second'}]
    assert len(calls) == 2


def test_freebusy_errors_are_not_free():
    request = {'operation': 'availability', 'calendar_id': 'synthetic@invalid',
               'time_min': '2030-01-01T00:00:00Z', 'time_max': '2030-01-02T00:00:00Z'}
    for response in ({}, {'calendars': {'synthetic@invalid': {}}}, {'calendars': {'synthetic@invalid': {'errors': [{'reason': 'notFound'}]}}}):
        with pytest.raises(client.CalendarError, match='availability_unknown'):
            run(request, lambda *args: response)


def test_package_binding_boundary_rechecks_actor_account_and_grant(tmp_path):
    users = V2UserStore(tmp_path / 'users.sqlite')
    users.onboard(display_name='Admin', user_id='owner', users_root=tmp_path / 'profiles')
    users.create_user(display_name='Member', user_id='member')
    users.create_user(display_name='Other', user_id='other')
    skills = SkillStore(tmp_path / 'skills')
    assert skills.get('skill:family-calendar') is not None
    integrations = SQLiteIntegrationStore(tmp_path / 'accounts.sqlite')
    integrations.upsert(integration_id='synthetic-google', provider_key='google-calendar', display_name='Synthetic',
                        enabled=True, secrets={'refresh_token': 'synthetic-refresh', 'client_id': 'synthetic-client', 'client_secret': 'synthetic-secret'})
    resolver = SkillBindings(integrations, users, skills)
    request = create()
    resolver.configure(actor_id='owner', owner_id='member', binding_id='calendar-test',
                       skill_id='skill:family-calendar', kind='account', target='synthetic-google',
                       required_secrets=('refresh_token', 'client_id', 'client_secret'),
                       configuration=bindings()['calendar']['configuration'])
    resolver.select(actor_id='member', skill_id='skill:family-calendar', name='calendar', binding_id='calendar-test')
    with pytest.raises(PermissionError):
        resolver.select(actor_id='other', skill_id='skill:family-calendar', name='calendar', binding_id='calendar-test')
    payload = resolver.prepare(actor_id='member', skill_id='skill:family-calendar', names=['calendar'])
    with pytest.raises(client.CalendarError, match='authorization_required'):
        run(request, bound=payload.values)
    resolver.configure(actor_id='owner', owner_id='member', binding_id='effect-test', skill_id='skill:family-calendar',
                       kind='configuration', configuration=bindings(request)['authorization']['configuration'])
    resolver.select(actor_id='member', skill_id='skill:family-calendar', name='authorization', binding_id='effect-test')
    payload = resolver.prepare(actor_id='member', skill_id='skill:family-calendar', names=['calendar', 'authorization'])
    assert run(request, bound=payload.values)['verified']
    integrations.set_enabled('synthetic-google', False)
    with pytest.raises(ValueError, match='account_unavailable'):
        resolver.prepare(actor_id='member', skill_id='skill:family-calendar', names=['calendar', 'authorization'])


def test_concurrent_create_conflict_is_reconciled_by_readback():
    request, google = create(), Google()
    original = google.__call__
    def send(method, url, body, headers):
        if method == 'POST':
            original(method, url, body, headers)  # A concurrent request saved it first.
            raise client.ProviderError(409)
        return original(method, url, body, headers)
    assert run(request, send)['verified']
    assert len(google.events) == 1


@pytest.mark.parametrize('status', [400, 401, 403, 429, 500])
def test_oauth_and_api_errors_are_static_without_secret_bodies(monkeypatch, status):
    from io import BytesIO
    from urllib.error import HTTPError
    def failed(request, timeout):
        raise HTTPError(request.full_url, status, 'synthetic-secret', {}, BytesIO(b'synthetic-refresh-token'))
    monkeypatch.setattr(client, 'urlopen', failed)
    with pytest.raises(client.CalendarError) as error:
        client.refresh({'client_id': 'synthetic-client', 'client_secret': 'synthetic-secret', 'refresh_token': 'synthetic-refresh-token'})
    assert 'synthetic' not in str(error.value)
    with pytest.raises(client.ProviderError) as error:
        client.transport('GET', client.API + '/synthetic')
    assert 'synthetic' not in str(error.value)


def test_main_uses_managed_credentials_and_real_http_encoding_without_logging(monkeypatch, capsys):
    from io import StringIO
    from urllib.error import HTTPError
    request = create()
    bound = bindings(request)
    bound['calendar']['credentials'] = {'client_id': 'synthetic-client', 'client_secret': 'synthetic-secret',
                                        'refresh_token': 'synthetic-refresh'}
    bound['authorization']['configuration']['expires_at'] = '2099-01-01T00:00:00Z'
    saved, calls = {}, []
    class Response(StringIO):
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
    def urlopen(request, timeout):
        calls.append(request)
        assert timeout == 25
        if request.full_url == 'https://oauth2.googleapis.com/token':
            assert parse_qs(request.data.decode())['grant_type'] == ['refresh_token']
            return Response(json.dumps({'access_token': 'synthetic-access', 'scope': client.SCOPE + 'calendar'}))
        assert request.headers['Authorization'] == 'Bearer synthetic-access'
        if request.method == 'GET':
            if not saved:
                raise HTTPError(request.full_url, 404, 'missing', {}, None)
            return Response(json.dumps(saved))
        assert request.method == 'POST'
        assert parse_qs(urlsplit(request.full_url).query)['sendUpdates'] == ['none']
        saved.update(json.loads(request.data))
        return Response(json.dumps(saved))
    monkeypatch.setattr(client, 'urlopen', urlopen)
    monkeypatch.setenv('ALPHONSE_SKILL_BINDINGS', json.dumps(bound))
    monkeypatch.setattr(client.sys, 'stdin', StringIO(json.dumps(request)))
    assert client.main() == 0
    output = capsys.readouterr()
    assert json.loads(output.out)['verified']
    assert not output.err
    assert all(value not in output.out for value in ['synthetic-secret', 'synthetic-refresh', 'synthetic-access'])
    assert len(calls) == 4  # OAuth, initial GET, insert, readback.


def test_native_bash_runs_shipped_package_without_catalog_or_network(tmp_path):
    from alphonse.agent_v2.core.core import ToolExecutionContext
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash
    users = V2UserStore(tmp_path / 'users.sqlite')
    users.onboard(display_name='Owner', user_id='owner', users_root=tmp_path / 'profiles')
    skills = SkillStore(tmp_path / 'skills with spaces')
    integrations = SQLiteIntegrationStore(tmp_path / 'accounts.sqlite')
    integrations.upsert(integration_id='synthetic', provider_key='google-calendar', display_name='Synthetic', enabled=True,
                        secrets={'refresh_token': 'synthetic-secret'})
    resolver = SkillBindings(integrations, users, skills)
    resolver.configure(actor_id='owner', owner_id='owner', binding_id='calendar-test', skill_id='skill:family-calendar',
                       kind='account', target='synthetic', required_secrets=('refresh_token',),
                       configuration=bindings()['calendar']['configuration'])
    resolver.select(actor_id='owner', skill_id='skill:family-calendar', name='calendar', binding_id='calendar-test')
    task = TaskState(goal='Synthetic calendar validation', user='owner',
                     metadata={'v3_skill_selection': {'selected_skill_ids': ['skill:family-calendar']}})
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), skill_bindings=resolver)
    command = '"' + client.sys.executable + '" scripts/google_calendar.py <<\'REQUEST\'\n' + json.dumps(create()) + '\nREQUEST'
    result = execute_bash({'skill_id': 'skill:family-calendar', 'binding_names': ['calendar'], 'command': command}, context=context)
    assert result['exit_code'] == 1
    assert 'exact_effect_authorization_required' in result['stderr']
    assert 'synthetic-secret' not in json.dumps(result)
