"""Owner Calendar agenda through the simpler loop with disposable Google data."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import os
import time
from unittest.mock import patch
from urllib.parse import urlencode

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.simple_loop.owner_calendar import OWNER_CALENDAR_TOOL_ID, OwnerCalendarService, calendar_window
from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_read_memory_source_tool_definition, build_search_memory_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.integrations.google_calendar import CalendarHTTPError
from alphonse.agent_v2.ipc import V2DaemonClient, V2DaemonServer
from alphonse.agent_v2.runtime import build_runtime_host
from test_gmail_adapter import Wire
from test_oauth_lifecycle import expire
from test_owner_gmail import f

WINDOW = {'time_min': '2026-12-01T00:00:00-06:00', 'time_max': '2026-12-02T00:00:00-06:00'}
GOAL = '¿Qué tengo en mi calendario de Google el 1 de diciembre de 2026? Usa la hora de Ciudad de México.'
CALENDAR_ID = 'family@group.calendar.google.com'
TITLE = 'Synthetic pottery handoff'
PAGE = {'accessRole': 'owner', 'timeZone': 'America/Mexico_City', 'items': [
    {'summary': TITLE, 'start': {'dateTime': '2026-12-01T09:15:00-06:00'},
     'end': {'dateTime': '2026-12-01T10:15:00-06:00'}, 'description': 'PRIVATE_DESCRIPTION_MARKER'},
    {'summary': 'Synthetic multi-day item', 'start': {'date': '2026-12-01'}, 'end': {'date': '2026-12-03'}},
    {'summary': 'CANCELLED_EVENT_MARKER', 'status': 'cancelled'},
]}


def connect_calendar(f, *, grant=True):
    f.service.begin(f.owner, connection_id='one', provider='google', capabilities=['calendar_read'], expected_revision=0)
    client = f.service.clients['google']
    f.service.complete(f.owner, client.config.redirect_uri + '?' + urlencode({'state': client.state, 'code': 'disposable-code'}))
    if grant:
        grant_calendar(f, CALENDAR_ID, 'grant')


def grant_calendar(f, resource, grant_id):
    f.accounts.save_grant(f.owner, grant_id=grant_id, connection_id='one', expected_revision=0,
        beneficiary_user_id='owner', operations=['calendar_read'], resources=[resource], recipient_user_ids=['owner'],
        expires_at=(datetime.now(timezone.utc) + timedelta(days=1)).isoformat())


class ScriptedProvider:
    supports_image_input = False

    def __init__(self, *, second=None, arguments=None):
        self.requests = []
        self.second = second
        self.arguments = WINDOW if arguments is None else arguments

    def generate_json(self, request):
        self.requests.append(request)
        if len(self.requests) == 1:
            action = {'kind': 'tool', 'tool_id': OWNER_CALENDAR_TOOL_ID, 'arguments': self.arguments, 'text': None}
        else:
            if self.second is not None:
                self.second()
            action = {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'An unrelated model draft.'}
        return InferenceResult(json_value=action, raw_response=json.dumps(action),
                               usage={'input_tokens': 100, 'output_tokens': 20})


@contextmanager
def calendar_worker(f, root, wire, provider, *, model='fixture', socket=False):
    path = Path(f.accounts.users.db_path)
    with patch.dict(os.environ, {'ALPHONSE_V2_DB_PATH': str(path)}), patch.object(Path, 'home', classmethod(lambda cls: root)):
        queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
        conversation = SQLiteConversationStore(path)
        registry = InMemoryToolRegistry()
        registry.register(build_search_memory_tool_definition())
        registry.register(build_read_memory_source_tool_definition())
        runtime = build_runtime_host(user_store=f.accounts.users, integration_store=f.store, messages=queue,
            question_store=SQLiteQuestionStore(path), conversation_store=conversation, outbox=SQLiteOutboundStore(path),
            inference=InferenceRouter(provider=provider, default_profile=ModelProfile('openai', model, 'calendar-dojo')),
            tools=registry, coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
        runtime.owner_calendar_service = OwnerCalendarService(f.accounts, lambda _: f.service, transport=wire)
        daemon = V2Daemon(runtime, daemon_id='calendar-dojo', experimental_owner_only=True)
        daemon._acquire_single_instance_lock = lambda: None
        daemon._release_single_instance_lock = lambda: None
        if socket:
            daemon.ipc = V2DaemonServer(daemon, socket_path=root / 'owner.sock', experimental_owner_ingress=True)
        else:
            daemon.ipc.start = lambda: None
            daemon.ipc.stop = lambda: None
        daemon.start()
        try:
            yield daemon, queue, conversation, V2DaemonClient(root / 'owner.sock') if socket else None
        finally:
            daemon.stop()


def worker_status(status, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = status()
        if result not in {'pending', 'processing'}:
            return result
        time.sleep(.02)
    return 'timeout'


def test_calendar_titles_are_private_to_verified_desktop_pickup(f, tmp_path):
    connect_calendar(f)
    wire, provider = Wire(PAGE), ScriptedProvider()
    with calendar_worker(f, tmp_path, wire, provider) as (daemon, queue, conversation, _):
        message = queue.enqueue_trusted_local_owner(prompt=GOAL, peer_uid=os.getuid())
        assert worker_status(lambda: daemon.owner_message_status(message)) == 'ready'
        assert TITLE not in '\n'.join(request.prompt for request in provider.requests)
        answer = daemon.take_owner_desktop_draft(message)
        assert answer['private_calendar'] is True and answer['kind'] == 'answer'
        assert TITLE in answer['message'] and '09:15:00-06:00' in answer['message']
        assert '2026-12-01 to 2026-12-02 (all day)' in answer['message']
        create, update = answer['private_a2ui']
        assert create['createSurface']['surfaceId'] == f'private-calendar:{message}'
        visible = [item.get('text', '') for item in update['updateComponents']['components']]
        assert TITLE in visible
        assert 'All day' in visible and 'Continues through Wed, 02 Dec 2026' in visible
        assert not any(marker in answer['message'] for marker in ('PRIVATE_DESCRIPTION_MARKER', 'CANCELLED_EVENT_MARKER', 'model draft'))
        daemon.acknowledge_owner_desktop_draft(message, answer['acknowledgement_token'])
        daemon.acknowledge_owner_desktop_draft(message, answer['acknowledgement_token'])
        history = conversation.list(owner_user_id='owner')
        assert history[-1].content == 'Google Calendar results were displayed privately in Desktop.'
        assert TITLE not in '\n'.join(event.content for event in history)
        assert len(wire.calls) == 1 and wire.calls[0][0] == 'calendars/family%40group.calendar.google.com/events'
        assert wire.calls[0][1]['params']['timeMin'] == WINDOW['time_min']
        assert 'description' not in wire.calls[0][1]['params']['fields']
        assert f.service.clients['google'].refresh_count == 0


@pytest.mark.parametrize('case, expected', [
    ('no_grant', 'grant access to the exact calendar'),
    ('expired', 'Reconnect Google Calendar'),
    ('ambiguous', 'More than one Google calendar'),
    ('partial', 'first page of results'),
    ('empty', 'No events were returned'),
    ('api_disabled', 'enable Google Calendar API in the project configured for Alphonse'),
])
def test_calendar_observation_distinguishes_blockers_and_coverage(f, tmp_path, case, expected):
    connect_calendar(f, grant=case != 'no_grant')
    if case == 'expired':
        expire(f)
    if case == 'ambiguous':
        grant_calendar(f, 'other@group.calendar.google.com', 'second-grant')
    page = {**PAGE, 'nextPageToken': 'private-cursor'} if case == 'partial' else {**PAGE, 'items': []}
    wire = Wire(CalendarHTTPError(403, api_disabled=True) if case == 'api_disabled' else page)
    with calendar_worker(f, tmp_path, wire, ScriptedProvider()) as (daemon, queue, _, _):
        message = queue.enqueue_trusted_local_owner(prompt=GOAL, peer_uid=os.getuid())
        assert worker_status(lambda: daemon.owner_message_status(message)) == 'ready'
        answer = daemon.take_owner_desktop_draft(message)
        assert expected in answer['message'] and 'private-cursor' not in answer['message']
        assert len(wire.calls) == (1 if case in {'partial', 'empty', 'api_disabled'} else 0)


def test_relative_calendar_days_use_owner_zone_and_local_midnights():
    start, end = calendar_window({'day_offset': 1}, at=datetime(2026, 11, 26, 2, 16, tzinfo=timezone.utc),
                                 timezone_name='America/Mexico_City')
    assert start.isoformat() == '2026-11-26T00:00:00-06:00'
    assert end.isoformat() == '2026-11-27T00:00:00-06:00'
    start, end = calendar_window({'day_offset': 1}, at=datetime(2026, 10, 31, 15, tzinfo=timezone.utc),
                                 timezone_name='America/New_York')
    assert start.isoformat() == '2026-11-01T00:00:00-04:00'
    assert end.isoformat() == '2026-11-02T00:00:00-05:00'


def test_owner_worker_resolves_relative_calendar_window(f, tmp_path):
    from zoneinfo import ZoneInfo

    connect_calendar(f)
    zone = ZoneInfo('America/Mexico_City')
    f.accounts.users.set_timezone(zone.key, f.owner.actor_user_id)
    local_day = datetime.now(zone).date()
    wire = Wire({**PAGE, 'items': []})
    with calendar_worker(f, tmp_path, wire, ScriptedProvider(arguments={'day_offset': 1})) as (daemon, queue, _, _):
        message = queue.enqueue_trusted_local_owner(prompt='What is on my Google Calendar tomorrow?', peer_uid=os.getuid())
        assert worker_status(lambda: daemon.owner_message_status(message)) == 'ready'
        answer = daemon.take_owner_desktop_draft(message)
        assert answer['private_calendar'] is True and 'No events were returned' in answer['message']
        params = wire.calls[0][1]['params']
        assert params['timeMin'] == datetime.combine(local_day + timedelta(days=1), datetime.min.time(), zone).isoformat()
        assert params['timeMax'] == datetime.combine(local_day + timedelta(days=2), datetime.min.time(), zone).isoformat()


@pytest.mark.parametrize('timing', ['during_read', 'before_pickup', 'during_pickup'])
def test_calendar_revocation_withholds_observed_events(f, tmp_path, monkeypatch, timing):
    connect_calendar(f)
    wire = Wire(PAGE)
    def revoke():
        f.accounts.revoke_grant(f.owner, grant_id='grant', expected_revision=1)
    if timing == 'during_read':
        wire.after = revoke
    with calendar_worker(f, tmp_path, wire, ScriptedProvider()) as (daemon, queue, _, _):
        message = queue.enqueue_trusted_local_owner(prompt=GOAL, peer_uid=os.getuid())
        status = worker_status(lambda: daemon.owner_message_status(message))
        if timing == 'during_read':
            assert status == 'failed'
        else:
            assert status == 'ready'
            if timing == 'before_pickup':
                revoke()
            else:
                from alphonse.agent_v2.core.intelligence.simple_loop import owner_read_recovery
                restore = owner_read_recovery.restore_owner_read_observations
                def restore_then_revoke(*args, **kwargs):
                    restored = restore(*args, **kwargs)
                    revoke()
                    return restored
                monkeypatch.setattr(owner_read_recovery, 'restore_owner_read_observations', restore_then_revoke)
            with pytest.raises((OwnerRecoveryUnavailable, ValueError, RuntimeError)):
                daemon.take_owner_desktop_draft(message)
        assert len(wire.calls) == 1
