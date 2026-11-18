"""Disposable owner worker tests for an atomic, local one-time reminder."""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.owner_desktop_reminders import OwnerDesktopReminders
from alphonse.agent_v2.runtime import build_runtime_host
from alphonse.agent_v2.users import V2UserStore


def worker(tmp_path, monkeypatch, *, use_clock=True, goal=None, args=None, interrupt=False, on_commit=None, decision_override=None):
    due = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(microsecond=0).isoformat()
    goal = goal or f'Please remind me to call the plumber at {due}'
    args = args or {'message': 'call the plumber', 'due_at': due, 'request_quote': goal}
    holder = {}
    class Provider:
        supports_image_input = False
        calls = 0
        def generate_json(self, request):
            self.calls += 1
            if decision_override is not None:
                decision = decision_override
            elif use_clock and self.calls == 1:
                decision = {'kind': 'tool', 'tool_id': 'native.owner_clock', 'arguments': {}, 'text': None}
            else:
                if interrupt:
                    holder['daemon'].trigger_killswitch(actor_user_id=holder['owner'].user_id)
                decision = {'kind': 'tool', 'tool_id': 'native.owner_reminder_create', 'arguments': args, 'text': None}
            return InferenceResult(json_value=decision, raw_response=json.dumps(decision),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    path = tmp_path / 'fixture.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    owner = users.update_user(owner.user_id, role='owner')
    users.set_timezone('UTC', owner.user_id)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversations = SQLiteConversationStore(path)
    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition()); tools.register(build_read_memory_source_tool_definition())
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=conversations, outbox=SQLiteOutboundStore(path), tools=tools,
        inference=InferenceRouter(provider=Provider(), default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='fixture', experimental_owner_only=True)
    holder.update(daemon=daemon, owner=owner)
    if on_commit is not None:
        original_commit = daemon.kill_switch.commit_if_active
        def commit_then_notify(message_id, callback):
            result = original_commit(message_id, callback)
            on_commit(daemon, owner)
            return result
        monkeypatch.setattr(daemon.kill_switch, 'commit_if_active', commit_then_notify)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    message_id = queue.enqueue_trusted_local_owner(prompt=goal, peer_uid=os.getuid())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = daemon.owner_message_status(message_id)
        if status not in {'pending', 'processing'}:
            break
        time.sleep(.01)
    return daemon, owner, message_id, status, path, conversations


def test_owner_conversation_creates_exactly_one_reminder_and_confirms(tmp_path, monkeypatch):
    daemon, owner, message_id, status, path, conversations = worker(tmp_path, monkeypatch)
    try:
        assert status == 'ready'
        reminders = OwnerDesktopReminders(str(path)).list_open(owner.user_id)
        assert len(reminders) == 1 and reminders[0]['message'] == 'call the plumber'
        draft = daemon.take_owner_desktop_draft(message_id)
        assert draft['message'].startswith('Reminder saved: call the plumber')
        assert 'Desktop is open' in draft['message']
        daemon.acknowledge_owner_desktop_draft(message_id, draft['acknowledgement_token'])
        assert conversations.list(owner_user_id=owner.user_id)[-1].content == draft['message']
        assert len(OwnerDesktopReminders(str(path)).list_open(owner.user_id)) == 1
    finally:
        daemon.stop()


def test_owner_conversation_creates_tomorrow_reminder(tmp_path, monkeypatch):
    tomorrow = (datetime.now(timezone.utc) + timedelta(days=1)).date()
    due = datetime(tomorrow.year, tomorrow.month, tomorrow.day, 9, tzinfo=timezone.utc).isoformat()
    goal = 'Remind me to call the plumber tomorrow at 9 am'
    args = {'message': 'call the plumber', 'due_at': due, 'request_quote': goal}
    daemon, owner, message_id, status, path, _ = worker(tmp_path, monkeypatch, goal=goal, args=args)
    try:
        assert status == 'ready'
        reminders = OwnerDesktopReminders(str(path)).list_open(owner.user_id)
        assert len(reminders) == 1 and reminders[0]['due_at'] == due
        assert daemon.take_owner_desktop_draft(message_id)['message'].startswith('Reminder saved: call the plumber')
    finally:
        daemon.stop()


def test_one_minute_relative_reminder_can_be_saved(tmp_path, monkeypatch):
    due = (datetime.now(timezone.utc) + timedelta(minutes=1)).replace(microsecond=0).isoformat()
    goal = 'Remind me to drink water in 1 minute'
    args = {'message': 'drink water', 'due_at': due, 'request_quote': goal}
    daemon, owner, message_id, status, path, _ = worker(tmp_path, monkeypatch, goal=goal, args=args)
    try:
        assert status == 'ready'
        reminders = OwnerDesktopReminders(str(path)).list_open(owner.user_id)
        assert len(reminders) == 1 and reminders[0]['due_at'] == due
        assert daemon.take_owner_desktop_draft(message_id)['message'].startswith('Reminder saved: drink water')
    finally:
        daemon.stop()


def test_elapsed_relative_reminder_is_not_saved(tmp_path, monkeypatch):
    # Simulate a valid interpretation whose due time has elapsed by commit.
    # The atomic host must independently reject it even after validation.
    from alphonse.agent_v2.core.intelligence.simple_loop import owner_reminder_create
    due = (datetime.now(timezone.utc) - timedelta(seconds=1)).replace(microsecond=0).isoformat()
    goal = 'Remind me to drink water in 1 minute'
    args = {'message': 'drink water', 'due_at': due, 'request_quote': goal}
    monkeypatch.setattr(owner_reminder_create, 'validate_reminder_request',
                        lambda *_args, **_kwargs: {'message': 'drink water', 'due_at': due})
    daemon, owner, _, status, path, _ = worker(tmp_path, monkeypatch, goal=goal, args=args)
    try:
        assert status != 'ready'
        assert OwnerDesktopReminders(str(path)).list_open(owner.user_id) == []
    finally:
        daemon.stop()


@pytest.mark.parametrize('mode', ['no_clock', 'negated', 'invented', 'cancel'])
def test_reminder_rejects_missing_authority_or_clock_without_effect(tmp_path, monkeypatch, mode):
    goal = 'Do not remind me to call the plumber' if mode == 'negated' else None
    args = {'message': 'invented secret', 'due_at': (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            'request_quote': 'remind me to call the plumber'} if mode == 'invented' else None
    daemon, owner, _, status, path, _ = worker(tmp_path, monkeypatch, use_clock=mode != 'no_clock',
                                              goal=goal, args=args, interrupt=mode == 'cancel')
    try:
        assert status != 'ready'
        assert OwnerDesktopReminders(str(path)).list_open(owner.user_id) == []
    finally:
        daemon.stop()


@pytest.mark.parametrize('template', [
    'Should you remind me to call the plumber at {due}?',
    'If I am free, remind me to call the plumber at {due}',
    'Remind me to call the plumber every day at {due}',
    'Remind me to call the plumber only after I confirm at {due}',
    'Remind me to call the plumber after tomorrow at {due}',
    'Recuérdame llamar al plomero cada día a las {due}',
])
def test_ambiguous_or_recurring_reminder_never_creates_an_effect(tmp_path, monkeypatch, template):
    due = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(microsecond=0).isoformat()
    goal = template.format(due=due)
    spanish = 'Recuérdame' in goal
    args = {'message': 'llamar al plomero' if spanish else 'call the plumber',
            'due_at': due, 'request_quote': goal}
    daemon, owner, _, status, path, _ = worker(tmp_path, monkeypatch, goal=goal, args=args)
    try:
        assert status != 'ready'
        assert OwnerDesktopReminders(str(path)).list_open(owner.user_id) == []
    finally:
        daemon.stop()


@pytest.mark.parametrize('kind,text', [
    ('answer', "I've set your reminder."),
    ('question', "I've set your reminder. Is that okay?"),
])
@pytest.mark.parametrize('goal', [
    'Can you remind me to call the plumber in 10 minutes?',
    'I am busy. Please remind me to call the plumber in 10 minutes.',
])
def test_model_cannot_claim_reminder_without_committed_effect(tmp_path, monkeypatch, kind, text, goal):
    decision = {'kind': kind, 'tool_id': None, 'arguments': None, 'text': text}
    daemon, owner, _, status, path, _ = worker(tmp_path, monkeypatch, goal=goal, decision_override=decision)
    try:
        assert status != 'ready'
        assert OwnerDesktopReminders(str(path)).list_open(owner.user_id) == []
    finally:
        daemon.stop()


def test_reminder_clarification_is_available_without_effect(tmp_path, monkeypatch):
    decision = {'kind': 'question', 'tool_id': None, 'arguments': None,
                'text': 'What time would you like the reminder?'}
    daemon, owner, message_id, status, path, _ = worker(tmp_path, monkeypatch, decision_override=decision)
    try:
        assert status == 'ready'
        assert daemon.take_owner_desktop_draft(message_id)['message'] == decision['text']
        assert OwnerDesktopReminders(str(path)).list_open(owner.user_id) == []
    finally:
        daemon.stop()


def test_reminder_cannot_take_content_from_unrelated_prior_clause(tmp_path, monkeypatch):
    due = (datetime.now(timezone.utc) + timedelta(hours=1)).replace(microsecond=0).isoformat()
    goal = f'I took medicine. Remind me to check mail at {due}'
    args = {'message': 'took medicine', 'due_at': due, 'request_quote': goal}
    daemon, owner, _, status, path, _ = worker(tmp_path, monkeypatch, goal=goal, args=args)
    try:
        assert status != 'ready'
        assert OwnerDesktopReminders(str(path)).list_open(owner.user_id) == []
    finally:
        daemon.stop()


@pytest.mark.parametrize('content', ['call the plumber only after I confirm',
                                     'call the plumber after tomorrow'])
def test_relative_time_does_not_override_temporal_qualifier(tmp_path, monkeypatch, content):
    due = (datetime.now(timezone.utc) + timedelta(minutes=10)).replace(microsecond=0).isoformat()
    goal = f'Remind me to {content} in 10 minutes.'
    args = {'message': content, 'due_at': due, 'request_quote': goal}
    daemon, owner, _, status, path, _ = worker(tmp_path, monkeypatch, goal=goal, args=args)
    try:
        assert status != 'ready'
        assert OwnerDesktopReminders(str(path)).list_open(owner.user_id) == []
    finally:
        daemon.stop()


def test_reminder_creation_and_checkpoint_roll_back_together(tmp_path, monkeypatch):
    original = OwnerDesktopReminders.create_locked
    def interrupted(conn, *args, **kwargs):
        original(conn, *args, **kwargs)
        raise RuntimeError('synthetic interruption after insert')
    monkeypatch.setattr(OwnerDesktopReminders, 'create_locked', staticmethod(interrupted))
    daemon, owner, _, status, path, _ = worker(tmp_path, monkeypatch)
    try:
        assert status != 'ready'
        assert OwnerDesktopReminders(str(path)).list_open(owner.user_id) == []
    finally:
        daemon.stop()


def test_committed_reminder_draft_recovers_without_creating_again(tmp_path, monkeypatch):
    daemon, owner, message_id, status, path, _ = worker(tmp_path, monkeypatch)
    try:
        assert status == 'ready'
        before = OwnerDesktopReminders(str(path)).list_open(owner.user_id)
        daemon._lock_file = object()
        daemon._owner_checkpoints.pop(message_id)
        daemon._owner_statuses.pop(message_id)
        assert daemon._restore_owner_ready(message_id)
        draft = daemon.take_owner_desktop_draft(message_id)
        assert draft['message'].startswith('Reminder saved: call the plumber')
        assert OwnerDesktopReminders(str(path)).list_open(owner.user_id) == before
    finally:
        daemon.stop()


def test_cancel_after_atomic_effect_commit_cannot_hide_confirmation(tmp_path, monkeypatch):
    observed = []
    def commit_then_cancel(daemon, owner):
        observed.append(daemon.trigger_killswitch(actor_user_id=owner.user_id)['status'])
    daemon, owner, message_id, status, path, _ = worker(tmp_path, monkeypatch, on_commit=commit_then_cancel)
    try:
        assert observed == ['effect_already_committed']
        assert status == 'ready'
        assert len(OwnerDesktopReminders(str(path)).list_open(owner.user_id)) == 1
        assert daemon.take_owner_desktop_draft(message_id)['message'].startswith('Reminder saved:')
    finally:
        daemon.stop()


@pytest.mark.parametrize('acknowledge', [False, True])
def test_reminder_offered_before_confirmation_is_still_truthful(tmp_path, monkeypatch, acknowledge):
    daemon, owner, message_id, status, path, _ = worker(tmp_path, monkeypatch)
    try:
        assert status == 'ready'
        store = OwnerDesktopReminders(str(path))
        reminder = store.take_due(owner.user_id, now=datetime.now(timezone.utc) + timedelta(hours=2))['reminder']
        assert reminder is not None
        if acknowledge:
            store.acknowledge(owner.user_id, reminder['id'], reminder['acknowledgement_token'])
        assert daemon.take_owner_desktop_draft(message_id)['message'].startswith('Reminder saved:')
    finally:
        daemon.stop()


def test_relative_reminder_time_must_match_owner_clock():
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_reminder_create import validate_reminder_request
    now = datetime(2026, 11, 15, 12, tzinfo=timezone.utc)
    task = SimpleNamespace(goal='Please remind me to call the plumber in 10 minutes')
    args = {'message': 'call the plumber', 'request_quote': task.goal,
            'due_at': (now + timedelta(minutes=10)).isoformat()}
    assert validate_reminder_request(task, args, 'UTC', observed_at=now.isoformat())['due_at'] == args['due_at']
    args['due_at'] = (now + timedelta(hours=10)).isoformat()
    with pytest.raises(ValueError, match='owner_reminder_time_invalid'):
        validate_reminder_request(task, args, 'UTC', observed_at=now.isoformat())
    task.goal += ' and in 30 minutes'
    args['request_quote'] = task.goal
    with pytest.raises(ValueError, match='owner_reminder_time_invalid'):
        validate_reminder_request(task, args, 'UTC', observed_at=now.isoformat())


@pytest.mark.parametrize('goal,message,due', [
    ('Remind me to call the plumber tomorrow at 9 am', 'call the plumber', '2026-11-16T09:00:00-06:00'),
    ('Please remind me to check mail today at 12:30 pm.', 'check mail', '2026-11-15T12:30:00-06:00'),
])
def test_owner_local_calendar_time_matches_requested_day_and_time(goal, message, due):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_reminder_create import validate_reminder_request
    task = SimpleNamespace(goal=goal)
    args = {'message': message, 'request_quote': goal, 'due_at': due}
    observed = '2026-11-15T14:00:00+00:00'
    result = validate_reminder_request(task, args, 'America/Mexico_City', observed_at=observed)
    assert result['due_at'] == datetime.fromisoformat(due).astimezone(timezone.utc).isoformat()
    args['due_at'] = (datetime.fromisoformat(due) + timedelta(hours=1)).isoformat()
    with pytest.raises(ValueError, match='owner_reminder_time_invalid'):
        validate_reminder_request(task, args, 'America/Mexico_City', observed_at=observed)


@pytest.mark.parametrize('goal,due,observed', [
    ('Remind me to call the plumber after I confirm tomorrow at 9 am', '2026-11-16T09:00:00-06:00', '2026-11-15T14:00:00+00:00'),
    ('Remind me to call the plumber tomorrow at 9 am', '2026-11-15T09:00:00-06:00', '2026-11-15T14:00:00+00:00'),
    ('Remind me to call the plumber tomorrow at 9 am', '2026-11-16T09:00:00-06:00', None),
    ('Remind me to call the plumber tomorrow at 13 pm', '2026-11-16T13:00:00-06:00', '2026-11-15T14:00:00+00:00'),
])
def test_calendar_time_rejects_conditions_or_invalid_interpretation(goal, due, observed):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_reminder_create import validate_reminder_request
    args = {'message': 'call the plumber', 'request_quote': goal, 'due_at': due}
    with pytest.raises(ValueError):
        validate_reminder_request(SimpleNamespace(goal=goal), args, 'America/Mexico_City', observed_at=observed)


def test_calendar_time_rejects_dst_repeated_wall_time():
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_reminder_create import validate_reminder_request
    goal = 'Remind me to call the plumber tomorrow at 1:30 am'
    args = {'message': 'call the plumber', 'request_quote': goal, 'due_at': '2026-11-01T01:30:00-04:00'}
    with pytest.raises(ValueError, match='owner_reminder_time_invalid'):
        validate_reminder_request(SimpleNamespace(goal=goal), args, 'America/New_York', observed_at='2026-10-31T14:00:00+00:00')


def test_reminder_confirmation_survives_interruption_before_handoff(tmp_path, monkeypatch):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_atomic_checkpoint import OwnerAtomicCheckpoint
    original = OwnerAtomicCheckpoint.mark_ready_handoff
    def interrupted(_self):
        raise RuntimeError('synthetic interruption before normal handoff')
    monkeypatch.setattr(OwnerAtomicCheckpoint, 'mark_ready_handoff', interrupted)
    daemon, owner, message_id, _, path, _ = worker(tmp_path, monkeypatch)
    try:
        assert len(OwnerDesktopReminders(str(path)).list_open(owner.user_id)) == 1
        monkeypatch.setattr(OwnerAtomicCheckpoint, 'mark_ready_handoff', original)
        daemon._lock_file = object()
        daemon._owner_checkpoints.pop(message_id, None)
        daemon._owner_statuses.pop(message_id, None)
        assert daemon._restore_owner_ready(message_id)
        assert daemon.take_owner_desktop_draft(message_id)['message'].startswith('Reminder saved:')
        assert len(OwnerDesktopReminders(str(path)).list_open(owner.user_id)) == 1
    finally:
        daemon.stop()
