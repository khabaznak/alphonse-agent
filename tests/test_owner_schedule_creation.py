from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import json
import os
import time

import pytest

from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.intelligence.simple_loop.owner_schedule_create import validate_schedule_request
from alphonse.agent_v2.database import connect_database


def _create(store, conn, now):
    return store.create_task(owner_user_id='owner', name='Test', prompt='Original owner request',
        schedule_kind='once', run_at=(now + timedelta(minutes=5)).isoformat(),
        idempotency_key='owner-schedule:request', now=now, connection=conn)


def test_schedule_creation_participates_in_owner_transaction(tmp_path):
    path = tmp_path / 'shared.sqlite'
    store = ScheduledTaskStore(str(path))
    now = datetime.now(timezone.utc)
    with pytest.raises(RuntimeError):
        with connect_database(path) as conn:
            conn.execute('BEGIN IMMEDIATE')
            _create(store, conn, now)
            raise RuntimeError('simulate interrupted owner commit')
    assert store.list_tasks(owner_user_id='owner') == []
    with connect_database(path) as conn:
        conn.execute('BEGIN IMMEDIATE')
        first = _create(store, conn, now)
        second = _create(store, conn, now)
        assert first.scheduled_task_id == second.scheduled_task_id
    assert len(store.list_tasks(owner_user_id='owner')) == 1


def test_schedule_creation_rejects_wrong_or_absent_transaction(tmp_path):
    store = ScheduledTaskStore(str(tmp_path / 'schedules.sqlite'))
    with connect_database(tmp_path / 'other.sqlite') as conn:
        with pytest.raises(ValueError, match='transaction_required'):
            _create(store, conn, datetime.now(timezone.utc))
        conn.execute('BEGIN IMMEDIATE')
        with pytest.raises(ValueError, match='store_mismatch'):
            _create(store, conn, datetime.now(timezone.utc))


def test_schedule_request_preserves_scope_and_rejects_invalid_times():
    now = datetime.now(timezone.utc)
    goal = 'Check the pantry in five minutes.'
    task = SimpleNamespace(goal=goal)
    args = {'request_quote': goal, 'run_at': (now + timedelta(minutes=5)).isoformat()}
    assert validate_schedule_request(task, args, 'UTC', observed_at=now.isoformat(), now=now) == args['run_at']
    other_offset = (now + timedelta(minutes=5)).astimezone(timezone(timedelta(hours=-6)))
    assert validate_schedule_request(task, {**args, 'run_at': other_offset.isoformat()}, 'UTC',
                                     observed_at=now.isoformat(), now=now) == other_offset.astimezone(timezone.utc).isoformat()
    with pytest.raises(ValueError):
        validate_schedule_request(task, {**args, 'request_quote': 'Do something else'}, 'UTC',
                                  observed_at=now.isoformat(), now=now)
    with pytest.raises(ValueError):
        validate_schedule_request(task, {**args, 'run_at': (now - timedelta(seconds=1)).isoformat()}, 'UTC',
                                  observed_at=now.isoformat(), now=now)


def test_owner_runtime_retains_injected_legacy_schedule_store(tmp_path):
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore

    path = tmp_path / 'owner.sqlite'
    users = V2UserStore(path)
    users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    legacy = ScheduledTaskStore(str(tmp_path / 'legacy.sqlite'))
    runtime = build_runtime_host(user_store=users, messages=SQLiteMessageQueue(path, experimental_claim_fencing=True),
        question_store=SQLiteQuestionStore(path), conversation_store=SQLiteConversationStore(path),
        outbox=SQLiteOutboundStore(path), tools=InMemoryToolRegistry(), schedule_store=legacy,
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    assert runtime.schedule_store is legacy
    assert runtime.owner_schedule_store.db_path == str(path)
    assert runtime.core.schedule_store is legacy


@pytest.mark.parametrize('clock_available,invalid_first', [(True, False), (False, False), (True, True)])
def test_owner_loop_automatically_observes_clock_and_schedules(tmp_path, monkeypatch, clock_available, invalid_first):
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
    from alphonse.agent_v2.core.inference.router import InferenceRouter
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
    from alphonse.agent_v2.core.tools.registry.native.scheduled_task import build_scheduled_task_tool_definition
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore

    goal = 'In 5 minutes, count the beans in the pantry note.'
    class Provider:
        supports_image_input = False
        def __init__(self):
            self.prompts = []
        def generate_json(self, request):
            self.prompts.append(request.prompt)
            run_at = ('2026-11-19T<TIME>-06:00' if invalid_first and len(self.prompts) == 1 else
                      (datetime.now(timezone.utc) + timedelta(minutes=5)).astimezone(timezone(timedelta(hours=-6))).isoformat())
            value = {'kind': 'tool', 'tool_id': 'native.scheduled_task', 'text': None,
                'arguments': {'request_quote': goal, 'run_at': run_at}}
            return InferenceResult(json_value=value, raw_response=json.dumps(value),
                                   usage={'input_tokens': 100, 'output_tokens': 20})
    path = tmp_path / 'fixture.sqlite'
    users = V2UserStore(path)
    owner = users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    users.set_timezone('UTC', owner.user_id)
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    tools = InMemoryToolRegistry()
    for definition in (build_search_memory_tool_definition(), build_read_memory_source_tool_definition(),
                       build_scheduled_task_tool_definition()):
        tools.register(definition)
    provider = Provider()
    runtime = build_runtime_host(user_store=users, messages=queue, question_store=SQLiteQuestionStore(path),
        conversation_store=SQLiteConversationStore(path), outbox=SQLiteOutboundStore(path), tools=tools,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='fixture', experimental_owner_only=True)
    if not clock_available:
        from alphonse.agent_v2.core.intelligence.simple_loop import owner_clock
        def unavailable(*_args, **_kwargs):
            raise ValueError('synthetic_clock_unavailable')
        monkeypatch.setattr(owner_clock, 'execute_owner_clock', unavailable)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        message_id = queue.enqueue_trusted_local_owner(prompt=goal, peer_uid=os.getuid())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and daemon.owner_message_status(message_id) in {'pending', 'processing'}:
            time.sleep(.01)
        assert len(provider.prompts) == (2 if invalid_first else 1)
        if not clock_available:
            assert daemon.owner_message_status(message_id) != 'ready'
            assert runtime.owner_schedule_store.list_tasks(owner_user_id=owner.user_id) == []
            return
        assert daemon.owner_message_status(message_id) == 'ready'
        assert len(runtime.owner_schedule_store.list_tasks(owner_user_id=owner.user_id)) == 1
        from alphonse.agent_v2.core.intelligence.simple_loop.task_budget import TaskBudget
        with queue._connect() as conn:
            raw = json.loads(conn.execute('SELECT task_state_json FROM v2_task_checkpoints WHERE task_id=?',
                                          (message_id,)).fetchone()[0])
        budget = TaskBudget.from_dict(raw['metadata']['single_loop_checkpoint']['budget'])
        assert budget.usage()['charged']['actions'] == 2  # host clock and schedule creation
        # The already committed effect remains pickable even after the clock
        # used to create it has aged beyond the live-observation window.
        from alphonse.agent_v2.core.intelligence.simple_loop import owner_clock
        real_datetime = datetime
        class LaterDatetime(real_datetime):
            @classmethod
            def now(cls, tz=None):
                return real_datetime.now(tz) + timedelta(minutes=6)
        monkeypatch.setattr(owner_clock, 'datetime', LaterDatetime)
        message = daemon.take_owner_desktop_draft(message_id)['message']
        assert 'Task scheduled for' in message and '(UTC)' in message
        assert 'Host clock observed as of:' in message
    finally:
        daemon.stop()
