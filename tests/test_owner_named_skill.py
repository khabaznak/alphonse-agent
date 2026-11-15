"""User-directed skill guidance must stay actor-scoped and revision-pinned."""
from types import SimpleNamespace
from dataclasses import replace

import pytest

from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.intelligence.simple_loop.owner_named_skill import (
    OWNER_NAMED_SKILL_TOOL_ID, build_owner_named_skill_tool_definition,
    execute_owner_named_skill, explicit_requested_skill, validate_owner_named_skill,
)
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.skills import SkillStore


class Resources:
    def __init__(self):
        self.private = set()

    def access_records(self):
        return [SimpleNamespace(skill_id=x, storage_kind='skill') for x in self.private]

    def can_use(self, record, actor):
        return False


@pytest.fixture
def setup(tmp_path, monkeypatch):
    root = tmp_path / 'skills'
    root.mkdir()
    for name in ('meal-planning', 'private'):
        path = root / name
        path.mkdir()
        (path / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: Example guidance\n---\nFull {name} guidance.\n')
    store = SkillStore(root)
    monkeypatch.setattr(store, '_bundled_skills_dir', lambda: tmp_path / 'none')
    monkeypatch.setattr(store, 'can_use', lambda identity, actor: actor == 'owner' and identity != 'skill:private')
    resources = Resources()
    task = TaskState(task_id='synthetic', user='owner', goal='Please use the meal-planning skill to suggest a menu.')
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), skill_store=store,
                                   artifact_resources=resources, cancellation_checker=lambda: False)
    return store, resources, context


def test_user_named_request_loads_complete_authorized_instructions(setup):
    _, _, context = setup
    payload = execute_owner_named_skill({'skill_id': 'skill:meal-planning'}, context=context)
    assert payload['instructions'] == 'Full meal-planning guidance.'
    assert payload['selection_reason'] == 'explicit_user_request'
    assert payload['execution_available'] is False
    assert validate_owner_named_skill('owner', {'skill_id': 'skill:meal-planning'}, payload, context=context)


@pytest.mark.parametrize('goal', [
    'Do not use the meal-planning skill.',
    'I heard someone say use the meal-planning skill.',
    'What is the meal-planning skill?',
    'Please use the private skill.',
    'Use the meal-planning skill? No, do not use it.',
    'Use the meal-planning skill only if I confirm later.',
    'Use the meal-planning skill. Actually, skip it.',
    'Use the meal-planning skill. Forget that request.',
    'Use the meal-planning skill but disregard that request',
])
def test_incidental_negated_or_private_references_cannot_activate(setup, goal):
    _, _, context = setup
    context.task.goal = goal
    with pytest.raises(PermissionError):
        execute_owner_named_skill({'skill_id': 'skill:meal-planning'}, context=context)


def test_model_cannot_substitute_an_unrequested_skill(setup):
    _, _, context = setup
    with pytest.raises(PermissionError):
        execute_owner_named_skill({'skill_id': 'skill:private'}, context=context)
    rows = [{'skill_id': 'a', 'name': 'same'}, {'skill_id': 'b', 'name': 'same'}]
    assert explicit_requested_skill('Use the same skill.', rows) == ''


def test_revocation_or_package_edit_invalidates_observation(setup, monkeypatch):
    store, resources, context = setup
    args = {'skill_id': 'skill:meal-planning'}
    payload = execute_owner_named_skill(args, context=context)
    resources.private.add('skill:meal-planning')
    assert not validate_owner_named_skill('owner', args, payload, context=context)
    resources.private.clear()
    (store.skills_dir / 'meal-planning' / 'SKILL.md').write_text(
        '---\nname: meal-planning\ndescription: Example guidance\n---\nChanged guidance.\n')
    assert not validate_owner_named_skill('owner', args, payload, context=context)
    monkeypatch.setattr(store, 'can_use', lambda *_: False)
    assert not validate_owner_named_skill('owner', args, payload, context=context)


def test_cancelled_or_wrong_actor_is_denied(setup):
    _, _, context = setup
    args = {'skill_id': 'skill:meal-planning'}
    payload = execute_owner_named_skill(args, context=context)
    assert not validate_owner_named_skill('other', args, payload, context=context)
    cancelled = replace(context, cancellation_checker=lambda: True)
    assert not validate_owner_named_skill('owner', args, payload, context=cancelled)

def test_invalid_or_oversized_package_is_generically_unavailable(setup, monkeypatch):
    store, _, context = setup
    args = {'skill_id': 'skill:meal-planning'}
    broken = store.skills_dir / 'broken'
    broken.mkdir()
    (broken / 'SKILL.md').write_text('invalid')
    assert execute_owner_named_skill(args, context=context) == {'schema_version': 1, 'status': 'unavailable'}
    (broken / 'SKILL.md').unlink()
    broken.rmdir()
    from alphonse.agent_v2.skills import MAX_SKILL_INSTRUCTIONS_LENGTH
    (store.skills_dir / 'meal-planning' / 'SKILL.md').write_text(
        '---\nname: meal-planning\ndescription: Example guidance\n---\n' + 'x' * (MAX_SKILL_INSTRUCTIONS_LENGTH + 1))
    assert execute_owner_named_skill(args, context=context) == {'schema_version': 1, 'status': 'unavailable'}

@pytest.mark.parametrize('change', ['none', 'revoke', 'revision', 'skip', 'reversal', 'conditional', 'skip_later', 'forget_later', 'same_sentence_reversal', 'name_only', 'unavailable', 'false_question'])
@pytest.mark.parametrize('restore', [False, True])
def test_owner_worker_uses_named_skill_and_checks_at_delivery(tmp_path, monkeypatch, change, restore):
    """The actual worker exposes instructions, then rechecks authorization and revision."""
    import json
    import os
    import time
    from pathlib import Path
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
    from alphonse.agent_v2.core.inference.router import InferenceRouter
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore

    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setenv('ALPHONSE_V2_DB_PATH', str(tmp_path / 'fixture.sqlite'))
    root = tmp_path / 'skills'
    root.mkdir()
    package = root / 'meal-planning'
    package.mkdir()
    skill_file = package / 'SKILL.md'
    skill_file.write_text('---\nname: meal-planning\ndescription: Meal planning\n---\nSuggest seasonal vegetables.\n')
    store = SkillStore(root)
    monkeypatch.setattr(store, '_bundled_skills_dir', lambda: tmp_path / 'no-bundled-skills')
    access = {'allowed': change not in {'unavailable', 'false_question'}}
    monkeypatch.setattr(store, 'can_use', lambda skill_id, actor: access['allowed'])

    class Provider:
        supports_image_input = False
        def __init__(self):
            self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            action = ({'kind': 'tool', 'tool_id': OWNER_NAMED_SKILL_TOOL_ID,
                       'arguments': {'skill_id': 'meal-planning' if change == 'name_only' else 'skill:meal-planning'}, 'text': None}
                      if len(self.requests) == 1 and change != 'skip' else
                      {'kind': 'question', 'tool_id': None, 'arguments': None, 'text': "I couldn't use the requested skill. Could you check its installation and access?" if change == 'unavailable' else 'I used your skill. Is that okay?'}
                      if change in {'unavailable', 'false_question'} else
                      {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Try seasonal vegetables.'})
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = tmp_path / 'fixture.sqlite'
    users = V2UserStore(path)
    users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition())
    tools.register(build_read_memory_source_tool_definition())
    provider = Provider()
    runtime = build_runtime_host(user_store=users, skill_store=store, messages=queue,
        question_store=SQLiteQuestionStore(path), conversation_store=SQLiteConversationStore(path),
        outbox=SQLiteOutboundStore(path), tools=tools,
        inference=InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline')),
        coding_skill_service=object(), experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        prompt = {'reversal': 'Use the meal-planning skill? No, do not use it.',
                  'conditional': 'Use the meal-planning skill only if I confirm later.',
                  'skip_later': 'Use the meal-planning skill. Actually, skip it.',
                  'forget_later': 'Use the meal-planning skill. Forget that request.',
                  'same_sentence_reversal': 'Use the meal-planning skill but disregard that request'}.get(change, 'Use the meal-planning skill to suggest a menu.')
        message_id = queue.enqueue_trusted_local_owner(prompt=prompt, peer_uid=os.getuid())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = daemon.owner_message_status(message_id)
            if status not in {'pending', 'processing'}:
                break
            time.sleep(.01)
        if change in {'skip', 'reversal', 'conditional', 'skip_later', 'forget_later', 'same_sentence_reversal', 'false_question'}:
            assert status != 'ready'
            return
        if change == 'unavailable':
            assert '"status": "unavailable"' in provider.requests[1].prompt
            assert status == 'ready', status
            draft = daemon.take_owner_desktop_draft(message_id)
            assert "couldn't use" in draft['message']
            return
        assert status == 'ready', status
        assert 'Suggest seasonal vegetables.' in provider.requests[1].prompt
        if restore:
            daemon._lock_file = object()
            daemon._owner_checkpoints.pop(message_id)
            daemon._owner_statuses.pop(message_id)
            assert daemon._restore_owner_ready(message_id)
        if change == 'revoke':
            access['allowed'] = False
        elif change == 'revision':
            skill_file.write_text('---\nname: meal-planning\ndescription: Meal planning\n---\nChanged guidance.\n')
        if change in {'none', 'name_only'}:
            assert daemon.take_owner_desktop_draft(message_id)
        else:
            with pytest.raises((ValueError, RuntimeError, OwnerRecoveryUnavailable)):
                daemon.take_owner_desktop_draft(message_id)
    finally:
        daemon.stop()
