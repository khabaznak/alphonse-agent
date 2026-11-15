"""Actor-scoped skill discovery is metadata only, with fresh access checks."""
from types import SimpleNamespace
import json
import time

import pytest

from alphonse.agent_v2.core.core import ToolExecutionContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.registry.native.skill_catalog import (
    SKILL_CATALOG_TOOL_ID, execute_skill_catalog, validate_skill_catalog_observation,
)
from alphonse.agent_v2.skills import SkillStore


def package(root, name, description, body='Private instruction sentinel'):
    path = root / name
    path.mkdir()
    (path / 'SKILL.md').write_text(f'---\nname: {name}\ndescription: {description}\n---\n{body}\n')
    return path


class Resources:
    def __init__(self):
        self.private_ids = set()

    def access_records(self):
        return [SimpleNamespace(skill_id=skill_id, storage_kind='skill') for skill_id in self.private_ids]

    def can_use(self, record, actor):
        return False


@pytest.fixture
def setup(tmp_path, monkeypatch):
    skills = tmp_path / 'skills'
    skills.mkdir()
    package(skills, 'visible', 'Public expertise')
    package(skills, 'hidden', 'Private description sentinel')
    store = SkillStore(skills)
    monkeypatch.setattr(store, '_bundled_skills_dir', lambda: tmp_path / 'no-bundled-skills')
    monkeypatch.setattr(store, 'can_use', lambda skill_id, actor: actor == 'owner' and skill_id != 'skill:hidden')
    resources = Resources()
    task = TaskState(task_id='fixture', user='owner')
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), skill_store=store,
                                   artifact_resources=resources, cancellation_checker=lambda: False)
    return store, resources, context


def test_metadata_only_filters_access_and_uses_standard_registry(setup):
    store, _, context = setup
    result = execute_skill_catalog({}, context=context)
    assert result == {'schema_version': 1, 'status': 'found', 'skills': [
        {'skill_id': 'skill:visible', 'name': 'visible', 'description': 'Public expertise'}]}
    assert 'sentinel' not in str(result).lower()
    assert build_native_tool_registry(skill_store=store).get(SKILL_CATALOG_TOOL_ID).read_only is True


def test_observation_rechecks_changed_permissions_metadata_and_legacy_acl(setup, monkeypatch):
    store, resources, context = setup
    original = execute_skill_catalog({}, context=context)
    valid = lambda: validate_skill_catalog_observation('owner', SKILL_CATALOG_TOOL_ID, {}, original, context=context)
    assert valid()
    resources.private_ids.add('skill:visible')
    assert not valid()
    resources.private_ids.clear()
    (store.skills_dir / 'visible' / 'SKILL.md').write_text('---\nname: visible\ndescription: Changed expertise\n---\nBody\n')
    assert not valid()
    monkeypatch.setattr(store, 'can_use', lambda *_: False)
    assert not valid()


def test_bad_actor_cancel_and_oversize_never_return_partial_catalog(setup, monkeypatch):
    store, resources, context = setup
    payload = execute_skill_catalog({}, context=context)
    assert not validate_skill_catalog_observation('other', SKILL_CATALOG_TOOL_ID, {}, payload, context=context)
    with pytest.raises(ValueError):
        execute_skill_catalog({'query': 'hidden'}, context=context)
    cancelled = ToolExecutionContext(task=context.task, messages=context.messages, skill_store=store,
                                     artifact_resources=resources, cancellation_checker=lambda: True)
    with pytest.raises(PermissionError):
        execute_skill_catalog({}, context=cancelled)
    record = store.get('skill:visible')
    monkeypatch.setattr(store, 'list_skills', lambda: [record] * 101)
    with pytest.raises(ValueError, match='skill_catalog_too_large'):
        execute_skill_catalog({}, context=context)


def test_invalid_package_refuses_incomplete_listing_without_revealing_path(setup):
    store, _, context = setup
    broken = store.skills_dir / 'broken'
    broken.mkdir()
    (broken / 'SKILL.md').write_text('not a standard skill package')
    with pytest.raises(ValueError, match='skill_catalog_incomplete') as failure:
        execute_skill_catalog({}, context=context)
    assert str(broken) not in str(failure.value)
    assert 'broken' not in str(failure.value)


@pytest.mark.parametrize('restore', [False, True])
def test_owner_worker_catalog_delivery_rechecks_visibility(tmp_path, monkeypatch, restore):
    """A ready answer cannot reveal a catalog after package access changes."""
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    from alphonse.agent_v2.core.inference.models import InferenceResult, ModelProfile
    from alphonse.agent_v2.core.inference.router import InferenceRouter
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_recovery import OwnerRecoveryUnavailable
    from alphonse.agent_v2.core.io.outbox import SQLiteOutboundStore
    from alphonse.agent_v2.core.messages.sqlite_queue import SQLiteMessageQueue
    from alphonse.agent_v2.core.questions import SQLiteQuestionStore
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
    from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
    from alphonse.agent_v2.core.tools.registry.native.skill_catalog import build_skill_catalog_tool_definition
    from alphonse.agent_v2.daemon import V2Daemon
    from alphonse.agent_v2.runtime import build_runtime_host
    from alphonse.agent_v2.users import V2UserStore
    from pathlib import Path

    # Every fallback path is disposable; no real user database or package root.
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: tmp_path))
    monkeypatch.setenv('ALPHONSE_V2_DB_PATH', str(tmp_path / 'fixture.sqlite'))
    root = tmp_path / 'skills'
    root.mkdir()
    package(root, 'visible', 'Public expertise')
    package(root, 'hidden', 'Private description sentinel')
    store = SkillStore(root)
    monkeypatch.setattr(store, '_bundled_skills_dir', lambda: tmp_path / 'no-bundled-skills')
    can_use = {'value': True}
    monkeypatch.setattr(store, 'can_use', lambda skill_id, actor: can_use['value'] and skill_id != 'skill:hidden')

    class Provider:
        supports_image_input = False
        def __init__(self):
            self.requests = []
        def generate_json(self, request):
            self.requests.append(request)
            action = ({'kind': 'tool', 'tool_id': SKILL_CATALOG_TOOL_ID, 'arguments': {}, 'text': None}
                      if len(self.requests) == 1 else
                      {'kind': 'answer', 'tool_id': None, 'arguments': None, 'text': 'Visible skill available.'})
            return InferenceResult(json_value=action, raw_response=json.dumps(action),
                                   usage={'input_tokens': 100, 'output_tokens': 20})

    path = tmp_path / 'fixture.sqlite'
    users = V2UserStore(path)
    users.onboard(display_name='Synthetic owner', users_root=tmp_path / 'users')
    queue = SQLiteMessageQueue(path, experimental_claim_fencing=True)
    conversation = SQLiteConversationStore(path)
    tools = InMemoryToolRegistry()
    tools.register(build_search_memory_tool_definition())
    tools.register(build_read_memory_source_tool_definition())
    tools.register(build_skill_catalog_tool_definition())
    provider = Provider()
    router = InferenceRouter(provider=provider, default_profile=ModelProfile('test', 'fixture', 'offline'))
    runtime = build_runtime_host(user_store=users, skill_store=store, messages=queue,
        question_store=SQLiteQuestionStore(path), conversation_store=conversation,
        outbox=SQLiteOutboundStore(path), tools=tools, inference=router, coding_skill_service=object(),
        experimental_owner_read=True, owner_selected_draft_sink=lambda *_: None)
    daemon = V2Daemon(runtime, daemon_id='fixture', experimental_owner_only=True)
    monkeypatch.setattr(daemon, '_acquire_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon, '_release_single_instance_lock', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'start', lambda: None)
    monkeypatch.setattr(daemon.ipc, 'stop', lambda: None)
    daemon.start()
    try:
        message_id = queue.enqueue_trusted_local_owner(prompt='What skills are installed?', peer_uid=__import__('os').getuid())
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            status = daemon.owner_message_status(message_id)
            if status not in {'pending', 'processing'}:
                break
            time.sleep(.01)
        assert status == 'ready', status
        assert 'Public expertise' in provider.requests[1].prompt
        assert 'Private description sentinel' not in provider.requests[1].prompt
        assert 'Private instruction sentinel' not in provider.requests[1].prompt
        if restore:
            daemon._lock_file = object()  # Test stands in for exclusive worker lock.
            daemon._owner_checkpoints.pop(message_id)
            daemon._owner_statuses.pop(message_id)
            assert daemon._restore_owner_ready(message_id)
        can_use['value'] = False
        with pytest.raises((ValueError, RuntimeError, OwnerRecoveryUnavailable)):
            daemon.take_owner_desktop_draft(message_id)
    finally:
        daemon.stop()
