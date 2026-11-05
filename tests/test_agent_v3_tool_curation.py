"""Synthetic request-state and exposure regressions; never use household data or live APIs."""
import json
import stat
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.tool_curation import request_tool_state, curate_request_tools
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings


def setup(tmp_path, probability=.5):
    skills = SkillStore(tmp_path / 'skills')
    skill = skills.create_skill('fixture-records', 'Look up local records', 'Read records.sqlite using a read-only local SQLite connection.')
    task = TaskState(task_id='fixture', user='owner', goal='Check the latest recorded schedule.',
        metadata={'v3_skill_selection': {'selected_skill_ids': [skill.skill_id]}})
    registry = InMemoryToolRegistry()
    registry.register(build_bash_tool_definition()); registry.register(build_respond_tool_definition())
    calls = []
    def transport(url, key, payload, timeout):
        calls.append(payload)
        return {'answers': {key: {'type': 'noul', 'noul': probability} for key in payload['questions']}}
    jev = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture-secret'), transport=transport)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=jev, skill_store=skills)
    return task, context, calls, skill


@pytest.mark.parametrize('probability,status', [(.05, 'empty_selection_llm_fallback'), (.5, 'used'), (.95, 'used')])
def test_negative_uncertain_and_positive_answers_have_explicit_routes(tmp_path, probability, status):
    task, context, calls, skill = setup(tmp_path, probability)
    selected = curate_request_tools(task, context.tools.list(), context)
    assert {tool.tool_id for tool in selected} == {'native.bash', 'native.respond'}
    metadata = task.metadata['v3_request_tool_curation']
    assert metadata['status'] == status
    if probability == .5:
        assert set(metadata['ambiguous_tool_ids']) == {'native.bash', 'native.respond'}
        assert metadata['rejected_tool_ids'] == []
    state = calls[0]['state']
    assert state['selected_skills'][0]['package_directory'] == str(skill.directory)
    assert state['selected_skills'][0]['workflow'] == skill.instructions
    assert state['coverage']['resource_paths_are_not_resource_contents']
    assert set(state) == {'schema_version', 'request', 'task_understanding', 'conversation', 'selected_skills', 'known_context', 'evidence', 'coverage', 'omissions'}
    assert 'plan_system_prompt' not in state and 'strategic_plan' not in state


def test_cache_binds_context_settings_and_current_request(tmp_path):
    task, context, calls, _ = setup(tmp_path)
    for _ in range(2):
        curate_request_tools(task, context.tools.list(), context)
    assert len(calls) == 1
    task.append_conversation_message(task.user, 'Also record the newly supplied value.')
    curate_request_tools(task, context.tools.list(), context)
    assert len(calls) == 2
    assert calls[-1]['state']['conversation'][-1]['text'] == 'Also record the newly supplied value.'
    context.system_one.settings = SystemOneSettings(api_key='fixture-secret', tool_selection_threshold=.9)
    curate_request_tools(task, context.tools.list(), context)
    assert len(calls) == 3


@pytest.mark.parametrize('value', [True, float('nan'), float('inf'), -1, 1.01, '0.9'])
def test_malformed_answers_are_errors_with_llm_fallback_not_negative_judgments(tmp_path, value):
    task, context, _, _ = setup(tmp_path)
    context.system_one.client = SimpleNamespace(evaluate=lambda **kwargs: {'answers': {
        key: {'type': 'noul', 'noul': value} for key in kwargs['questions']}})
    assert len(curate_request_tools(task, context.tools.list(), context)) == 2
    metadata = task.metadata['v3_request_tool_curation']
    assert metadata['status'] == 'error_llm_fallback'
    assert 'probabilities' not in metadata and metadata['exception_type'] == 'ValueError'


def test_api_failure_is_recorded_without_leaking_error_body(tmp_path):
    task, context, _, _ = setup(tmp_path)
    context.system_one = SimpleNamespace(curate_request_tools=lambda **kwargs: (_ for _ in ()).throw(RuntimeError('PRIVATE ERROR BODY')))
    assert len(curate_request_tools(task, context.tools.list(), context)) == 2
    assert task.metadata['v3_request_tool_curation']['status'] == 'error_llm_fallback'
    assert 'PRIVATE ERROR BODY' not in json.dumps(task.metadata)


def test_whole_oversized_skill_and_private_resources_never_enter_jev_payload(tmp_path):
    task, context, calls, skill = setup(tmp_path)
    resource = SimpleNamespace(skill_id=skill.skill_id, artifact_id='artifact.private')
    context.artifact_resources = SimpleNamespace(store=SimpleNamespace(list=lambda: [resource]),
        can_use=lambda *args: False, describe=lambda *args: pytest.fail('Private resource must not be read'))
    state = request_tool_state(task, context)
    assert state['selected_skills'][0]['resources'] == []
    assert 'artifact.private' not in json.dumps(state)
    task.goal = 'long complete user input ' * 5000
    assert len(curate_request_tools(task, context.tools.list(), context)) == 2
    assert task.metadata['v3_request_tool_curation']['status'] == 'incomplete_state_llm_fallback'
    assert calls == []


def test_opt_in_capture_is_exact_private_and_contains_no_credentials(tmp_path, monkeypatch):
    task, context, calls, _ = setup(tmp_path)
    monkeypatch.setenv('ALPHONSE_JEV_CAPTURE_DIR', str(tmp_path / 'debug'))
    curate_request_tools(task, context.tools.list(), context)
    files = list((tmp_path / 'debug' / 'jev-tool-requests').glob('*.json'))
    assert len(files) == 1 and json.loads(files[0].read_text()) == calls[0]
    assert stat.S_IMODE(files[0].stat().st_mode) == 0o600
    assert stat.S_IMODE(files[0].parent.stat().st_mode) == 0o700
    assert 'fixture-secret' not in files[0].read_text()


def test_non_native_programs_never_become_curation_candidates(tmp_path):
    task, context, calls, _ = setup(tmp_path)
    artifact = ToolDescriptor('artifact.fixture', 'Fixture', ToolKind.ARTIFACT)
    selected = curate_request_tools(task, (*context.tools.list(), artifact), context)
    assert artifact not in selected
    assert 'artifact.fixture' not in json.dumps(calls)


def test_unknown_native_template_defers_to_llm_instead_of_hiding_tool(tmp_path):
    task, context, calls, _ = setup(tmp_path)
    unknown = ToolDescriptor('native.future_tool', 'Future', ToolKind.NATIVE)
    selected = curate_request_tools(task, (*context.tools.list(), unknown), context)
    assert unknown in selected and calls == []
    assert task.metadata['v3_request_tool_curation']['status'] == 'error_llm_fallback'


def test_recovered_origin_is_ordered_before_recent_history_and_other_owners_are_excluded(tmp_path):
    from alphonse.agent_v2.conversations import SQLiteConversationStore
    task, context, calls, _ = setup(tmp_path)
    store = SQLiteConversationStore(tmp_path / 'conversation.sqlite')
    origin = store.record(owner_user_id=task.user, role='user', content='ORIGIN_SOURCE', source='test', source_message_id='origin')
    private = store.record(owner_user_id='other', role='user', content='PRIVATE_OTHER_OWNER', source='test', source_message_id='private')
    for i in range(12):
        store.record(owner_user_id=task.user, role='user', content=f'Recent request {i}', source='test', source_message_id=f'recent-{i}')
    context.conversation_store = store
    task.metadata['v3_retry_continuity'] = {'status': 'selected', 'access_status': 'available',
        'source_ids': ['conversation:' + origin.event_id]}
    state = request_tool_state(task, context)
    assert state['conversation'][0]['text'] == 'ORIGIN_SOURCE'
    assert 'Recent request 11' in str(state)
    assert 'PRIVATE_OTHER_OWNER' not in str(state)
    task.metadata['v3_retry_continuity']['source_ids'] = ['conversation:' + private.event_id]
    assert len(curate_request_tools(task, context.tools.list(), context)) == 2
    assert task.metadata['v3_request_tool_curation']['status'] == 'incomplete_state_llm_fallback'
    assert calls == []
