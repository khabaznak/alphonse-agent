"""Disposable standard packages exercise activation, routing and native reads."""
from pathlib import Path
from types import SimpleNamespace
import hashlib

import pytest

from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.skill_resources import SkillActivationError, read_resource
from alphonse.agent_v2.core.core import CoreLoopContext, ToolExecutionContext
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.context import selected_skill_guidance, skill_candidates_for_curation
from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase
from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor
from alphonse.agent_v2.core.intelligence.v3.contracts import new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.core.inference import InferenceResult, InferencePurpose
from alphonse.agent_v2.system_one import (
    JevCriterionDecisionProvider, SystemOneSettings, SystemOneAdmissionDecision,
    SystemOneToolRegistrySelection, SystemOneTacticalReview,
)


@pytest.fixture
def library(tmp_path, monkeypatch):
    store = SkillStore(tmp_path / 'skills')
    monkeypatch.setattr(store, '_bundled_skills_dir', lambda: tmp_path / 'no-bundled')
    return store


def task_for(*skills):
    return TaskState(goal='Use the reusable workflow', user='alex', task_id='fresh', message_id='fresh',
                     metadata={'v3_skill_selection': {'selected_skill_ids': [s.skill_id for s in skills]}})


def loop(store, **kwargs):
    return CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=store, **kwargs)


def add_resource(skill, name='references/guide.md', content='RESOURCE_SENTINEL'):
    path = Path(skill.directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_multiple_full_instructions_are_atomic_and_budget_overflow_is_explicit(library):
    first = library.create_skill('first', 'Core expertise', 'FIRST_START' + 'x' * 10000 + 'FIRST_END')
    second = library.create_skill('second', 'Complementary expertise', 'SECOND_START' + 'y' * 10000 + 'SECOND_END')
    task = task_for(first, second)
    text, loaded = selected_skill_guidance(task, loop(library))
    assert first.instructions in text and second.instructions in text
    assert loaded == [first.skill_id, second.skill_id]
    with pytest.raises(SkillActivationError, match='instruction_budget_exceeded'):
        selected_skill_guidance(task, loop(library), max_chars=16000)
    audit = task.metadata['v3_skill_selection']
    assert audit['loaded_skill_ids'] == []
    assert audit['required_chars'] > audit['budget_chars']
    assert 'never be clipped' in audit['recovery']


def test_resources_load_on_demand_against_skill_root_with_source_receipts(library, tmp_path, monkeypatch):
    skill = library.create_skill('fixture', 'Fixture expertise', 'Use references/guide.md only when needed.')
    add_resource(skill)
    add_resource(skill, 'scripts/README.md', 'SCRIPT_DOCUMENTATION')
    add_resource(skill, 'assets/template.txt', 'ASSET_TEMPLATE')
    add_resource(skill, 'live.sqlite3', 'DATABASE_SENTINEL')
    other = tmp_path / 'daemon'
    other.mkdir()
    monkeypatch.chdir(other)
    task = task_for(skill)
    task.project_id = 'irrelevant-project'
    text, _ = selected_skill_guidance(task, loop(library))
    assert str(Path(skill.directory).resolve()) in text
    assert 'RESOURCE_SENTINEL' not in text and 'DATABASE_SENTINEL' not in text
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue())
    for name, content in [('references/guide.md', 'RESOURCE_SENTINEL'),
                          ('scripts/README.md', 'SCRIPT_DOCUMENTATION'), ('assets/template.txt', 'ASSET_TEMPLATE')]:
        result = read_resource({'skill_id': skill.skill_id, 'path': name}, context=context, store=library)
        assert result['content'] == content
        assert result['sha256'] == hashlib.sha256(content.encode()).hexdigest()
    assert len(task.metadata['v3_skill_resource_reads']) == 3
    listing = read_resource({'skill_id': skill.skill_id, 'path': '.', 'operation': 'list'}, context=context, store=library)
    assert 'live.sqlite3' not in [entry['path'] for entry in listing['entries']]


@pytest.mark.parametrize('path', ['../outside', 'references/../SKILL.md', '/etc/passwd', '.env', 'live.sqlite3', 'SKILL.md', './SKILL.md'])
def test_resource_paths_and_private_data_are_deterministic(library, path):
    skill = library.create_skill('fixture', 'Expertise', 'Use the workflow.')
    task = task_for(skill)
    selected_skill_guidance(task, loop(library))
    with pytest.raises((PermissionError, ValueError)):
        read_resource({'skill_id': skill.skill_id, 'path': path},
                      context=ToolExecutionContext(task=task, messages=InMemoryMessageQueue()), store=library)


def test_symlink_escape_and_in_package_alias_are_rejected(library, tmp_path):
    skill = library.create_skill('fixture', 'Expertise', 'Use the workflow.')
    target = add_resource(skill)
    outside = tmp_path / 'private'
    outside.write_text('PRIVATE')
    for destination in (target, outside):
        link = Path(skill.directory) / 'link'
        link.symlink_to(destination)
        with pytest.raises(ValueError, match='symlink'):
            library.read_file(skill.skill_id, 'link')
        task = task_for(skill)
        with pytest.raises(SkillActivationError, match='package_unavailable'):
            selected_skill_guidance(task, loop(library))
        link.unlink()


def test_discovery_and_each_read_recheck_access_independently_of_selection(library, monkeypatch):
    skill = library.create_skill('fixture', 'Shared code', 'Use the workflow.')
    add_resource(skill)
    allowed = {'alex'}
    monkeypatch.setattr(library, 'can_use', lambda identity, actor: actor in allowed)
    task = task_for(skill)
    assert skill_candidates_for_curation(loop(library), task)[0]['id'] == skill.skill_id
    selected_skill_guidance(task, loop(library))
    allowed.clear()
    assert skill_candidates_for_curation(loop(library), task) == []
    with pytest.raises(SkillActivationError, match='access_denied'):
        read_resource({'skill_id': skill.skill_id, 'path': 'references/guide.md'},
                      context=ToolExecutionContext(task=task, messages=InMemoryMessageQueue()), store=library)
    task.metadata['v3_skill_selection']['selected_skill_ids'] = []
    with pytest.raises(PermissionError, match='not_selected'):
        read_resource({'skill_id': skill.skill_id, 'path': 'references/guide.md'},
                      context=ToolExecutionContext(task=task, messages=InMemoryMessageQueue()), store=library)


def test_unavailable_and_changed_packages_do_not_silently_disappear(library):
    skill = library.create_skill('fixture', 'Expertise', 'Use the workflow.')
    reference = add_resource(skill)
    task = task_for(skill)
    selected_skill_guidance(task, loop(library))
    reference.write_text('NEW_REVISION')
    with pytest.raises(SkillActivationError, match='package_changed_replan_required'):
        selected_skill_guidance(task, loop(library))
    # A fresh task sees the new package; it never inherits another task's pin.
    fresh = task_for(skill)
    selected_skill_guidance(fresh, loop(library))
    assert fresh.metadata['v3_skill_selection']['packages'] != task.metadata['v3_skill_selection']['packages']
    (Path(skill.directory) / 'SKILL.md').unlink()
    with pytest.raises(SkillActivationError, match='package_unavailable'):
        selected_skill_guidance(fresh, loop(library))


def test_legacy_private_resource_package_cannot_be_read_via_generic_skill_reader(library):
    skill = library.create_skill('fixture', 'Shared workflow', 'Use the workflow.')
    add_resource(skill, 'references/private.txt', 'PRIVATE')
    resources = SimpleNamespace(store=SimpleNamespace(list=lambda: [SimpleNamespace(
        skill_id=skill.skill_id, storage_kind='skill')]), can_use=lambda record, actor: False)
    task = task_for(skill)
    with pytest.raises(PermissionError, match='private_legacy'):
        read_resource({'skill_id': skill.skill_id, 'path': 'references/private.txt'}, context=ToolExecutionContext(
            task=task, messages=InMemoryMessageQueue(), artifact_resources=resources), store=library)


def test_jev_scores_multiple_core_and_complementary_skills_even_with_existing_context(library):
    skills = [library.create_skill(name, 'Metadata only', f'INSTRUCTION_SECRET_{name}')
              for name in ['core-one', 'core-two', 'complement', 'irrelevant']]
    payloads = []
    def transport(url, key, payload, timeout):
        payloads.append(payload)
        answers = {}
        for identity, question in payload['questions'].items():
            if identity.startswith('skill_fit_'):
                level = [2, 2, 1, 0][int(identity.rsplit('_', 1)[1])]
                probabilities = {str(i): 0.96 if i == level else 0.02 for i in range(3)}
                answers[identity] = {'type': 'score', 'score': level, 'confidence': 0.96, 'probabilities': probabilities}
            else:
                answers[identity] = {'type': 'noul', 'noul': 0.02}
        return {'answers': answers, 'model': 'fixture-jev'}
    provider = JevCriterionDecisionProvider(SystemOneSettings(enabled=True, api_key='fixture', validated_at='now'), transport=transport)
    decision = provider.classify_task_admission(message='Use combined expertise', selected_context='Already selected context',
                                                skill_candidates=[skill.candidate() for skill in skills])
    assert decision.selected_skill_ids == tuple(s.skill_id for s in skills[:3])
    assert decision.primary_skill_ids == tuple(s.skill_id for s in skills[:2])
    assert len(decision.to_metadata()['skill_scores']) == 4
    assert 'INSTRUCTION_SECRET' not in str(payloads)


class Inference:
    def __init__(self):
        self.requests = []
    def generate_markdown(self, request):
        self.requests.append(request)
        return InferenceResult(content='Fixture reply', model_profile=request.model_profile)
    def generate_json(self, request):
        self.requests.append(request)
        if request.purpose == InferencePurpose.PHASE_PLANNING:
            value = {'phase_id': 'inspect', 'objective': 'Read supporting documentation',
                     'acceptance_criteria': ['Required documentation is inspected'],
                     'authorized_capabilities': ['skill_resources'], 'subgoals': [{
                         'subgoal_id': 'read', 'objective': 'Inspect guide', 'required_output_type': 'reference',
                         'allowed_capabilities': ['skill_resources'], 'allowed_side_effects': [],
                         'completion': {'kind': 'output_present', 'output_type': 'reference'}}]}
        else:
            value = {'tool_id': 'native.skill_read', 'arguments': {'skill_id': 'skill:fixture', 'path': 'references/guide.md'},
                     'acceptance_questions': [{'question_id': 'reference-read', 'type': 'noul',
                                              'instructions': 'Was the reference read?',
                                              'criteria': {'true': 'Reference content is returned.', 'false': 'Read failed.'}}]}
        return InferenceResult(content='', json_value=value, model_profile=request.model_profile)


class Jev:
    def __init__(self, skills):
        self.skills = skills
        self.candidates = []
    def classify_task_admission(self, **kwargs):
        self.candidates.append(kwargs.get('skill_candidates', []))
        ids = tuple(s.skill_id for s in self.skills)
        return SystemOneAdmissionDecision(False, 0.02, True, selected_skill_ids=ids, primary_skill_ids=ids)
    def curate_request_tools(self, **kwargs):
        return SystemOneToolRegistrySelection(selected_tool_ids=tuple(t.tool_id for t in kwargs['tools']))
    select_plan_tools = curate_request_tools
    def evaluate_tactical_progress(self, **kwargs):
        return SystemOneTacticalReview(True, 0.99, True, model='fixture-jev', answers={'reference-read': 0.99})


def fresh_task():
    return TaskState(goal='Please help with this workflow', user='alex', task_id='new', message_id='new',
                     metadata={'routing_disposition': 'pdca_task'})


def test_fresh_no_project_direct_reply_receives_complete_combined_instructions(library):
    skills = [library.create_skill(name, 'Expertise', name.upper() + 'x' * 9500 + '_END') for name in ['first', 'second']]
    inference, jev = Inference(), Jev(skills)
    task = fresh_task()
    result = HierarchicalCAPDProcessor().process(task, loop(library, inference=inference, system_one=jev))
    assert result.status.value == 'completed'
    assert not task.project_id
    for skill in skills:
        assert skill.instructions in inference.requests[0].prompt
    assert len(jev.candidates[0]) == 2
    assert task.metadata['v3_skill_selection']['loaded_skill_ids'] == [s.skill_id for s in skills]


def test_supporting_resources_route_through_planning_and_execution_on_demand(library):
    skill = library.create_skill('fixture', 'Expertise', 'Read references/guide.md when needed. INSTRUCTIONS_END')
    add_resource(skill)
    inference, jev = Inference(), Jev([skill])
    task = fresh_task()
    registry = build_native_tool_registry(skill_store=library)
    context = loop(library, inference=inference, system_one=jev, tools=registry)
    assert HierarchicalCAPDProcessor._admit_initial_human_task(task, context) is False
    assert task.metadata['v3_admission']['route'] == 'task'
    phase = plan_phase(task, context)
    planning_prompt = next(r.prompt for r in inference.requests if r.purpose == InferencePurpose.PHASE_PLANNING)
    assert skill.instructions in planning_prompt and 'RESOURCE_SENTINEL' not in planning_prompt
    state = new_tactical_state(phase)
    tools = ToolRevealPolicy().reveal(task, state, phase.subgoals[0], registry.list()).tools
    assert [t.tool_id for t in tools] == ['native.skill_read']
    action = PhaseExecutor()._select_action(task, state, phase.subgoals[0], tools, context)
    assert skill.instructions in inference.requests[-1].prompt
    outcome = ToolInvocationService(context=context, task=task).invoke(action['tool_id'], action['arguments'])
    assert outcome['status'] == 'success'
    assert outcome['result']['content'] == 'RESOURCE_SENTINEL'
    assert not any(t.tool_id.startswith('artifact.') for t in registry.list())


def test_missing_selected_package_fails_before_any_response(library):
    skill = SimpleNamespace(skill_id='skill:missing')
    inference = Inference()
    task = fresh_task()
    result = HierarchicalCAPDProcessor().process(task, loop(library, inference=inference, system_one=Jev([skill])))
    assert result.status.value == 'failed'
    assert task.metadata['v3_skill_selection']['loaded_skill_ids'] == []
    assert inference.requests == []


def test_activation_fingerprints_without_reading_supporting_resources(library, monkeypatch):
    skill = library.create_skill('fixture', 'Expertise', 'Read references/guide.md only when needed.')
    add_resource(skill)
    original = Path.read_bytes
    def reject_resource_read(path):
        raise AssertionError(f'Activation read supporting bytes: {path.name}')
    monkeypatch.setattr(Path, 'read_bytes', reject_resource_read)
    selected_skill_guidance(task_for(skill), loop(library))
    monkeypatch.setattr(Path, 'read_bytes', original)


def test_planned_resource_subgoal_executes_through_real_executor(library):
    skill = library.create_skill('fixture', 'Expertise', 'Read references/guide.md.')
    add_resource(skill)
    task = task_for(skill)
    inference, jev = Inference(), Jev([skill])
    # Even a negative Jev tool vote cannot hide the generic reader that System
    # Two explicitly plans. Deterministic reveal still needs skill_resources.
    jev.curate_request_tools = lambda **kwargs: SystemOneToolRegistrySelection()
    jev.select_plan_tools = jev.curate_request_tools
    context = loop(library, inference=inference, system_one=jev,
                   tools=build_native_tool_registry(skill_store=library))
    phase = plan_phase(task, context)
    assert phase.curated_tool_ids == ('native.skill_read',)
    state = new_tactical_state(phase)
    outcome = PhaseExecutor(reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status.value == 'phase_complete'
    assert state.actions[0].result['content'] == 'RESOURCE_SENTINEL'
    assert task.metadata['v3_skill_resource_reads'][0]['path'] == 'references/guide.md'


def test_resource_failures_are_explicit_and_cancelled_task_cannot_read(library):
    skill = library.create_skill('fixture', 'Expertise', 'Use references on demand.')
    add_resource(skill, 'assets/binary.bin', 'abc\x00def')
    task = task_for(skill)
    selected_skill_guidance(task, loop(library))
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue())
    for path, message in [('assets/missing.txt', 'not_found'), ('assets/binary.bin', 'binary')]:
        with pytest.raises(ValueError, match=message):
            read_resource({'skill_id': skill.skill_id, 'path': path}, context=context, store=library)
    task.status = 'cancelled'
    with pytest.raises(PermissionError, match='cancelled'):
        read_resource({'skill_id': skill.skill_id, 'path': 'assets/binary.bin'}, context=context, store=library)
    assert not task.metadata.get('v3_skill_resource_reads')


def test_budget_failure_stops_fresh_direct_reply_before_inference(library, monkeypatch):
    skill = library.create_skill('fixture', 'Expertise', 'Use the entire essential instruction body.')
    monkeypatch.setenv('ALPHONSE_SKILL_INSTRUCTION_BUDGET_CHARS', '100')
    task, inference = fresh_task(), Inference()
    result = HierarchicalCAPDProcessor().process(task, loop(library, inference=inference, system_one=Jev([skill])))
    assert result.status.value == 'failed'
    assert 'instruction_budget_exceeded' in task.outcome['reason']
    assert inference.requests == []


def test_revision_changed_during_inference_stops_native_dispatch(library):
    skill = library.create_skill('fixture', 'Expertise', 'Read references/guide.md.')
    resource = add_resource(skill)
    task, inference = task_for(skill), Inference()
    context = loop(library, inference=inference, system_one=Jev([skill]),
                   tools=build_native_tool_registry(skill_store=library))
    phase = plan_phase(task, context)
    original = inference.generate_json
    def change_package(request):
        resource.write_text('CHANGED_DURING_INFERENCE')
        return original(request)
    inference.generate_json = change_package
    with pytest.raises(SkillActivationError, match='package_changed_replan_required'):
        PhaseExecutor(reveal_policy=ToolRevealPolicy()).run(task, new_tactical_state(phase), context)
    assert not task.metadata.get('v3_skill_resource_reads')


def test_standard_script_dispatch_rechecks_activation_access_and_revision(library, monkeypatch):
    from alphonse.agent_v2.core.tools.registry.native.bash import execute_bash
    skill = library.create_skill('fixture', 'Expertise', 'Run the documented command.')
    reference = add_resource(skill)
    task = task_for(skill)
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), skill_store=library)
    selected_skill_guidance(task, loop(library))
    result = execute_bash({'skill_id': skill.skill_id, 'command': 'printf fixture'}, context=context)
    assert result['stdout'] == 'fixture'
    reference.write_text('CHANGED_SCRIPT_PACKAGE')
    with pytest.raises(SkillActivationError, match='package_changed_replan_required'):
        execute_bash({'skill_id': skill.skill_id, 'command': 'printf forbidden'}, context=context)
    monkeypatch.setattr(library, 'can_use', lambda identity, actor: False)
    with pytest.raises(SkillActivationError, match='access_denied'):
        execute_bash({'skill_id': skill.skill_id, 'command': 'printf forbidden'}, context=context)


def test_resource_reader_honors_live_cancellation_checker(library):
    skill = library.create_skill('fixture', 'Expertise', 'Use references.')
    add_resource(skill)
    task = task_for(skill)
    selected_skill_guidance(task, loop(library))
    context = ToolExecutionContext(task=task, messages=InMemoryMessageQueue(), cancellation_checker=lambda: True)
    with pytest.raises(PermissionError, match='cancelled'):
        read_resource({'skill_id': skill.skill_id, 'path': 'references/guide.md'}, context=context, store=library)
