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














def test_legacy_private_resource_package_cannot_be_read_via_generic_skill_reader(library):
    skill = library.create_skill('fixture', 'Shared workflow', 'Use the workflow.')
    add_resource(skill, 'references/private.txt', 'PRIVATE')
    resources = SimpleNamespace(access_records=lambda: [SimpleNamespace(
        skill_id=skill.skill_id, storage_kind='skill')], can_use=lambda record, actor: False)
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
