"""Issue #128: disposable skill/SQLite workflow, no providers or household data."""
from pathlib import Path
import shlex
from copy import deepcopy
import sqlite3
import sys
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.context import task_context_for_decision
from alphonse.agent_v2.core.intelligence.v3.contracts import PhasePlan, PhaseSubgoal, PhaseStatus, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase, _curate_request_tools, _curate_phase_tools
from alphonse.agent_v2.core.intelligence.v3.outer import review_phase, PhaseReviewStatus
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
from alphonse.agent_v2.core.tools.registry.native.memory import build_search_memory_tool_definition, build_read_memory_source_tool_definition
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection, SystemOneTacticalReview


@pytest.mark.parametrize('projected', [False, True])
@pytest.mark.parametrize('operation', ['commit', 'denied', 'unavailable', 'rollback'])
@pytest.mark.parametrize('invalid_plan_first', [False, True])
def test_confirmed_record_update_reaches_shell_and_verifies_transaction(tmp_path, monkeypatch, projected, operation, invalid_plan_first):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', str(int(projected)))
    skills = SkillStore(tmp_path / '.alphonse' / 'skills')
    skill = skills.create_skill('fixture-records', 'Maintain disposable records',
        'Authoritative SQLite resource: records.sqlite in this skill package. '
        'Use native.bash, BEGIN a transaction, update the requested record, COMMIT and read back with a fresh connection. '
        'On error ROLLBACK; never claim a write from a reply.')
    database = Path(skill.directory) / 'records.sqlite'
    with sqlite3.connect(database) as db:
        db.execute('CREATE TABLE records (id INTEGER PRIMARY KEY, value TEXT)')
        db.execute("INSERT INTO records VALUES (1, 'old')")
    task = TaskState(task_id='fixture', user='alex', goal='Yes, save the reviewed value.',
        recent_conversation_md='User: Update record 1 to reviewed. Assistant: Please confirm.',
        metadata={'v3_skill_selection': {'selected_skill_ids': [skill.skill_id]}})
    phase = dict(schema_version=3, phase_id='write-record', objective='Persist reviewed record and read back',
        acceptance_criteria=['Record 1 is persisted as reviewed and verified by fresh read-back'], criterion_ids=['ac-1'],
        authorized_capabilities=['local_shell'], mutation_scope={'allowed_paths': ['records.sqlite'], 'allow_external_effects': True},
        originating_decision='Current confirmed request; selected skill provides workflow only', subgoals=[dict(
            subgoal_id='write', objective='Transaction and persisted read-back', required_output_type='verified_record',
            depends_on=[], allowed_capabilities=['local_shell'], allowed_side_effects=['external_reversible'],
            completion={'kind': 'output_present', 'output_type': 'verified_record'}, failure_policy='stop')])
    seen = []

    class Jev:
        def curate_request_tools(self, *, task_context, **kwargs):
            # This content-sensitive stub reproduces the original context-loss
            # failure: without skill guidance the only candidate is respond.
            seen.append(task_context)
            selected = ('native.bash',) if 'Authoritative SQLite resource' in task_context else ('native.respond',)
            return SystemOneToolRegistrySelection(selected_tool_ids=selected)

        def select_plan_tools(self, *, task_context, **kwargs):
            seen.append(task_context)
            assert 'COMMIT and read back' in task_context
            return SystemOneToolRegistrySelection(selected_tool_ids=('native.bash',))

        def evaluate_tactical_progress(self, **kwargs):
            return SystemOneTacticalReview(True, .99, True)

        def judge_recall(self, **kwargs):
            return False

    registry = InMemoryToolRegistry()
    registry.register(build_bash_tool_definition())
    registry.register(build_respond_tool_definition())
    planning_requests = []

    def generate_json(request):
        planning_requests.append(request)
        candidate = deepcopy(phase)
        if invalid_plan_first and len(planning_requests) == 1:
            candidate['mutation_scope']['allowed_paths'] = [str(database)]
        if len(planning_requests) == 2:
            assert 'No tool or mutation ran' in request.prompt
            assert 'Do not drop the mutation target' in request.prompt
            with sqlite3.connect(database) as db:
                assert db.execute('SELECT value FROM records WHERE id=1').fetchone()[0] == 'old'
            assert any(r['source_type'] == 'plan_validation' for manifest in request.context_manifests for r in manifest['records'])
        return SimpleNamespace(json_value=candidate)

    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=skills, tools=registry, system_one=Jev(),
        inference=SimpleNamespace(generate_json=generate_json))
    planned = plan_phase(task, context)
    assert planned.curated_tool_ids == ('native.bash',)
    assert planned.mutation_scope.allowed_paths == ('records.sqlite',)
    assert len(planning_requests) == (2 if invalid_plan_first else 1)
    assert 'persisted' in task.acceptance_criteria_md
    for text in seen:
        included = [r for r in text.records if r['status'] == 'included' and r['source_id'] == ('skills:selected' if projected else skill.skill_id)]
        assert included
        for receipt in included:
            span = receipt['input_range']
            assert 'Authoritative SQLite' in text[span['start_char']:span['end_char']]
    uri = str(database) + '?mode=ro' if operation == 'denied' else str(database)
    if operation == 'unavailable':
        uri = str(database.parent / 'missing.sqlite') + '?mode=rw'
    script = f'''import sqlite3
connection = sqlite3.connect({('file:' + uri)!r}, uri=True)
try:
    connection.execute('BEGIN')
    connection.execute("UPDATE records SET value='reviewed' WHERE id=1")
    if {operation == 'rollback'!r}:
        raise RuntimeError('synthetic rollback')
    connection.commit()
except Exception:
    connection.rollback()
    raise
finally:
    connection.close()
with sqlite3.connect({str(database)!r}) as db:
    assert db.execute('SELECT value FROM records WHERE id=1').fetchone()[0] == 'reviewed'
print('PERSISTED_READ_BACK:reviewed')
'''
    command = shlex.quote(sys.executable) + ' -c ' + shlex.quote(script)
    state = new_tactical_state(planned)
    outcome = PhaseExecutor(reveal_policy=ToolRevealPolicy(), action_selector=lambda *_: {
        'tool_id': 'native.bash', 'arguments': {'command': command, 'cwd': skill.directory},
    }).run(task, state, context)
    with sqlite3.connect(database) as db:
        value = db.execute('SELECT value FROM records WHERE id=1').fetchone()[0]
    assert value == ('reviewed' if operation == 'commit' else 'old'), state.evidence.entries
    assert outcome.status == (PhaseStatus.PHASE_COMPLETE if operation == 'commit' else PhaseStatus.BLOCKED)
    if operation != 'commit':
        # Even a positive semantic stub cannot promote failed command evidence.
        review = review_phase(task, state, outcome, context)
        assert review.status == PhaseReviewStatus.PHASE_BLOCKED
    else:
        assert 'PERSISTED_READ_BACK:reviewed' in state.evidence.entries[-1]['result']['stdout']


@pytest.mark.parametrize('signal', ['requires_recall', 'history_unavailable'])
def test_default_curation_retains_authorized_recall_after_structured_omission(monkeypatch, signal):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', '0')
    task = TaskState(user='alex', goal='Yes, use those details.', metadata={'conversation_projection': {signal: True}})
    registry = InMemoryToolRegistry()
    for builder in (build_search_memory_tool_definition, build_read_memory_source_tool_definition):
        registry.register(builder())
    jev = SimpleNamespace(curate_request_tools=lambda **_: SystemOneToolRegistrySelection(),
                          select_plan_tools=lambda **_: SystemOneToolRegistrySelection())
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=jev)
    tools = registry.list()
    curated = _curate_request_tools(task, tools, system_prompt='', session_history='', project_context='',
        durable_memory='', attachment_manifest=[], context=context)
    assert {t.tool_id for t in curated} == {'native.search_memory', 'native.read_memory_source'}
    phase = PhasePlan('recall', 'Recover missing antecedent', (
        PhaseSubgoal('recall', 'Read authorized prior details', 'memory', allowed_capabilities=('memory_recall',)),
    ), authorized_capabilities=('memory_recall',))
    phase = _curate_phase_tools(task, phase, curated, context)
    assert set(phase.curated_tool_ids) == {'native.search_memory', 'native.read_memory_source'}
    assert set(task.metadata['system_one_request_tool_selections'][-1]['retained_recall_tool_ids']) == set(phase.curated_tool_ids)
    # Retention operates only over the exposure-policy candidates it receives.
    assert not _curate_request_tools(task, (), system_prompt='', session_history='', project_context='',
        durable_memory='', attachment_manifest=[], context=context)


@pytest.mark.parametrize('projected', [False, True])
def test_short_confirmed_followup_selects_antecedent_skill_with_coverage_receipt(tmp_path, monkeypatch, projected):
    from alphonse.agent_v2.core.intelligence.v3.context import conversation_coverage
    from alphonse.agent_v2.system_one import JevCriterionDecisionProvider, SystemOneSettings

    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', str(int(projected)))
    task = TaskState(user='alex', goal='Yes, save it.', task_id='followup',
        recent_conversation_md='User: Use fixture-records to persist record 1 as reviewed. Assistant: Confirm?',
        metadata={'conversation_projection': {'requires_recall': True}})
    skills = SkillStore(tmp_path / 'skills')
    relevant = skills.create_skill('fixture-records', 'Maintain local records', 'Transaction and read-back workflow.')
    unrelated = skills.create_skill('fixture-music', 'Play music', 'Music workflow.')
    captured = []

    def transport(_url, _key, payload, _timeout):
        captured.append(payload)
        answers = {}
        for key, question in payload['questions'].items():
            if question['type'] == 'score':
                assert 'short confirmed follow-up' in question['instructions']
                state = payload['state']
                assert 'persist record 1' in str(state)
                assert 'requires_recall' in str(state)
                score = 2 if 'fixture-records' in question['instructions'] else 0
                answers[key] = {'type': 'score', 'score': score, 'confidence': .99,
                                'probabilities': {str(i): 1 if i == score else 0 for i in range(3)}}
            else:
                answers[key] = {'type': 'noul', 'noul': .99}
        return {'answers': answers}

    provider = JevCriterionDecisionProvider(SystemOneSettings(api_key='fixture'), transport=transport)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=skills, system_one=provider)
    result = provider.classify_task_admission(message=task.goal, recent_conversation=task.recent_conversation_md,
        conversation_coverage=conversation_coverage(task), skill_candidates=[relevant.candidate(), unrelated.candidate()],
        decision_context=(lambda ids: task_context_for_decision(task, context, purpose='admission')) if projected else None)
    assert result.requires_task
    assert result.selected_skill_ids == (relevant.skill_id,)
    assert result.primary_skill_ids == (relevant.skill_id,)
    assert captured


@pytest.mark.parametrize('path,attempts', [('/synthetic/records.sqlite', 2), ('../escape.sqlite', 1), ('.alphonse/settings.sqlite', 1)])
def test_scope_recovery_is_bounded_and_never_disables_path_validation(path, attempts):
    from alphonse.agent_v2.core.intelligence.v3.planner import V3PhasePlanValidationError
    requests = []
    invalid = dict(schema_version=3, phase_id='invalid', objective='Update synthetic record',
        acceptance_criteria=['Record persisted and read back'], criterion_ids=['ac-1'],
        authorized_capabilities=['local_shell'], mutation_scope={'allowed_paths': [path], 'allow_external_effects': False},
        subgoals=[dict(subgoal_id='write', objective='Update synthetic record', required_output_type='record',
            allowed_capabilities=['local_shell'], allowed_side_effects=['project_mutation'], completion={'kind': 'output_present'})])

    def generate(request):
        requests.append(request)
        return SimpleNamespace(json_value=deepcopy(invalid))

    task = TaskState(user='alex', goal='Save the synthetic record', task_id='validation')
    task.set_acceptance_contract_from_markdown('1.- [ ] The record is persisted and read back')
    contract = deepcopy(task.acceptance_contract)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), inference=SimpleNamespace(generate_json=generate))
    with pytest.raises(V3PhasePlanValidationError, match='mutation_scope_path_'):
        plan_phase(task, context)
    assert len(requests) == attempts
    assert task.acceptance_contract == contract
    assert not task.hierarchical_state


def test_plan_validation_failure_is_reported_as_planning_not_database_denial():
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    task = TaskState(user='alex', goal='Update the synthetic record', status='failed',
        outcome={'status': 'failure', 'reason': 'v3_phase_plan_invalid:mutation_scope_path_invalid:/private/synthetic.sqlite'})
    context = CoreLoopContext(messages=InMemoryMessageQueue())
    result = HierarchicalCAPDProcessor._result(task, context)
    message = task.metadata['v3_failure_response']['message']
    assert result.status.value == 'failed'
    assert 'execution plan was invalid' in message
    assert 'does not establish' in message
    assert '/private/synthetic.sqlite' not in message


def test_cancellation_after_rejected_plan_prevents_correction_and_remains_cancelled():
    from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
    cancelled = [False]
    requests = []

    def generate(request):
        requests.append(request)
        cancelled[0] = True
        return SimpleNamespace(json_value=dict(schema_version=3, phase_id='invalid', objective='Update fixture',
            mutation_scope={'allowed_paths': ['/fixture/records.sqlite']}, subgoals=[dict(
                subgoal_id='write', objective='Write fixture', required_output_type='record', completion={'kind': 'output_present'})]))

    task = TaskState(user='alex', goal='Update the synthetic record')
    context = CoreLoopContext(messages=InMemoryMessageQueue(), inference=SimpleNamespace(generate_json=generate),
                              cancellation_checker=lambda: cancelled[0])
    result = HierarchicalCAPDProcessor().process(task, context)
    assert result.status.value == 'cancelled'
    assert task.status == 'cancelled'
    assert len(requests) == 1
    assert not task.hierarchical_state
