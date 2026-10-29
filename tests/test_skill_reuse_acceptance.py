"""#28 recovery -> authenticated approval -> installation -> fresh reuse."""
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.evaluation.skill_reuse import (
    BUDGET, SKILL, Trial, entries, run_trial,
)
from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.core.tools.registry.native.skill_install import execute_skill_install
from alphonse.agent_v2.system_one import SystemOneTacticalReview


def test_complete_real_processor_trial_preserves_attempts_and_provenance():
    report = run_trial()
    assert report['mechanical_pass'], report['checks']
    assert report['mode'] == 'deterministic-models-real-processor-and-native-shell'
    initial = report['initial_task']
    assert initial['status'] == 'waiting_user'
    attempts = entries(SimpleNamespace(metadata=initial['metadata']))
    shell = [e for e in attempts if e.get('tool_id') == 'native.bash']
    assert [(e['subgoal_id'], e['status']) for e in shell] == [('primary', 'failed'), ('backup', 'success')]
    assert shell[0]['result']['exit_code'] == 1
    assert 'no measurement' in shell[0]['result']['stderr']
    assert json.loads(shell[1]['result']['stdout'])['temperature_c'] == 24
    approval = report['approval']
    assert approval['kind'] == 'authenticated_question_answer'
    assert approval['result']['answered_by'] == 'fixture-admin'
    assert approval['result']['answer'] == {'answer': True}
    draft = json.loads(approval['result']['question'].split('\n', 1)[1])
    installed = next(e['result'] for e in entries(SimpleNamespace(metadata=report['learning_task']['metadata']))
                     if e.get('tool_id') == 'native.skill_install')
    assert installed['instructions'] == draft['instructions']
    assert installed['skill']['description'] == draft['description']
    assert installed['verification']['status'] == 'verified'
    for counts in report['calls'].values():
        assert counts['planning'] <= BUDGET['planning_calls']
        assert counts['tactical'] <= BUDGET['tactical_calls']
    assert report['budgets']['automatic_shell_retries'] == 0
    assert '27 C' in report['fresh_task']['metadata']['prepared_user_response']['message']
    assert report['fresh_task']['metadata']['v3_skill_selection']['loaded_skill_ids'] == [SKILL]


def test_blocked_recovery_reports_evidence_without_install_or_success():
    report = run_trial(offline=True)
    assert report['mechanical_pass'], report['checks']
    assert report['approval'] is None and report['fresh_task'] is None
    assert report['learning_task']['status'] == 'failed'
    assert 'Both available routes failed' in report['learning_task']['outcome']['reason']
    message = report['learning_task']['metadata']['v3_failure_response']['message']
    assert 'could not obtain a fresh reading' in message
    assert '24' not in message
    assert report['calls']['learning'] == {'planning': 4, 'tactical': 4}


def approved_trial(tmp_path):
    trial = Trial(tmp_path)
    task, context = trial.task_context('learning')
    processor = HierarchicalCAPDProcessor()
    processor.process(task, context)
    assert task.status == 'waiting_user'
    question = trial.questions.list_pending_for_respondent('fixture-admin')[0]
    answer = trial.questions.route_answer(respondent_user_id='fixture-admin', question_id=question.question_id,
                                         payload={'answer': True})
    task = answer.resumed_task
    context.inference.task = task
    return trial, task, context, processor


def test_checkpoint_resume_after_install_does_not_repeat_install(tmp_path):
    trial, task, context, processor = approved_trial(tmp_path)
    calls = []
    create = trial.skills.create_skill
    def counted(*args, **kwargs):
        calls.append(args)
        return create(*args, **kwargs)
    trial.skills.create_skill = counted
    def reviewed(**values):
        if values['subgoal']['subgoal_id'] == 'install':
            context.cancellation_checker = lambda: True
        return SystemOneTacticalReview(True, .99, True)
    context.system_one.evaluate_tactical_progress = reviewed
    processor.process(task, context)
    assert task.status == 'cancelled'
    assert len(calls) == 1 and trial.skills.get(SKILL)
    restored = trial.questions.load_task_checkpoint(task.task_id)
    restored.status = 'running'
    restored.hierarchical_state['status'] = 'running'
    restored.hierarchical_state['cancellation_pending'] = False
    context.cancellation_checker = None
    context.inference.task = restored
    processor.process(restored, context)
    assert restored.status == 'completed'
    assert len(calls) == 1
    processor.process(restored, context)
    assert len(calls) == 1
    assert len(restored.metadata['v3_resumption_evidence']) == 1
    receipt = restored.metadata['v3_resumption_evidence'][0]
    imported = [e for e in restored.hierarchical_state['evidence']['entries']
                if e['evidence_ref'] == receipt['evidence_ref']]
    assert len(imported) == 1 and imported[0]['result'] == receipt['result']


def test_exact_repeat_is_verified_and_conflicting_definition_is_preserved(tmp_path):
    trial = Trial(tmp_path)
    task, context = trial.task_context('install')
    invoke = ToolInvocationService(context=context, task=task)
    first = invoke.execute_or_raise('native.skill_install', trial.draft)
    repeated = invoke.execute_or_raise('native.skill_install', trial.draft)
    assert first['created'] and not repeated['created']
    assert repeated['instructions'] == trial.draft['instructions']
    assert trial.skills.management_status(SKILL)['owner'] == 'fixture-admin'
    with pytest.raises(ValueError, match='existing_definition_mismatch'):
        invoke.execute_or_raise('native.skill_install', {**trial.draft, 'instructions': 'Unapproved alteration'})
    assert trial.skills.get(SKILL).instructions == trial.draft['instructions']
    assert len(trial.skills.list_skills()) == 1


@pytest.mark.parametrize('user', ['', 'stranger'])
def test_selection_or_exact_existing_package_never_grants_install_permission(tmp_path, user):
    trial = Trial(tmp_path)
    trial.skills.create_skill(**trial.draft)
    with pytest.raises(PermissionError, match='skill_manager_required'):
        execute_skill_install(trial.draft, context=SimpleNamespace(task=SimpleNamespace(user=user)),
                              store=trial.skills, is_admin=trial.users.is_admin)


def test_repeat_still_validates_package_name_and_original_description(tmp_path):
    trial = Trial(tmp_path)
    task, context = trial.task_context('install')
    invoke = ToolInvocationService(context=context, task=task)
    draft = {**trial.draft, 'description': '  Whitespace retained  '}
    assert invoke.execute_or_raise('native.skill_install', draft)['created']
    assert not invoke.execute_or_raise('native.skill_install', draft)['created']
    with pytest.raises(ValueError, match='skill_name_invalid'):
        invoke.execute_or_raise('native.skill_install', {**draft, 'name': SKILL})
