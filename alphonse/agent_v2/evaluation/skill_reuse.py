"""Disposable #28 integration trial: real V3/native tools, deterministic model doubles.

This proves harness behavior, not live provider planning quality. No household
library, program, credential, or device is opened. All executable fixtures and
state are created under one TemporaryDirectory and removed after the receipt.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import tempfile
from time import monotonic

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.tools.registry.native import build_native_tool_registry
from alphonse.agent_v2.skills import SkillStore
from alphonse.agent_v2.system_one import (
    SystemOneAdmissionDecision, SystemOneActRecommendation, SystemOneReviewResult,
    SystemOneTacticalReview, SystemOneToolRegistrySelection,
)
from alphonse.agent_v2.users import V2UserStore

SKILL = 'skill:study-temperature'
BUDGET = {'planning_calls': 4, 'tactical_calls': 10, 'command_timeout_seconds': 5,
          'wall_seconds': 30, 'automatic_shell_retries': 0}
QUESTIONS = [{'question_id': 'observed', 'type': 'noul',
              'instructions': 'Does the observed result establish this stage?',
              'criteria': {'true': 'Direct successful evidence.', 'false': 'Missing or failed.'}}]
SENSOR = '''import json, sys
from datetime import datetime, timezone
from pathlib import Path
root = Path(__file__).parent
route = sys.argv[1]
if route == '--help':
    print('Available sensor routes: primary, backup. Repair the host connection if both are unavailable.')
    sys.exit(0)
if route == 'primary' or (root / 'backup-offline').exists():
    print('route unavailable: no measurement', file=sys.stderr)
    sys.exit(1)
value = json.loads((root / 'sensor.json').read_text())
print(json.dumps({'temperature_c': value['temperature_c'], 'source': 'study-backup',
                  'observed_at': datetime.now(timezone.utc).isoformat()}))
'''


def entries(task):
    return [e for h in task.metadata.get('v3_phase_history', [])
            for e in h.get('evidence', {}).get('entries', [])]


class Decisions:
    """Only bounded decisions. System Two supplies plans, calls and prose below."""
    def classify_task_admission(self, **values):
        selected = tuple(s['id'] for s in values['skill_candidates'] if s['id'] == SKILL)
        return SystemOneAdmissionDecision(True, .99, True,
                                         selected_skill_ids=selected, primary_skill_ids=selected)

    def curate_request_tools(self, **values):
        return SystemOneToolRegistrySelection(selected_tool_ids=tuple(t.tool_id for t in values['tools']))
    select_plan_tools = curate_request_tools

    def evaluate_tactical_progress(self, **values):
        return SystemOneTacticalReview(True, .99, True, answers={'observed': .99})

    def evaluate(self, **values):
        evidence = values['evidence']['entries']
        if not any(e.get('tool_id') == 'native.bash' and e.get('status') == 'success'
                   and 'temperature_c' in e.get('result', {}).get('stdout', '') for e in evidence):
            return SystemOneReviewResult((), ())
        refs = [e['evidence_ref'] for e in evidence if e.get('status') == 'success']
        return SystemOneReviewResult(tuple({'criterion_id': c['id'], 'status': 'satisfied',
                                           'evidence_refs': refs} for c in values['contract']['criteria']), ())

    def recommend_act(self, **values):
        if values['state']['check_verdict'] == 'wip':
            return SystemOneActRecommendation('fail', .99, True, 'Both available routes failed; restore sensor connectivity.',
                                             answers={'closure_explanation_is_warranted': .99})
        return SystemOneActRecommendation('complete', .99, True, 'Fixture verified', answers={})


class ScriptedSystemTwo:
    """Explicit deterministic doubles; requests and command evidence are retained."""
    def __init__(self, trial, task, *, fresh=False):
        self.trial, self.task, self.fresh = trial, task, fresh
        self.plans = self.calls = 0
        self.prompts = []

    def generate_markdown(self, request):
        if request.purpose == InferencePurpose.ACCEPTANCE_CRITERIA:
            content = '1.- [ ] Obtain and report the verified fresh reading' if self.fresh else '1.- [ ] Install the exact approved recovered workflow'
        elif self.task.status == 'failed':
            content = 'Both primary and backup routes are unavailable. I could not obtain a fresh reading or install a verified workflow.'
        else:
            content = 'I will check the study sensor.'
        return InferenceResult(content=content, model_profile=request.model_profile)

    def generate_json(self, request):
        self.prompts.append(request.prompt)
        if request.purpose == InferencePurpose.PHASE_PLANNING:
            self.plans += 1
            assert self.plans <= BUDGET['planning_calls'], 'fixed planning budget exhausted'
            if self.fresh:
                assert self.trial.draft['instructions'] in request.prompt
                steps, phase = ['read', 'respond'], 'fresh-reuse'
            elif self.task.metadata.get('act_directive', {}).get('closure_only'):
                steps, phase = ['respond'], 'explain-unresolved'
            elif self.task.metadata.get('v3_resumption_evidence'):
                assert 'authenticated_question_answer' in request.prompt or 'Approve installation' in request.prompt
                steps, phase = ['install', 'respond'], 'approved-install'
            elif not entries(self.task):
                steps, phase = ['primary'], 'initial-route'
            elif any(e.get('subgoal_id') == 'backup' for e in entries(self.task)):
                steps, phase = ['diagnose'], 'inspect-remaining-routes'
            else:
                steps, phase = ['backup', 'approve'], 'recover-and-draft'
            value = self.plan(steps, phase)
        else:
            self.calls += 1
            assert self.calls <= BUDGET['tactical_calls'], 'fixed tactical budget exhausted'
            stage = request.metadata['subgoal_id']
            value = self.action(stage)
        return InferenceResult(json_value=value, model_profile=request.model_profile)

    def plan(self, steps, phase):
        goals = []
        for step in steps:
            capability = {'approve': 'user_interaction', 'install': 'skill_management',
                          'respond': 'user_response'}.get(step, 'local_shell')
            effect = 'local_skill_installation' if step == 'install' else 'user_response' if step in ('approve', 'respond') else 'read_only'
            goals.append({'subgoal_id': step, 'objective': step, 'required_output_type': step,
                          'depends_on': [goals[-1]['subgoal_id']] if goals else [],
                          'allowed_capabilities': [capability], 'allowed_side_effects': [effect],
                          'completion': {'kind': 'field_equals', 'field': 'exit_code', 'expected': 0}
                          if step in ('primary', 'backup', 'read', 'diagnose') else {'kind': 'output_present', 'output_type': step}})
        return {'phase_id': phase, 'objective': 'Recover, approve and reuse a standard skill',
                'criterion_ids': ['ac-1'], 'acceptance_criteria': ['Verified requested outcome'],
                'authorized_capabilities': list({g['allowed_capabilities'][0] for g in goals}),
                'mutation_scope': {'allow_external_effects': True}, 'subgoals': goals}

    def action(self, stage):
        if stage in ('primary', 'backup', 'read', 'diagnose'):
            arguments = {'command': self.trial.command('--help' if stage == 'diagnose' else 'primary' if stage == 'primary' else 'backup'),
                         'timeout_seconds': BUDGET['command_timeout_seconds']}
            if stage == 'read':
                arguments['skill_id'] = SKILL
            tool = 'native.bash'
        elif stage == 'approve':
            tool, arguments = 'native.ask_question', {'question': self.trial.question_text, 'question_kind': 'yes_no'}
        elif stage == 'install':
            tool, arguments = 'native.skill_install', self.trial.draft
        else:
            if self.fresh:
                state = self.task.hierarchical_state
                result = next(a['result'] for a in state['actions'] if a['subgoal_id'] == 'read')
                observed = json.loads(result['stdout'])
                message = f"Study: {observed['temperature_c']} C from {observed['source']} at {observed['observed_at']}."
            elif self.task.metadata.get('act_directive', {}).get('closure_only'):
                message = 'Both primary and backup routes are unavailable. I could not obtain a fresh reading or install a verified workflow. Restore sensor connectivity before trying again.'
            else:
                message = 'Installed and read back the exact approved study-temperature workflow.'
            tool, arguments = 'native.respond', {'message': message}
        return {'tool_id': tool, 'arguments': arguments, 'acceptance_questions': QUESTIONS}


class Trial:
    def __init__(self, root, *, offline=False):
        self.root = root
        self.skills = SkillStore(root / 'temporary skills with spaces')
        self.skills._bundled_skills_dir = lambda: root / 'no-bundled'
        self.users = V2UserStore(root / 'users.sqlite')
        self.users.onboard(display_name='Fixture admin', user_id='fixture-admin', users_root=root / 'profiles')
        self.questions = SQLiteQuestionStore(root / 'questions.sqlite')
        self.host = root / 'fixture host with spaces'
        self.host.mkdir()
        (self.host / 'sensor.py').write_text(SENSOR)
        (self.host / 'sensor.json').write_text(json.dumps({'temperature_c': 24}))
        if offline:
            (self.host / 'backup-offline').touch()
        self.draft = {'name': 'study-temperature', 'description': 'Read the current study temperature using the recovered sensor route.',
                      'instructions': 'The primary route failed. Prefer the verified backup route. Run '
                                      + self.command('backup') + '. Require exit 0 and report the temperature, source and timestamp '
                                      'from the fresh JSON response. If unavailable, report the failure; do not fabricate a reading. '
                                      'Do not repeat the failed primary route without new evidence.'}
        self.question_text = 'Approve installation of this exact complete skill draft?\n' + json.dumps(self.draft, indent=2)
        self.started = monotonic()

    def command(self, route):
        return shlex.join([sys.executable, str(self.host / 'sensor.py'), route])

    def task_context(self, identity, *, fresh=False):
        task = TaskState(task_id=identity, message_id=identity, user='fixture-admin',
                         goal='What is the study temperature now?' if fresh else
                         'Read the study temperature; recover if needed, show a complete reusable skill draft and ask before installing.',
                         metadata={'routing_disposition': 'pdca_task'})
        task.append_conversation_message('fixture-admin', task.goal)
        context = CoreLoopContext(messages=InMemoryMessageQueue(), skill_store=self.skills,
                                  question_store=self.questions,
                                  tools=build_native_tool_registry(skill_store=self.skills, user_store=self.users),
                                  system_one=Decisions(), cancellation_checker=lambda: monotonic() - self.started > BUDGET['wall_seconds'])
        context.inference = ScriptedSystemTwo(self, task, fresh=fresh)
        return task, context


def run_trial(*, offline=False):
    """Return an evidence receipt only after all temporary stores are removed."""
    with tempfile.TemporaryDirectory(prefix='skill-reuse-28-') as directory:
        trial = Trial(Path(directory), offline=offline)
        processor = HierarchicalCAPDProcessor()
        task, context = trial.task_context('learn')
        processor.process(task, context)
        initial = task.to_dict()
        pending = trial.questions.list_pending_for_respondent('fixture-admin')
        approval = None
        fresh = None
        fresh_provider = None
        boundary = datetime.now(timezone.utc)
        checks = {'temporary_stores_only': True, 'recovery_parked_for_approval': task.status == 'waiting_user',
                  'no_install_before_approval': not trial.skills.list_skills()}
        if pending:
            question = pending[0]
            assert question.message == trial.question_text
            assert not trial.questions.route_answer(respondent_user_id='stranger', question_id=question.question_id,
                                                    payload={'answer': True}).handled
            answer = trial.questions.route_answer(respondent_user_id='fixture-admin', question_id=question.question_id,
                                                  payload={'answer': True})
            task = answer.resumed_task
            approval = task.metadata['v3_resumption_evidence'][0]
            context.inference.task = task
            processor.process(task, context)
            duplicate = trial.questions.route_answer(respondent_user_id='fixture-admin', question_id=question.question_id,
                                                     payload={'answer': True})
            checks['duplicate_answer_ignored'] = not duplicate.handled
            installed = trial.skills.get(SKILL)
            checks['exact_install_readback'] = bool(installed and installed.description == trial.draft['description']
                                                   and installed.instructions == trial.draft['instructions'])
            checks['learning_completed'] = task.status == 'completed'
            # A new task gets no old plan, acceptance contract, selection or answer.
            (trial.host / 'sensor.json').write_text(json.dumps({'temperature_c': 27}))
            fresh, fresh_context = trial.task_context('fresh', fresh=True)
            fresh_provider = fresh_context.inference
            processor.process(fresh, fresh_context)
            selected = fresh.metadata.get('v3_skill_selection', {})
            reads = [a for a in fresh.hierarchical_state['actions'] if a['tool_id'] == 'native.bash']
            checks.update(fresh_task_completed=fresh.status == 'completed',
                          ordinary_discovery=selected.get('loaded_skill_ids') == [SKILL],
                          no_project_required=not fresh.project_id,
                          single_backup_read=len(reads) == 1 and reads[0]['arguments']['command'] == trial.command('backup'),
                          no_artifact_argument=all('artifact_id' not in a['arguments'] for a in reads),
                          one_install=len(trial.skills.list_skills()) == 1)
            if reads and reads[0]['status'] == 'success':
                reading = json.loads(reads[0]['result']['stdout'])
                checks['fresh_result'] = reading['temperature_c'] == 27 and datetime.fromisoformat(reading['observed_at']) >= boundary
                message = fresh.metadata.get('prepared_user_response', {}).get('message', '')
                checks['accurate_final_response'] = all(str(reading[k]) in message for k in ('temperature_c', 'source', 'observed_at'))
        if offline:
            checks = {'no_install': not trial.skills.list_skills(), 'terminal_failure': task.status == 'failed',
                      'failed_routes_preserved': len([e for e in entries(task) if e.get('status') == 'failed']) == 2,
                      'blocker_explained': 'unavailable' in task.metadata.get('v3_failure_response', {}).get('message', '')}
        report = {'mode': 'deterministic-models-real-processor-and-native-shell', 'budgets': BUDGET,
                  'mechanical_pass': all(checks.values()), 'checks': checks, 'initial_task': initial,
                  'approval': approval, 'learning_task': task.to_dict(), 'fresh_task': fresh.to_dict() if fresh else None,
                  'calls': {'learning': {'planning': context.inference.plans, 'tactical': context.inference.calls},
                            'fresh': {'planning': fresh_provider.plans, 'tactical': fresh_provider.calls} if fresh_provider else None}}
        # Portable receipts exclude temporary root and host-specific interpreter.
        encoded = json.dumps(report, default=str).replace(str(trial.root), '<temporary-root>').replace(sys.executable, '<python>')
    return json.loads(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Make both sensor routes fail')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_trial(offline=args.offline)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'mechanical_pass': report['mechanical_pass'], 'checks': report['checks']}))
    return 0 if report['mechanical_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
