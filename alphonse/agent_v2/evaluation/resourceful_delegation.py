"""Versioned synthetic integration qualification through the unmodified V3 processor.

System One and Two are explicit scripted doubles. This measures runtime evidence
flow, NOT autonomous model behavior or live-site readiness. No production state
or credentials are loaded. Executable fixture programs exist only in temp storage.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from threading import Thread
from time import perf_counter

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.io.identity import V2IdentityResolver
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.core.scheduled_tasks import ScheduledTaskStore
from alphonse.agent_v2.core.tools.invocation import ToolInvocationService
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.ask_question import build_ask_question_tool_definition
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.respond import build_respond_tool_definition
from alphonse.agent_v2.core.tools.registry.native.scheduled_task import build_scheduled_task_tool_definition
from alphonse.agent_v2.outcome_judgments import OUTCOME_QUESTIONS
from alphonse.agent_v2.system_one import (SystemOneAdmissionDecision, SystemOneActRecommendation,
    SystemOneReviewResult, SystemOneTacticalReview, SystemOneToolRegistrySelection)
from alphonse.agent_v2.users import V2UserStore

CORPUS = Path(__file__).resolve().parents[3] / 'tests/fixtures/resourceful_delegation/v1.json'
QUESTIONS = [{'question_id': 'observed', 'type': 'noul', 'instructions': 'Is this stage evidenced by the result?',
              'criteria': {'true': 'Direct successful evidence.', 'false': 'Missing or failed evidence.'}}]
FORM_KINDS = {'form', 'absent', 'broken', 'changed', 'restart', 'clarify'}


def entries(task):
    rows = [e for h in task.metadata.get('v3_phase_history', []) for e in h.get('evidence', {}).get('entries', [])]
    # Resumed phases import prior evidence; count each actual observation once.
    return list({e['evidence_ref']: e for e in rows}.values())


@contextmanager
def fixture_server(kind, *, identity='Synthetic Alex'):
    """Loopback-only service; POST independently validates discovered field names."""
    receipts = []
    field = 'attendee_label' if kind == 'changed' else 'full_name'

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            status = 403 if kind == 'denied' else 503 if kind == 'unavailable' else 200
            receipts.append({'method': 'GET', 'status': status, **({'places': []} if kind == 'empty' else {})})
            self.send_response(status)
            self.end_headers()
            body = {'places': []} if kind == 'empty' else None
            self.wfile.write(json.dumps(body).encode() if body is not None else
                             f'<form action="/register" method="post"><input required name="{field}"><button>Join</button></form>'.encode())

        def do_POST(self):
            from urllib.parse import parse_qs
            values = parse_qs(self.rfile.read(int(self.headers.get('Content-Length', '0'))).decode())
            valid = self.path == '/register' and values == {field: [identity]}
            receipts.append({'method': 'POST', 'status': 201 if valid else 422, 'values': values})
            self.send_response(201 if valid else 422)
            self.end_headers()
            self.wfile.write(json.dumps({'registered': valid, 'receipt': 'synthetic-registration' if valid else None}).encode())

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', receipts
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class ScriptedDecisions:
    """Bounded controlled judgments; all qualification oracles remain outside these."""
    requires_mission_outcome_review = True

    def __init__(self, trial):
        self.trial = trial

    def classify_task_admission(self, **_):
        return SystemOneAdmissionDecision(True, .99, True)

    def curate_request_tools(self, *, tools, **_):
        return SystemOneToolRegistrySelection(selected_tool_ids=tuple(t.tool_id for t in tools))
    select_plan_tools = curate_request_tools

    def evaluate_tactical_progress(self, **_):
        return SystemOneTacticalReview(True, .99, True, answers={'observed': .99})

    def evaluate(self, **values):
        done = self.trial.oracle_effect()
        refs = [e['evidence_ref'] for e in values['evidence']['entries'] if e.get('status') == 'success']
        return SystemOneReviewResult(tuple({'criterion_id': c['id'], 'status': 'satisfied', 'evidence_refs': refs}
            for c in values['contract']['criteria']) if done else (), ())

    def recommend_act(self, **values):
        done = self.trial.oracle_effect()
        answers = {key: .01 for key in OUTCOME_QUESTIONS}
        answers.update(mission_outcome_is_fulfilled=.99 if done else .01,
                       mission_requires_execution=.01 if self.trial.kind == 'proposal' else .99,
                       mission_requires_recurring_setup=.99 if self.trial.kind == 'schedule' else .01,
                       capability_claims_are_supported=.99, lookup_claims_are_supported=.99,
                       continuation_is_worthwhile=.99)
        if self.trial.kind in {'denied', 'unavailable'}:
            answers.update(blocker_is_supported=.99, closure_explanation_is_warranted=.99)
            return SystemOneActRecommendation('fail', .99, True, 'Observed service blocker', answers=answers)
        return SystemOneActRecommendation('complete' if done else 'replan', .99, True,
                                           'Controlled synthetic judgment', answers=answers)


class ScriptedSystemTwo:
    def __init__(self, trial):
        self.trial = trial
        self.requests = []
        self.cancelled = False
        self.invalid_choices = 0

    def generate_markdown(self, request):
        self.requests.append({'purpose': request.purpose.value, 'prompt': request.prompt})
        content = '- [ ] Fulfill the original delegated outcome with verified effects'
        if request.purpose != InferencePurpose.ACCEPTANCE_CRITERIA:
            content = self.trial.response()
        return InferenceResult(content=content)

    def generate_json(self, request):
        self.requests.append({'purpose': request.purpose.value, 'prompt': request.prompt})
        if len(self.requests) > 24:
            raise RuntimeError('evaluation_inference_budget_exhausted')
        if request.purpose == InferencePurpose.PHASE_PLANNING:
            value = self.plan()
        else:
            stage = request.metadata['subgoal_id']
            value = self.action(stage)
        return InferenceResult(json_value=value)

    def plan(self):
        trial = self.trial
        seen = entries(trial.task)
        if trial.task.metadata.get('act_directive', {}).get('closure_only'):
            steps = ['respond']
        elif trial.kind in {'denied', 'unavailable'} and seen:
            steps = ['respond']
        elif trial.kind == 'notes':
            steps = ['notes', 'respond']
        elif trial.kind == 'schedule':
            steps = ['schedule', 'respond']
        elif trial.kind == 'proposal':
            steps = ['respond']
        elif trial.kind in {'denied', 'unavailable', 'empty', 'invalid', 'cancel'}:
            steps = ['inspect', 'respond']
        elif trial.kind in {'absent', 'broken'} and not seen:
            steps = ['dependency']
        elif trial.kind == 'clarify' and not trial.task.metadata.get('v3_resumption_evidence'):
            steps = ['inspect', 'ask']
        else:
            steps = ['inspect', 'submit', 'respond'] if not seen else ['submit', 'respond']
        goals = []
        for stage in steps:
            capability = {'ask': 'user_interaction', 'respond': 'user_response', 'schedule': 'scheduling'}.get(stage, 'local_shell')
            effect = {'ask': 'user_response', 'respond': 'user_response', 'schedule': 'external_reversible',
                      'submit': 'external_irreversible', 'notes': 'project_mutation'}.get(stage, 'read_only')
            goals.append({'subgoal_id': stage, 'objective': stage, 'required_output_type': stage,
                          'allowed_capabilities': [capability], 'allowed_side_effects': [effect],
                          'depends_on': [goals[-1]['subgoal_id']] if goals else [],
                          'completion': {'kind': 'output_present', 'output_type': stage}})
        return {'phase_id': 'phase-' + str(len(seen)), 'objective': trial.task.goal,
                'criterion_ids': ['ac-1'], 'acceptance_criteria': ['Verified delegated outcome'],
                'authorized_capabilities': list({g['allowed_capabilities'][0] for g in goals}),
                'mutation_scope': {'allowed_paths': ['workshop-notes.md'], 'allow_external_effects': True}, 'subgoals': goals}

    def action(self, stage):
        trial = self.trial
        if trial.kind == 'cancel':
            self.cancelled = True
        if trial.kind == 'invalid':
            self.invalid_choices += 1
            return {'tool_id': 'not-a-tool', 'arguments': {}}
        if stage == 'respond':
            tool, args = 'native.respond', {'message': trial.response()}
        elif stage == 'ask':
            tool, args = 'native.ask_question', {'question': 'What name should the workshop registration use?', 'question_kind': 'open_text'}
        elif stage == 'schedule':
            tool, args = 'native.scheduled_task', {'name': 'Workshop availability', 'prompt': 'Check ' + trial.url,
                'schedule_kind': 'rrule', 'rrule': 'FREQ=DAILY;BYHOUR=9;BYMINUTE=0;BYSECOND=0',
                'dtstart': '2030-01-01T09:00:00+00:00', 'timezone': 'UTC'}
        else:
            tool, args = 'native.bash', {'command': trial.command(stage), 'timeout_seconds': 5}
        return {'tool_id': tool, 'arguments': args, 'acceptance_questions': QUESTIONS}


class SimulatedRestart(BaseException):
    pass


class RestartStore(SQLiteQuestionStore):
    def save_task_checkpoint(self, task, **kwargs):
        result = super().save_task_checkpoint(task, **kwargs)
        state = task.hierarchical_state
        if getattr(self, 'crash_once', False) and state.get('completed_subgoal_ids') == ['inspect']:
            self.crash_once = False
            raise SimulatedRestart()
        return result


class Trial:
    def __init__(self, root, case, url, receipts):
        self.root, self.case, self.kind, self.url, self.receipts = root, case, case['kind'], url, receipts
        self.users = V2UserStore(root / 'users.sqlite')
        self.users.set_users_root(root / 'users')
        for owner in ('synthetic-alex', 'synthetic-other'):
            self.users.create_user(display_name=owner, user_id=owner)
        self.questions = RestartStore(root / 'questions.sqlite')
        self.questions.crash_once = self.kind == 'restart'
        self.schedules = ScheduledTaskStore(root / 'schedules.sqlite')
        self.deliveries = []
        self.started = perf_counter()
        self.restarted = False
        self.isolation_verified = False
        self.interventions = 0
        self.task = TaskState(task_id=case['id'], message_id=case['id'], user='synthetic-alex',
                              goal=case['request'].format(url=url), metadata={'routing_disposition': 'pdca_task'})
        self.task.append_conversation_message('synthetic-alex', self.task.goal)
        registry = InMemoryToolRegistry()
        for definition in (build_bash_tool_definition(), build_respond_tool_definition(),
                           build_ask_question_tool_definition(), build_scheduled_task_tool_definition()):
            registry.register(definition)
        self.inference = ScriptedSystemTwo(self)
        self.context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
            identity_resolver=V2IdentityResolver(user_store=self.users), question_store=self.questions,
            schedule_store=self.schedules, system_one=ScriptedDecisions(self), inference=self.inference,
            delivery_sink=lambda payload: self.deliveries.append(payload) or {'delivered': True},
            cancellation_checker=lambda: self.inference.cancelled or perf_counter() - self.started > 30)
        self.workspace = self.users.ordinary_workspace('synthetic-alex')
        if self.kind == 'broken':
            (self.workspace / 'optional_form_runtime.py').write_text("raise RuntimeError('synthetic initialization failure')\n")

    def command(self, stage):
        # Programs are ephemeral fixture implementations, never registered native tools.
        if stage == 'notes':
            program = "from pathlib import Path; p=Path('workshop-notes.md'); p.open('x').write('Workshop on Friday'); print(p.read_text())"
        elif stage == 'dependency':
            program = """import json
try:
 import optional_form_runtime
 print(json.dumps({'dependency': 'ready'}))
except (ImportError, RuntimeError) as error:
 print(json.dumps({'dependency': type(error).__name__, 'detail': str(error), 'alternative': 'standard library available'}))
"""
        elif stage == 'inspect':
            program = f"from urllib.request import urlopen; print(urlopen({self.url!r}, timeout=2).read().decode())"
        else:
            program = f"""from urllib.request import urlopen, Request
from urllib.parse import urlencode
from html.parser import HTMLParser
class Form(HTMLParser):
 def handle_starttag(self, tag, attrs):
  if tag == 'input': self.field = dict(attrs)['name']
p = Form()
p.feed(urlopen({self.url!r}, timeout=2).read().decode())
request = Request({self.url + '/register'!r}, data=urlencode({{p.field: 'Synthetic Alex'}}).encode())
print(urlopen(request, timeout=2).read().decode())
"""
        return shlex.join([sys.executable, '-c', program])

    def oracle_effect(self):
        if self.kind in FORM_KINDS:
            return len([r for r in self.receipts if r['method'] == 'POST' and r['status'] == 201]) == 1
        if self.kind == 'notes':
            file = self.workspace / 'workshop-notes.md'
            return file.is_file() and file.read_text() == 'Workshop on Friday'
        if self.kind == 'schedule':
            rows = self.schedules.list_tasks()
            return (len(rows) == 1 and rows[0].status == 'active'
                    and rows[0].next_run_at == '2030-01-01T09:00:00+00:00'
                    and rows[0].owner_user_id == 'synthetic-alex' and rows[0].timezone == 'UTC'
                    and rows[0].prompt == 'Check ' + self.url and rows[0].delivery_mode == 'pdca'
                    and rows[0].schedule == {'kind': 'rrule', 'rrule': 'FREQ=DAILY;BYHOUR=9;BYMINUTE=0;BYSECOND=0',
                                             'dtstart': '2030-01-01T09:00:00+00:00'})
        if self.kind == 'proposal':
            return not self.schedules.list_tasks()
        return self.kind == 'empty' and any(r['method'] == 'GET' and r['status'] == 200 and r.get('places') == [] for r in self.receipts)

    def response(self):
        if self.kind == 'empty':
            return 'The successful workshop lookup returned no open places.'
        if self.kind in {'denied', 'unavailable'}:
            return 'The workshop lookup failed with HTTP ' + ('403 access denied' if self.kind == 'denied' else '503 service unavailable') + '; availability remains unknown.'
        if self.kind == 'invalid':
            return 'The lookup was not completed because execution planning failed.'
        if self.kind == 'cancel':
            return 'Cancelled; no registration was submitted.'
        if self.kind == 'schedule':
            return 'The daily workshop check is active, persisted and read back for 09:00 UTC.'
        if self.kind == 'proposal':
            return 'Proposed: check the workshop daily at 09:00 UTC. No schedule is active.'
        if self.kind == 'notes':
            return 'Created workshop-notes.md in your workspace and read back: Workshop on Friday.'
        return 'Registered Synthetic Alex; the workshop returned receipt synthetic-registration.'

    def run(self):
        processor = HierarchicalCAPDProcessor()
        try:
            processor.process(self.task, self.context)
        except SimulatedRestart:
            self.restarted = True
            self.questions = RestartStore(self.root / 'questions.sqlite')
            self.context.question_store = self.questions
            self.task = self.questions.load_task_checkpoint(self.task.task_id)
            HierarchicalCAPDProcessor().process(self.task, self.context)
        if self.task.status == 'waiting_user':
            pending = self.questions.list_pending_for_respondent('synthetic-alex')
            if pending and self.case.get('answer'):
                question_id = pending[0].question_id
                denied = self.questions.route_answer(respondent_user_id='synthetic-other', question_id=question_id, text='Wrong identity')
                self.isolation_verified = not denied.handled
                # Reopen the SQLite store to exercise durable question resumption.
                self.questions = RestartStore(self.root / 'questions.sqlite')
                self.context.question_store = self.questions
                answer = self.questions.route_answer(respondent_user_id='synthetic-alex', question_id=question_id, text=self.case['answer'])
                self.task = answer.resumed_task
                self.interventions += 1
                HierarchicalCAPDProcessor().process(self.task, self.context)
        if self.kind == 'notes':
            other = TaskState(task_id='other', user='synthetic-other', goal='Read my notes')
            result = ToolInvocationService(context=self.context, task=other).execute_or_raise('native.bash', {'command': 'test ! -e workshop-notes.md'})
            self.isolation_verified = result['exit_code'] == 0 and result['cwd'] != str(self.workspace)
        return self.score()

    def score(self):
        task, kind = self.task, self.kind
        evidence = entries(task)
        response = str(task.metadata.get('prepared_user_response', {}).get('message') or
                       task.metadata.get('v3_failure_response', {}).get('message') or '')
        posts = [r for r in self.receipts if r['method'] == 'POST']
        questions = [e for e in evidence if e.get('tool_id') == 'native.ask_question'
                     and e.get('result', {}).get('question_interrupt')]
        invalid = [e for e in evidence if e.get('status') == 'rejected']
        checks = {'expected_terminal_status': task.status == self.case['expected'],
                  'no_unrequested_schedule': kind == 'schedule' or not self.schedules.list_tasks(),
                  'no_unrequested_submission': kind in FORM_KINDS or not posts,
                  'final_response_accurate': response == self.response() if kind != 'cancel' else not posts,
                  'question_count': len(questions) == (1 if kind == 'clarify' else 0),
                  'bounded_lookup_attempts': len([r for r in self.receipts if r['method'] == 'GET']) <= (1 if kind in {'denied', 'unavailable', 'empty'} else 3)}
        if self.case['expected'] == 'completed':
            checks['independent_effect_verification'] = self.oracle_effect()
        if kind in FORM_KINDS:
            checks['exactly_one_valid_submission'] = len(posts) == 1 and posts[0]['status'] == 201
        if kind in {'denied', 'unavailable'}:
            checks['failed_lookup_not_empty'] = any(e.get('status') == 'failed' for e in evidence) and not posts and 'unknown' in response
        if kind in {'absent', 'broken'}:
            checks['dependency_evidence_retained'] = any(('ModuleNotFoundError' if kind == 'absent' else 'RuntimeError') in json.dumps(e.get('result')) for e in evidence)
        if kind == 'invalid':
            checks['bounded_invalid_plan'] = len(invalid) == 2 and self.inference.invalid_choices == 2 and not self.receipts
        if kind == 'cancel':
            checks['cancelled_before_effect'] = not self.receipts and not posts
        if kind == 'restart':
            checks['checkpoint_resumed'] = self.restarted
            checks['inspection_not_replayed'] = len([e for e in evidence if e.get('subgoal_id') == 'inspect' and e.get('status') == 'success']) == 1
        if kind in {'clarify', 'notes'}:
            checks['owner_isolation'] = self.isolation_verified
        metrics = {'verified_task_completion': task.status == 'completed' and self.oracle_effect(),
                   # Scripted prose has exact per-case ground truth; no keyword classifier.
                   'unsupported_inability_claims': int(kind not in {'denied', 'unavailable', 'invalid', 'cancel'} and response != self.response()),
                   'unnecessary_questions': max(0, len(questions) - (1 if kind == 'clarify' else 0)),
                   'invalid_plan_proposals': len(invalid),
                   'repeated_invalid_plan_proposals': max(0, len(invalid) - 1),
                   'repeated_invalid_dispatches': len([e for e in evidence if e.get('tool_id') == 'not-a-tool' and e.get('status') != 'rejected']),
                   'user_interventions': self.interventions,
                   'unexpected_user_interventions': max(0, self.interventions - int(kind == 'clarify')),
                   'latency_seconds': perf_counter() - self.started, 'inference_calls': len(self.inference.requests),
                   'provider_cost_usd': 0, 'cost_basis': 'No network providers; scripted doubles only'}
        return {'id': self.case['id'], 'split': self.case['split'], 'expected': self.case['expected'],
                'checks': checks, 'scenario_pass': all(checks.values()), 'metrics': metrics,
                'http_receipts': self.receipts, 'requests': self.inference.requests,
                'task': task.to_dict(), 'schedules': [s.to_dict() for s in self.schedules.list_tasks()]}


def run_suite(corpus_path=CORPUS):
    raw = Path(corpus_path).read_bytes()
    corpus = json.loads(raw)
    results = []
    for repeat in range(corpus['repeats']):
        for case in corpus['cases']:
            with tempfile.TemporaryDirectory(prefix='resourceful-158-') as directory:
                with fixture_server(case['kind']) as (url, receipts):
                    trial = Trial(Path(directory), case, url, receipts)
                    try:
                        result = trial.run()
                    except Exception as error:
                        result = {'id': case['id'], 'split': case['split'], 'expected': case['expected'],
                                  'scenario_pass': False, 'error': f'{type(error).__name__}: {error}',
                                  'http_receipts': receipts,
                                  'task': trial.task.to_dict(), 'requests': trial.inference.requests}
                    result['repeat'] = repeat + 1
                    # Receipts portable; secrets/production stores are never loaded.
                    encoded = json.dumps(result, default=str).replace(directory, '<temporary-root>').replace(url, '<fixture-url>').replace(sys.executable, '<python>')
                    results.append(json.loads(encoded))
    threshold = corpus['thresholds']
    metrics = [r.get('metrics', {}) for r in results]
    completable = [r for r in results if r['expected'] == 'completed']
    summary = {'scenario_pass_rate': sum(r['scenario_pass'] for r in results) / len(results),
               'verified_completion_rate_for_completable_cases': sum(r.get('metrics', {}).get('verified_task_completion', False) for r in completable) / len(completable)}
    for name in ('unsupported_inability_claims', 'unnecessary_questions', 'repeated_invalid_dispatches', 'unexpected_user_interventions'):
        summary[name] = sum(m.get(name, 0) for m in metrics)
    summary.update(max_case_seconds=max(m.get('latency_seconds', 0) for m in metrics),
                   max_inference_calls=max(m.get('inference_calls', 0) for m in metrics),
                   max_provider_cost_usd=sum(m.get('provider_cost_usd', 0) for m in metrics))
    gates = {key: summary[key] >= value if key.endswith('_rate') or key.endswith('_cases') else summary[key] <= value for key, value in threshold.items()}
    # Missing metrics or exceptions can never pass by contributing zero to a sum.
    gates['all_trials_recorded'] = all('metrics' in r and 'error' not in r for r in results)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], text=True).strip()
    return {'version': corpus['version'], 'mode': corpus['mode'], 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
            'revision': revision, 'tracked_tree_dirty': bool(dirty), 'python': sys.version,
            'corpus_sha256': hashlib.sha256(raw).hexdigest(), 'thresholds': threshold, 'summary': summary,
            'gates': gates, 'qualified_synthetic_runtime': all(gates.values()),
            'live_model_qualified': False, 'live_site_qualified': False,
            'limitations': ['System One judgments, System Two plans and prose are scripted.',
                            'Held-out split tests fixture variation, not unseen model generalization.',
                            'No browser JavaScript, accounts, production daemon ingress, live providers or live sites are qualified.',
                            'Bash is a trusted host capability; disposable fixtures are not an OS security sandbox.'],
            'results': results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=CORPUS)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_suite(args.corpus)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('qualified_synthetic_runtime', 'summary', 'gates')}))
    return 0 if report['qualified_synthetic_runtime'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
