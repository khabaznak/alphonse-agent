"""Feasibility regression seams with real disposable HTTP and shell execution.

System Two choices and Jev judgments are scripted: these check evidence flow,
prerequisites and effects, not the live models' interpretation of natural language.
"""
from contextlib import contextmanager
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import shlex
import sys
from threading import Thread
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.contracts import PhasePlan, PhaseSubgoal, PhaseStatus, new_tactical_state
from alphonse.agent_v2.core.intelligence.v3.executor import PhaseExecutor
from alphonse.agent_v2.core.intelligence.v3.feasibility import FEASIBILITY_GUIDANCE, CAPABILITY_REPORTING_GUIDANCE
from alphonse.agent_v2.core.intelligence.v3.planner import plan_phase
from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.intelligence.v3.revealing import ToolRevealPolicy
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry
from alphonse.agent_v2.core.tools.registry.native.bash import build_bash_tool_definition
from alphonse.agent_v2.core.tools.registry.native.scheduled_task import build_scheduled_task_tool_definition
from alphonse.agent_v2.system_one import SystemOneToolRegistrySelection, SystemOneTacticalReview


@contextmanager
def web_form():
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(('GET', self.path))
            self.send_response(403 if self.path == '/denied' else 200)
            self.end_headers()
            self.wfile.write(b'<form method="post"><input name="full_name" required><button>Submit</button></form>')

        def do_POST(self):
            requests.append(('POST', self.path))
            self.send_response(200)
            self.end_headers()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


class Jev:
    def curate_request_tools(self, *, tools, **_):
        return SystemOneToolRegistrySelection(selected_tool_ids=tuple(t.tool_id for t in tools))

    select_plan_tools = curate_request_tools

    def evaluate_tactical_progress(self, **_):
        # Deliberately optimistic: the deterministic failed-command gate must win.
        return SystemOneTacticalReview(True, 0.99, True)


class Inference:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.requests = []

    def generate_json(self, request):
        self.requests.append(request)
        return SimpleNamespace(json_value=next(self.outputs))

    def generate_markdown(self, request):
        self.requests.append(request)
        return SimpleNamespace(content='The attempted check failed; the requested operation remains unverified.')


def run_probe(tmp_path, goal, program, *, timeout_seconds=5):
    task = TaskState(task_id='feasibility', user='fixture', goal=goal, status='running')
    task.set_acceptance_contract_from_markdown('- [ ] Assess the form and report verified prerequisites')
    registry = InMemoryToolRegistry()
    registry.register(build_bash_tool_definition())
    phase = PhasePlan('inspect', 'Assess feasibility without submission', (PhaseSubgoal(
        'probe', 'Inspect prerequisites and form without personal data', 'inspection',
        allowed_capabilities=('local_shell',),
    ),), authorized_capabilities=('local_shell',))
    action = {'tool_id': 'native.bash', 'arguments': {
        'command': shlex.join([sys.executable, '-c', program]),
        'cwd': str(tmp_path), 'timeout_seconds': timeout_seconds,
    }, 'acceptance_questions': [{
        'question_id': 'inspected', 'type': 'noul',
        'instructions': 'Does the result establish form feasibility prerequisites?',
        'criteria': {'true': 'Observed form and runtime are reported.', 'false': 'Inspection failed.'},
    }]}
    inference = Inference(phase.to_dict(), action)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=Jev(), inference=inference)
    state = new_tactical_state(plan_phase(task, context))
    outcome = PhaseExecutor(reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert len(inference.requests) == 2
    assert all(FEASIBILITY_GUIDANCE in request.prompt for request in inference.requests)
    assert len(state.actions) == 1  # no install, retries, form submission or scheduling
    return task, state, outcome, context


@pytest.mark.parametrize('dependency', ['installed', 'absent', 'broken'])
def test_dependency_evidence_and_missing_personal_input_do_not_submit(tmp_path, dependency):
    # All dependency programs are minimal disposable fixtures, never product features.
    if dependency == 'installed':
        (tmp_path / 'form_runtime.py').write_text('ready = True\n')
    elif dependency == 'broken':
        (tmp_path / 'form_runtime.py').write_text("raise RuntimeError('synthetic runtime failed to initialize')\n")
    with web_form() as (url, requests):
        goal = f'Can you fill in the form at {url} for me? I have not supplied my name yet.'
        assert all(name not in goal.lower() for name in ('playwright', 'selenium', 'urllib', 'form_runtime'))
        program = f'''
import form_runtime
from urllib.request import urlopen
from html.parser import HTMLParser
import json
class Form(HTMLParser):
    fields = []
    def handle_starttag(self, tag, attrs):
        if tag == 'input': self.fields.append(dict(attrs)['name'])
parser = Form()
with urlopen({url!r}, timeout=2) as response:
    parser.feed(response.read().decode())
print(json.dumps({{'fields': parser.fields, 'draft': {{'full_name': '<FULL_NAME>'}}, 'submitted': False}}))
'''
        task, state, outcome, context = run_probe(tmp_path, goal, program)
        entry = state.evidence.entries[-1]
        if dependency == 'installed':
            assert outcome.status == PhaseStatus.PHASE_COMPLETE
            assert json.loads(entry['result']['stdout']) == {
                'fields': ['full_name'], 'draft': {'full_name': '<FULL_NAME>'}, 'submitted': False,
            }
            assert requests == [('GET', '/')]
        else:
            assert outcome.status == PhaseStatus.BLOCKED
            assert entry['status'] == 'failed'
            assert entry['result']['exit_code'] != 0
            assert ('ModuleNotFoundError' if dependency == 'absent' else 'RuntimeError') in entry['error']
            assert state.completed_subgoal_ids == []
            assert requests == []
            task.hierarchical_state = state.to_dict()
            task.status = 'failed'
            task.outcome = {'status': 'failure', 'reason': outcome.reason}
            HierarchicalCAPDProcessor._result(task, context)
            assert CAPABILITY_REPORTING_GUIDANCE in context.inference.requests[-1].prompt
            assert task.status == 'failed'
        assert not any(method == 'POST' for method, _ in requests)


def test_standard_library_composition_can_inspect_without_optional_dependency(tmp_path):
    with web_form() as (url, requests):
        _, state, outcome, _ = run_probe(tmp_path, f'Could you handle this web form: {url}?', f'''
from urllib.request import urlopen
with urlopen({url!r}, timeout=2) as response:
    print(response.read().decode())
''')
        assert outcome.status == PhaseStatus.PHASE_COMPLETE
        assert 'full_name' in state.evidence.entries[-1]['result']['stdout']
        assert requests == [('GET', '/')]


def test_explicit_denial_remains_failed_even_with_optimistic_jev(tmp_path):
    with web_form() as (url, requests):
        _, state, outcome, _ = run_probe(tmp_path, f'Can you work with {url}/denied?', f'''
from urllib.request import urlopen
urlopen({url + '/denied'!r}, timeout=2)
''')
        assert outcome.status == PhaseStatus.BLOCKED
        assert '403' in state.evidence.entries[-1]['error']
        assert state.completed_subgoal_ids == []
        assert requests == [('GET', '/denied')]


def test_capability_only_recurring_question_does_not_reveal_or_dispatch_schedule():
    task = TaskState(task_id='assess', user='fixture',
                     goal='Is it possible for you to fill out a web form every week? Assess feasibility only.',
                     status='running')
    registry = InMemoryToolRegistry()
    registry.register(build_bash_tool_definition())
    registry.register(build_scheduled_task_tool_definition())
    phase = PhasePlan('assess', 'Assess prerequisites only', (PhaseSubgoal(
        'inspect', 'Check local runtime readiness', 'readiness', allowed_capabilities=('local_shell',),
    ),), authorized_capabilities=('local_shell',))
    state = new_tactical_state(phase)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, system_one=Jev())
    # Even an invalid tactical choice cannot escape the assessment phase.
    outcome = PhaseExecutor(action_selector=lambda *_: {
        'tool_id': 'native.scheduled_task', 'arguments': {'action': 'create'},
    }, reveal_policy=ToolRevealPolicy()).run(task, state, context)
    assert outcome.status == PhaseStatus.BLOCKED
    assert state.actions == []
    assert 'native.scheduled_task' not in state.revealed_tool_ids
    assert len(state.evidence.entries) == 2  # one correction, then stop
    assert all(entry['status'] == 'rejected' for entry in state.evidence.entries)


def test_runtime_probe_timeout_is_failed_and_not_retried(tmp_path):
    _, state, outcome, _ = run_probe(tmp_path, 'Could you handle a web form?',
                                    'import time; time.sleep(10)', timeout_seconds=0.1)
    assert outcome.status == PhaseStatus.BLOCKED
    assert len(state.actions) == 1
    assert state.evidence.entries[-1]['result']['timed_out'] is True


pytestmark = pytest.mark.usefixtures("task_understanding_provider")
