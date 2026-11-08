"""Compare single versus bounded independent reads in the real tactical loop.

All provider responses and fixture data are scripted; this is not a live-model
or household-reliability qualification.
"""
from __future__ import annotations

from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseSubgoal, SideEffectClass, new_tactical_state
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.system_one import SystemOneTacticalReview

QUESTION = {'question_id': 'result', 'type': 'noul', 'instructions': 'Does the result meet the stage?',
            'criteria': {'true': 'The stage is met.', 'false': 'The stage is incomplete.'}}
RECORDS = {'a': 'synthetic-one', 'b': 'synthetic-two', 'c': 'synthetic-three'}


def run_case(scenario: str, sequence: bool, *, inference=None, system_one=None) -> dict:
    if scenario not in {'normal', 'uncertain', 'revoked', 'cancelled'}:
        raise ValueError('unknown_scenario')
    observed, compositions, reviews, baseline_violations = [], [], [], []
    registry = InMemoryToolRegistry()
    def read(args):
        key = args['key']
        observed.append((key, RECORDS[key]))
        if scenario == 'revoked' and len(observed) == 1:
            registry.register(replace(registry.get_definition('native.fixture_read'), enabled=False))
        return {'key': key, 'value': RECORDS[key]}
    descriptor = ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE, read_only=True,
                                description='Read one of the independent synthetic records a, b, or c.',
                                argument_schema={'type': 'object', 'properties': {'key': {'type': 'string',
                                    'enum': ['a', 'b', 'c']}}, 'required': ['key']})
    registry.register(ToolDefinition(descriptor, read))
    subgoal = PhaseSubgoal('read', 'Read all three independent records', 'records',
                           allowed_capabilities=('native.fixture_read',), allowed_side_effects=(SideEffectClass.READ_ONLY,))
    state = new_tactical_state(PhasePlan('fixture', 'Read independent records', (subgoal,),
                                        authorized_capabilities=('native.fixture_read',)))
    state.revealed_tool_ids = ['native.fixture_read']
    def action(key):
        return {'tool_id': 'native.fixture_read', 'arguments': {'key': key}, 'acceptance_questions': [QUESTION]}
    class Provider:
        def generate_json(self, request):
            compositions.append(1)
            remaining = [key for key in RECORDS if key not in {key for key, _ in observed}]
            if not remaining:
                return SimpleNamespace(json_value=None)
            result = {'actions': [action(key) for key in remaining]} if sequence and len(compositions) == 1 else action(remaining[0])
            return SimpleNamespace(json_value=result)
    def review(**_):
        reviews.append(1)
        done = len(observed) == len(RECORDS)
        progress = .5 if scenario == 'uncertain' and len(observed) == 1 else .99
        return SystemOneTacticalReview(done, .99, True, answers={
            'runtime_action_added_progress': progress, 'runtime_declared_subgoal_complete': .99 if done else .01})
    class LiveProvider:
        def generate_json(self, request):
            compositions.append(1)
            if not sequence:
                request = replace(request, prompt=request.prompt.replace(
                    'For a read-only subgoal only, you may instead return {actions:[...]}',
                    'Do not return an actions array. For this comparison, never return {actions:[...]}'))
            result = inference.generate_json(request)
            if not sequence and isinstance(result.json_value, dict) and 'actions' in result.json_value:
                baseline_violations.append(1)
                return SimpleNamespace(json_value=None)
            return result
    class LiveReview:
        def __init__(self, delegate):
            self.delegate, self.settings = delegate, getattr(delegate, 'settings', None)
        def evaluate_tactical_progress(self, **kwargs):
            reviews.append(1)
            return self.delegate.evaluate_tactical_progress(**kwargs)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
        inference=LiveProvider() if inference is not None else Provider(),
        system_one=LiveReview(system_one) if system_one is not None else SimpleNamespace(evaluate_tactical_progress=review),
        cancellation_checker=lambda: (scenario == 'cancelled' and bool(observed)) or len(compositions) >= 8)
    started = perf_counter()
    failure_type = failure_code = None
    try:
        outcome = PhaseExecutor().run(TaskState(user='fixture', goal='Read each of the independent synthetic records a, b, and c.'), state, context)
        status = outcome.status.value
    except Exception as exc:
        status, failure_type = 'error', type(exc).__name__
        if str(exc) in {'openai_codex_invalid_json', 'openai_codex_invalid_tool_json', 'openai_codex_empty_response', 'openai_codex_timeout'}:
            failure_code = str(exc)
    correct = len(observed) == len(RECORDS) and dict(observed) == RECORDS
    expected = (status == 'phase_complete' and correct if scenario in {'normal', 'uncertain'}
                else status in {'blocked', 'cancelled'} and len(observed) == 1)
    return {'scenario': scenario, 'variant': 'independent_sequence' if sequence else 'single_action',
            'outcome': status, 'failure_type': failure_type, 'failure_code': failure_code, 'oracle_pass': expected,
            'observed_keys': [key for key, _ in observed], 'composition_calls': len(compositions),
            'jev_reviews': len(reviews), 'tool_calls': len(state.actions),
            'baseline_sequence_rejected': bool(baseline_violations),
            'targeted_recoveries': [dict((entry.get('progress_review') or {}).get('uncertainty_recovery') or {})
                                    for entry in state.evidence.entries
                                    if (entry.get('progress_review') or {}).get('uncertainty_recovery')],
            'elapsed_seconds': perf_counter() - started, 'provider_cost_usd': 0 if inference is None else None,
            'cost_basis': 'scripted providers' if inference is None else 'ChatGPT subscription; Jev charge recorded separately'}


def report(repeats: int = 3) -> dict:
    import hashlib
    from pathlib import Path
    from alphonse.agent_v2.core.intelligence.v3 import executor
    rows = []
    for repeat in range(repeats):
        for scenario in ('normal', 'uncertain', 'revoked', 'cancelled'):
            for sequence in (False, True):
                try:
                    row = run_case(scenario, sequence)
                except Exception as exc:
                    row = {'scenario': scenario, 'variant': 'independent_sequence' if sequence else 'single_action',
                           'oracle_pass': False, 'error_type': type(exc).__name__}
                row['repeat'] = repeat + 1
                rows.append(row)
    return {'version': 'independent-reads-v1', 'mode': 'scripted-decisions-real-tactical-executor',
            'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'executor_sha256': hashlib.sha256(Path(executor.__file__).read_bytes()).hexdigest(),
            'limitations': ['Scripted provider decisions do not demonstrate live model behavior.',
                'Read-only fixture operations do not qualify browser, write or external-effect batching.',
                'Elapsed times are local fixture execution, not provider latency or household reliability.'],
            'results': rows}


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps({'runs': len(data['results']), 'passed': sum(bool(row.get('oracle_pass')) for row in data['results'])}))
    return 0 if all(row.get('oracle_pass') for row in data['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
