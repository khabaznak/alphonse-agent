"""Scripted comparison of two tactical shapes on a disposable multi-field form.

Both shapes use the real PhaseExecutor. The batch is a fixture-only prototype,
not a native browser tool, and model answers are scripted. Its callback checks
fixture authorization, cancellation, form version and prerequisites before each
operation; submission remains separate from independent readback.
"""
from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace

from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseSubgoal, SideEffectClass, new_tactical_state
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
from alphonse.agent_v2.evaluation.planning_form_fixture import FormFixture, FixtureDenied
from alphonse.agent_v2.system_one import SystemOneTacticalReview


def run_case(scenario: str, batch: bool, *, inference=None, system_one=None) -> dict:
    if scenario not in {'normal', 'changed', 'denied', 'cancel', 'uncertain', 'cancel_during', 'changed_during', 'denied_during', 'restart'}:
        raise ValueError('unknown_scenario')
    fixture = FormFixture(allowed=scenario != 'denied')
    registry = InMemoryToolRegistry()
    model_calls, reviews = [], []
    original_fill = fixture.fill
    def guarded_fill(actor, version, field_name, value):
        result = original_fill(actor, version, field_name, value)
        if field_name == 'name':
            if scenario == 'cancel_during':
                fixture.cancelled = True
            elif scenario == 'changed_during':
                fixture.change_form({'name': 'Synthetic Robin', 'reference': 'fixture-17'})
            elif scenario == 'denied_during':
                fixture.allowed = False
        return result
    fixture.fill = guarded_fill
    actions = [
        {'operation': 'inspect'},
        *([{'operation': 'batch', 'version': 1}] if batch else [
            {'operation': 'fill', 'version': 1, 'field': 'name', 'value': 'Synthetic Robin'},
            {'operation': 'fill', 'version': 1, 'field': 'code', 'value': 'fixture-17'},
            {'operation': 'submit', 'version': 1}]),
        {'operation': 'verify'},
    ]
    def run(args):
        operation = args['operation']
        if operation == 'inspect':
            result = fixture.inspect('fixture-owner')
            if scenario == 'changed':
                fixture.change_form({'name': 'Synthetic Robin', 'reference': 'fixture-17'})
            if scenario == 'cancel':
                fixture.cancelled = True
            return result
        if operation == 'fill':
            return fixture.fill('fixture-owner', args['version'], args['field'], args['value'])
        if operation == 'submit':
            return fixture.submit('fixture-owner', args['version'], ambiguous=scenario == 'uncertain')
        if operation == 'batch':
            if not batch:
                raise FixtureDenied('batch_not_available')
            # Each sub-operation runs its own fixture checks. Stop on the first
            # exception and never replay a possibly dispatched submission.
            fixture.fill('fixture-owner', args['version'], 'name', 'Synthetic Robin')
            fixture.fill('fixture-owner', args['version'], 'code', 'fixture-17')
            return fixture.submit('fixture-owner', args['version'], ambiguous=scenario == 'uncertain')
        if operation == 'verify':
            return fixture.verify('fixture-owner')
        raise FixtureDenied('invalid_operation')
    operations = ['inspect', 'fill', 'submit', 'verify'] + (['batch'] if batch else [])
    descriptor = ToolDescriptor('native.fixture_form', 'fixture_form', ToolKind.NATIVE,
        description=('Disposable form: inspect returns version and fields; fill requires version, field and value; '
                     'submit and batch require the inspected version; verify reads the result. '
                     'Batch fills both requested fields and submits. Read the result separately to verify submission.'),
        argument_schema={'oneOf': [
            {'type': 'object', 'properties': {'operation': {'const': op}}, 'required': ['operation']}
            for op in operations if op in {'inspect', 'verify'}
        ] + [
            {'type': 'object', 'properties': {'operation': {'const': op}, 'version': {'type': 'integer'}},
             'required': ['operation', 'version']} for op in operations if op in {'submit', 'batch'}
        ] + [{'type': 'object', 'properties': {'operation': {'const': 'fill'}, 'version': {'type': 'integer'},
                    'field': {'type': 'string'}, 'value': {'type': 'string'}},
              'required': ['operation', 'version', 'field', 'value']}]},
        read_only=False, metadata={'side_effect_class': 'external_irreversible'})
    registry.register(ToolDefinition(descriptor, run))
    subgoal = PhaseSubgoal('form', 'Complete and independently verify the fixture form', 'receipt',
        allowed_capabilities=('native.fixture_form',),
        allowed_side_effects=(SideEffectClass.READ_ONLY, SideEffectClass.EXTERNAL_IRREVERSIBLE))
    state = new_tactical_state(PhasePlan('fixture', 'Complete fixture form', (subgoal,),
                                        authorized_capabilities=('native.fixture_form',)))
    state.revealed_tool_ids = ['native.fixture_form']
    def selector(*_):
        model_calls.append(1)
        return {'tool_id': 'native.fixture_form', 'arguments': actions.pop(0)} if actions else None
    def review(**values):
        reviews.append(1)
        done = values['action']['arguments']['operation'] == 'verify' and values['action']['result'].get('submitted') == fixture.fields
        return SystemOneTacticalReview(done, .99, True, answers={
            'runtime_declared_subgoal_complete': .99 if done else .01, 'runtime_action_added_progress': .99})
    class CountedInference:
        def generate_json(self, request):
            model_calls.append(1)
            return inference.generate_json(request)
    class CountedReview:
        def __init__(self, delegate):
            self.delegate = delegate
            self.settings = getattr(delegate, 'settings', None)
        def evaluate_tactical_progress(self, **values):
            reviews.append(1)
            return self.delegate.evaluate_tactical_progress(**values)
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry,
        inference=CountedInference() if inference is not None else None,
        system_one=CountedReview(system_one) if system_one is not None else SimpleNamespace(evaluate_tactical_progress=review),
        cancellation_checker=lambda: fixture.cancelled or len(model_calls) >= 8)
    task = TaskState(user='fixture-owner', goal=('Complete the synthetic form for name Synthetic Robin and code fixture-17. '
        'Inspect the version, fill both fields, submit once, and independently verify the submitted values.'))
    class SimulatedRestart(BaseException):
        pass
    class CheckpointStore:
        saved = None
        def save_task_checkpoint(self, saved_task, **_):
            self.saved = saved_task.to_dict()
            rows = saved_task.hierarchical_state.get('evidence', {}).get('entries', [])
            if scenario == 'restart' and any(r.get('progress_review') for r in rows):
                raise SimulatedRestart()
    checkpoint = CheckpointStore()
    if scenario == 'restart':
        context.question_store = checkpoint
    started = perf_counter()
    failure_type = failure_code = None
    blockers, restarted = [], False
    try:
        try:
            outcome = PhaseExecutor(action_selector=selector if inference is None else None).run(task, state, context)
        except SimulatedRestart:
            from alphonse.agent_v2.core.intelligence.v3 import TacticalState
            task = TaskState.from_dict(checkpoint.saved)
            state = TacticalState.from_dict(task.hierarchical_state)
            context.question_store = None
            restarted = True
            outcome = PhaseExecutor(action_selector=selector if inference is None else None).run(task, state, context)
        status = outcome.status.value
        allowed_blockers = {'tactical_action_repeated_without_progress', 'tactical_stage_no_new_progress',
                            'tactical_progress_uncertain', 'tactical_action_unavailable'}
        blockers = [prefix for item in outcome.blockers
                    if (prefix := str(item).split(':', 1)[0]) in allowed_blockers]
    except Exception as exc:
        status, failure_type = 'error', type(exc).__name__
        # Only known non-content error codes can enter the result artifact.
        code = str(exc)
        if code in {'openai_codex_invalid_json', 'openai_codex_invalid_tool_json', 'openai_codex_empty_response',
                    'openai_codex_timeout', 'system_one_api_key_required'}:
            failure_code = code
    oracle = fixture.oracle()
    expected_complete = scenario in {'normal', 'uncertain', 'restart'}
    expected = (oracle['verified'] and oracle['submission_count'] == 1 and status == 'phase_complete'
                if expected_complete else oracle['submission_count'] == 0 and status in {'blocked', 'cancelled'})
    return {'scenario': scenario, 'variant': 'fixture_batch' if batch else 'per_action',
            'outcome': status, 'failure_type': failure_type, 'failure_code': failure_code, 'blockers': blockers,
            'independent_oracle_pass': expected,
            'verified': oracle['verified'], 'submission_count': oracle['submission_count'],
            'composition_calls': len(model_calls), 'jev_reviews': len(reviews), 'tool_calls': len(state.actions),
            'restarted': restarted,
            'selected_operations': [item.arguments.get('operation') for item in state.actions],
            'action_statuses': [item.status for item in state.actions],
            'argument_keys': [sorted(item.arguments) for item in state.actions],
            'operations': [event['operation'] for event in fixture.events], 'elapsed_seconds': perf_counter() - started,
            'provider_cost_usd': 0 if inference is None else None,
            'cost_basis': 'scripted providers' if inference is None else 'ChatGPT subscription; Jev charge recorded separately'}


def report(repeats: int = 2) -> dict:
    import hashlib
    from pathlib import Path
    from alphonse.agent_v2.evaluation import planning_form_fixture
    scenarios = ('normal', 'changed', 'denied', 'cancel', 'uncertain', 'cancel_during', 'changed_during', 'denied_during', 'restart')
    rows = []
    for repeat in range(repeats):
        for scenario in scenarios:
            for batch in (False, True):
                try:
                    row = run_case(scenario, batch)
                except Exception as exc:
                    row = {'scenario': scenario, 'variant': 'fixture_batch' if batch else 'per_action',
                           'independent_oracle_pass': False, 'error_type': type(exc).__name__}
                row['repeat'] = repeat + 1
                rows.append(row)
    return {'version': 'multifield-v1', 'mode': 'scripted-decisions-real-tactical-executor',
            'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'fixture_sha256': hashlib.sha256(Path(planning_form_fixture.__file__).read_bytes()).hexdigest(),
            'limitations': ['The batch is a fixture-only prototype, not a production browser capability.',
                'Provider outputs and task actions are scripted; live model behavior and provider costs are not measured.',
                'In-memory fixture checks do not qualify real authorization, browser sessions or external websites.',
                'A fixture readback verifies only the in-memory fixture; it is not a general external-effect verifier.',
                'Wall time measures local test execution only.'],
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
    print(json.dumps({'runs': len(data['results']),
                      'passed': sum(bool(row.get('independent_oracle_pass')) for row in data['results'])}))
    return 0 if all(row.get('independent_oracle_pass') for row in data['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
