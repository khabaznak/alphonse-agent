"""Frozen, scripted comparison through the real V3 processor and disposable fixtures.

This is a control-flow comparison, not a live-model benchmark. The historical
progress rule is an emulation, and no simpler loop is implemented here.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter
from unittest.mock import patch

from . import resourceful_delegation as fixtures
from . import planning_independent_reads
from alphonse.agent_v2.core.intelligence.v3 import executor

CORPUS = Path(__file__).resolve().parents[3] / 'tests/fixtures/planning_comparison/v2.json'
VARIANTS = ('historical_progress_emulation', 'uncertainty_aware', 'targeted_recovery', 'combined_fixture_action', 'independent_read_sequence')


def _historical_no_progress(state, subgoal_id, threshold=.2):
    """Reproduce the old below-yes progress gate, with incomplete context guarded."""
    import math
    count = 0
    for entry in reversed(state.evidence.entries):
        if entry.get('phase_id') != state.phase.phase_id or entry.get('subgoal_id') != subgoal_id:
            continue
        review = entry.get('progress_review') or {}
        coverage = review.get('context_coverage') or {}
        if (review.get('context_complete') is False or coverage.get('omitted_prior_count', 0)
                or coverage.get('omitted_prior_result_count', 0) or coverage.get('omitted_prior_refs')):
            break
        value = (review.get('answers') or {}).get('runtime_action_added_progress')
        if type(value) not in (float, int) or not math.isfinite(value) or not 0 <= value < .8:
            break
        count += 1
    return count


def _run_case(case, variant):
    with tempfile.TemporaryDirectory(prefix='planning-comparison-') as directory:
        with fixtures.fixture_server(case['kind']) as (url, receipts):
            trial = fixtures.Trial(Path(directory), case, url, receipts)
            # Fixture-specific prototype: the already-existing submit command
            # independently reads the form before submitting and verifying it.
            # Remove redundant standalone inspection only for these cases.
            # This does not implement a generic one-loop architecture.
            if variant == 'combined_fixture_action' and case['kind'] in {'form', 'changed'}:
                original_plan = trial.inference.plan
                def plan():
                    value = original_plan()
                    steps = value['subgoals']
                    if len(steps) >= 2 and steps[0]['subgoal_id'] == 'inspect' and steps[1]['subgoal_id'] == 'submit':
                        value['subgoals'] = steps[1:]
                        value['subgoals'][0]['depends_on'] = []
                    return value
                trial.inference.plan = plan
            judgments = []
            original_review = trial.context.system_one.evaluate_tactical_progress
            def review(**kwargs):
                judgments.append(1)
                return original_review(**kwargs)
            trial.context.system_one.evaluate_tactical_progress = review
            with ExitStack() as stack:
                if variant not in {'targeted_recovery', 'combined_fixture_action', 'independent_read_sequence'}:
                    stack.enter_context(patch.object(executor, '_queue_uncertainty_recovery', lambda *_: None))
                if variant == 'historical_progress_emulation':
                    stack.enter_context(patch.object(executor, '_consecutive_no_progress', _historical_no_progress))
                started = perf_counter()
                try:
                    result = trial.run()
                except Exception as exc:
                    return {'case_id': case['id'], 'variant': variant, 'outcome': 'error',
                            'error_type': type(exc).__name__, 'elapsed_seconds': perf_counter() - started}
            entries = fixtures.entries(trial.task)
            purposes = {}
            for request in trial.inference.requests:
                purpose = request['purpose']
                purposes[purpose] = purposes.get(purpose, 0) + 1
            operations = [e for e in entries if e.get('tool_id')]
            get_count = sum(r['method'] == 'GET' for r in receipts)
            posts = sum(r['method'] == 'POST' for r in receipts)
            return {
                'case_id': case['id'], 'split': case['split'], 'variant': variant,
                'outcome': trial.task.status, 'expected_outcome': case['expected'],
                'independent_oracle_pass': result['scenario_pass'], 'checks': result['checks'],
                'verified_completion': result['metrics']['verified_task_completion'],
                'unnecessary_questions': result['metrics']['unnecessary_questions'],
                'inference_calls': len(trial.inference.requests), 'inference_by_purpose': purposes,
                'tactical_jev_calls': len(judgments), 'tool_calls': len(operations),
                'observations': get_count, 'submission_attempts': posts,
                'rejected_actions': sum(e.get('status') == 'rejected' for e in entries),
                'elapsed_seconds': perf_counter() - started,
                'provider_cost_usd': 0, 'cost_basis': 'scripted doubles; no provider calls',
            }


def _run_tactical_case(case, variant):
    """Run the real tactical executor over independent, useful synthetic fragments."""
    from types import SimpleNamespace
    from alphonse.agent_v2.core.core import CoreLoopContext, ToolDescriptor, ToolKind
    from alphonse.agent_v2.core.inference import InferenceResult
    from alphonse.agent_v2.core.intelligence.task_state import TaskState
    from alphonse.agent_v2.core.intelligence.v3 import PhaseExecutor, PhasePlan, PhaseSubgoal, new_tactical_state
    from alphonse.agent_v2.core.messages import InMemoryMessageQueue
    from alphonse.agent_v2.core.tools.registry import InMemoryToolRegistry, ToolDefinition
    from alphonse.agent_v2.system_one import SystemOneTacticalReview

    fragments = case['fragments']
    observed, prompts, reviews = [], [], []
    registry = InMemoryToolRegistry()
    def read(args):
        index = args['index']
        observed.append(fragments[index])
        return {'fragment': fragments[index]}
    registry.register(ToolDefinition(ToolDescriptor('native.fixture_read', 'fixture_read', ToolKind.NATIVE,
        read_only=True, capabilities=('native.fixture_read',)), read))
    subgoal = PhaseSubgoal('read', 'Read all fixture fragments', 'fragments', allowed_capabilities=('native.fixture_read',))
    state = new_tactical_state(PhasePlan('fixture', 'Read fixture', (subgoal,), authorized_capabilities=('native.fixture_read',)))
    state.revealed_tool_ids = ['native.fixture_read']
    class Provider:
        def generate_json(self, request):
            prompts.append(request.prompt)
            return InferenceResult(json_value={'tool_id': 'native.fixture_read', 'arguments': {'index': len(observed)},
                'acceptance_questions': [{'question_id': 'complete', 'type': 'noul',
                    'instructions': 'Have all fragments been read?',
                    'criteria': {'true': 'All fragments are present.', 'false': 'At least one fragment is missing.'}}]})
    def review(**kwargs):
        reviews.append(1)
        done = len(observed) == len(fragments)
        return SystemOneTacticalReview(done, .9 if done else .6, done,
            answers={'complete': .99 if done else .1, 'runtime_declared_subgoal_complete': .99 if done else .1,
                     'runtime_action_added_progress': .99 if done else .6})
    context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=registry, inference=Provider(),
                              system_one=SimpleNamespace(evaluate_tactical_progress=review))
    task = TaskState(task_id=case['id'], user='fixture', goal='Read every synthetic fragment')
    with ExitStack() as stack:
        if variant not in {'targeted_recovery', 'combined_fixture_action', 'independent_read_sequence'}:
            stack.enter_context(patch.object(executor, '_queue_uncertainty_recovery', lambda *_: None))
        if variant == 'historical_progress_emulation':
            stack.enter_context(patch.object(executor, '_consecutive_no_progress', _historical_no_progress))
        started = perf_counter()
        outcome = PhaseExecutor().run(task, state, context)
    oracle = observed == fragments
    completed = outcome.status.value == 'phase_complete'
    return {'case_id': case['id'], 'split': case['split'], 'variant': variant,
            'outcome': outcome.status.value, 'expected_outcome': 'phase_complete',
            'independent_oracle_pass': oracle and completed, 'verified_completion': oracle and completed,
            'false_stall': not completed and observed == fragments[:len(observed)] and len(observed) < len(fragments),
            'inference_calls': len(prompts), 'tactical_jev_calls': len(reviews), 'tool_calls': len(observed),
            'unnecessary_questions': 0, 'elapsed_seconds': perf_counter() - started,
            'targeted_recovery_prompted': any('Targeted uncertainty recovery' in p for p in prompts),
            'provider_cost_usd': 0, 'cost_basis': 'scripted doubles; no provider calls'}


def _run_independent_case(case, variant):
    from . import planning_independent_reads
    with ExitStack() as stack:
        if variant not in {'targeted_recovery', 'combined_fixture_action', 'independent_read_sequence'}:
            stack.enter_context(patch.object(executor, '_queue_uncertainty_recovery', lambda *_: None))
        if variant == 'historical_progress_emulation':
            stack.enter_context(patch.object(executor, '_consecutive_no_progress', _historical_no_progress))
        result = planning_independent_reads.run_case(case['scenario'], variant == 'independent_read_sequence')
    return {**result, 'case_id': case['id'], 'split': case['split'], 'variant': variant,
            'independent_oracle_pass': result['oracle_pass'], 'inference_calls': result['composition_calls'],
            'tactical_jev_calls': result['jev_reviews'], 'unnecessary_questions': 0}


def run_comparison(corpus_path=CORPUS):
    raw = Path(corpus_path).read_bytes()
    corpus = json.loads(raw)
    source = json.loads(fixtures.CORPUS.read_text())
    if source['version'] != corpus['source']:
        raise ValueError('source_corpus_version_mismatch')
    by_id = {case['id']: case for case in source['cases']}
    if len(set(corpus['case_ids'])) != len(corpus['case_ids']) or any(i not in by_id for i in corpus['case_ids']):
        raise ValueError('unknown_or_duplicate_case')
    cases = [by_id[i] for i in corpus['case_ids']]
    rows = []
    for repeat in range(corpus['repeats']):
        for case in cases:
            for variant in VARIANTS:
                row = _run_case(case, variant)
                row['repeat'] = repeat + 1
                rows.append(row)
        for case in corpus.get('tactical_cases', []):
            for variant in VARIANTS:
                try:
                    row = _run_tactical_case(case, variant)
                except Exception as exc:
                    row = {'case_id': case['id'], 'split': case['split'], 'variant': variant,
                           'outcome': 'error', 'error_type': type(exc).__name__, 'independent_oracle_pass': False}
                row['repeat'] = repeat + 1
                rows.append(row)
        for case in corpus.get('independent_cases', []):
            for variant in VARIANTS:
                try:
                    row = _run_independent_case(case, variant)
                except Exception as exc:
                    row = {'case_id': case['id'], 'split': case['split'], 'variant': variant,
                           'outcome': 'error', 'error_type': type(exc).__name__, 'independent_oracle_pass': False}
                row['repeat'] = repeat + 1
                rows.append(row)
    return {
        'version': corpus['version'], 'mode': 'scripted-models-real-processor',
        'comparison_complete': False,
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'executor_sha256': hashlib.sha256(Path(executor.__file__).read_bytes()).hexdigest(),
        'corpus_sha256': hashlib.sha256(raw).hexdigest(),
        'source_corpus_sha256': hashlib.sha256(fixtures.CORPUS.read_bytes()).hexdigest(),
        'source_harness_sha256': hashlib.sha256(Path(fixtures.__file__).read_bytes()).hexdigest(),
        'independent_harness_sha256': hashlib.sha256(Path(planning_independent_reads.__file__).read_bytes()).hexdigest(),
        'variants': list(VARIANTS),
        'unavailable_variants': {'simpler_loop': 'No independently implemented and verified prototype yet.'},
        'limitations': [
            'The historical progress rule is an emulation, not a checkout of the historical revision.',
            'Scripted providers return fixed answers and do not test model response to prompts.',
            'The ambiguous tactical case scripts both the model judgments and its subsequent actions; it cannot establish live recovery effectiveness.',
            'The combined fixture action removes inspection only for two scripted cases; it is not a generic simpler loop.',
            'The independent-read sequence is implemented only for bounded read-only actions, not a general simpler phase loop.',
            'One-field form fixtures and nonce receipts do not qualify browser workflows or live sites.',
            'Wall time is local elapsed time, not provider latency; provider cost is not measured.',
            'Observation counts are recorded, but repeat necessity and false stalls are not classified by this oracle.',
        ], 'results': rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run_comparison()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'runs': len(report['results']),
                      'passes': sum(bool(r.get('independent_oracle_pass')) for r in report['results']),
                      'comparison_complete': report['comparison_complete'],
                      'unavailable_variants': report['unavailable_variants']}))
    return 0 if report['comparison_complete'] and all(r.get('independent_oracle_pass') for r in report['results']) else 1


if __name__ == '__main__':
    raise SystemExit(main())
