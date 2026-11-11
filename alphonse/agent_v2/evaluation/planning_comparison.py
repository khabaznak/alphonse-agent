"""Frozen, scripted comparison through the real V3 processor and disposable fixtures.

This is a control-flow comparison, not a live-model benchmark. The historical
progress rule is an emulation; the flat-phase prototype is not a general runtime.
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

CORPUS = Path(__file__).resolve().parents[3] / 'tests/fixtures/planning_comparison/v7.json'
VARIANTS = ('historical_progress_emulation', 'uncertainty_aware', 'targeted_recovery', 'combined_fixture_action', 'independent_read_sequence', 'flat_phase_prototype')


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


def _run_flat_trial(trial):
    """One planned phase, real dispatch and outer verification; no replan loop.

    This evaluation-only prototype deliberately reports unresolved questions,
    recovery and replanning as non-completion. It is not a replacement runtime.
    """
    from alphonse.agent_v2.core.inference import InferencePurpose, InferenceRequest
    from alphonse.agent_v2.core.intelligence.v3 import (PhasePlan, PhaseExecutor, ToolRevealPolicy,
        V3OuterController, HierarchicalCAPDProcessor, new_tactical_state)
    from alphonse.agent_v2.core.intelligence.v3.contracts import TacticalState
    from alphonse.agent_v2.core.intelligence.v3.revealing import validate_phase_actions
    trial.task.status = 'running'
    trial.task.set_acceptance_contract_from_markdown('- [ ] Fulfill the original delegated outcome with verified effects')
    request = InferenceRequest(prompt='Prepare the synthetic task phase', purpose=InferencePurpose.PHASE_PLANNING,
                               task_id=trial.task.task_id)
    phase = PhasePlan.from_dict(trial.inference.generate_json(request).json_value)
    validate_phase_actions(phase, tuple(trial.context.tools.list()))
    state = new_tactical_state(phase)
    try:
        outcome = PhaseExecutor(reveal_policy=ToolRevealPolicy()).run(trial.task, state, trial.context)
    except fixtures.SimulatedRestart:
        trial.restarted = True
        trial.questions = fixtures.RestartStore(trial.root / 'questions.sqlite')
        trial.context.question_store = trial.questions
        trial.task = trial.questions.load_task_checkpoint(trial.task.task_id)
        state = TacticalState.from_dict(trial.task.hierarchical_state)
        outcome = PhaseExecutor(reveal_policy=ToolRevealPolicy()).run(trial.task, state, trial.context)
    HierarchicalCAPDProcessor._append_history(trial.task, state, outcome.to_dict())
    V3OuterController().review_and_route(trial.task, state, outcome, trial.context)
    if trial.task.status == 'running':
        # This prototype does not autonomously replan or invent a user response.
        trial.task.status = 'failed'
        trial.task.outcome = {'status': 'failure', 'reason': 'flat_prototype_requires_replanning'}
    return trial.score()


def _use_evidence_only_review(trial):
    """Script Check from recorded readback evidence, never the fixture oracle."""
    from alphonse.agent_v2.system_one import SystemOneReviewResult
    def evaluate(**values):
        entries = values['evidence'].get('entries', [])
        refs = [entry['evidence_ref'] for entry in entries
                if entry.get('status') == 'success' and entry.get('evidence_ref')
                and isinstance(entry.get('result'), dict)
                and (entry['result'].get('verification') or {}).get('status') == 'verified']
        updates = tuple({'criterion_id': criterion['id'], 'status': 'satisfied', 'evidence_refs': refs}
                        for criterion in values['contract']['criteria']) if refs else ()
        return SystemOneReviewResult(updates, ())
    trial.context.system_one.evaluate = evaluate


def _run_case(case, variant):
    with tempfile.TemporaryDirectory(prefix='planning-comparison-') as directory:
        with fixtures.fixture_server(case['kind']) as (url, receipts):
            trial = fixtures.Trial(Path(directory), case, url, receipts)
            _use_evidence_only_review(trial)
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
                    result = _run_flat_trial(trial) if variant == 'flat_phase_prototype' else trial.run()
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
                'observations': get_count, 'repeated_observations': max(0, get_count - 1),
                'false_stall': case['expected'] == 'completed' and trial.task.status == 'failed' and not trial.oracle_effect(),
                'submission_attempts': posts,
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
            completed = sum(entry.get('status') == 'success' for entry in state.evidence.entries)
            return InferenceResult(json_value={'tool_id': 'native.fixture_read', 'arguments': {'index': completed},
                'acceptance_questions': [{'question_id': 'complete', 'type': 'noul',
                    'instructions': 'Have all fragments been read?',
                    'criteria': {'true': 'All fragments are present.', 'false': 'At least one fragment is missing.'}}]})
    def review(**kwargs):
        reviews.append(1)
        successful = [entry for entry in kwargs.get('execution_log', []) if entry.get('status') == 'success']
        done = len(successful) == len(fragments)
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
            'unnecessary_questions': 0, 'repeated_observations': len(observed) - len(set(observed)),
            'elapsed_seconds': perf_counter() - started,
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
            'false_stall': case['scenario'] in {'normal', 'uncertain'} and result['outcome'] in {'blocked', 'failed'},
            'repeated_observations': len(result['observed_keys']) - len(set(result['observed_keys'])),
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
    complete = len(rows) == corpus['repeats'] * len(VARIANTS) * (
        len(cases) + len(corpus.get('tactical_cases', [])) + len(corpus.get('independent_cases', []))) and all(
        all(key in row for key in ('outcome', 'independent_oracle_pass', 'false_stall',
            'repeated_observations', 'unnecessary_questions', 'inference_calls', 'tool_calls',
            'provider_cost_usd', 'elapsed_seconds')) for row in rows)
    flat_failures = [row for row in rows if row.get('variant') == 'flat_phase_prototype'
                     and not row.get('independent_oracle_pass')]
    decision = ('Comparison incomplete; no architecture decision.' if not complete else
                'Retain the hierarchical loop; the flat prototype fails required recovery cases. Revisit with stronger evidence.'
                if flat_failures else
                'Retain the hierarchical loop pending general replanning, question-resumption and live qualification; this sample found no flat failures.')
    return {
        'version': corpus['version'], 'mode': 'scripted-models-real-processor',
        'comparison_complete': complete,
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'executor_sha256': hashlib.sha256(Path(executor.__file__).read_bytes()).hexdigest(),
        'corpus_sha256': hashlib.sha256(raw).hexdigest(),
        'source_corpus_sha256': hashlib.sha256(fixtures.CORPUS.read_bytes()).hexdigest(),
        'source_harness_sha256': hashlib.sha256(Path(fixtures.__file__).read_bytes()).hexdigest(),
        'independent_harness_sha256': hashlib.sha256(Path(planning_independent_reads.__file__).read_bytes()).hexdigest(),
        'variants': list(VARIANTS),
        'prototype_limitations': {'flat_phase_prototype': 'The prototype omits general replanning and authenticated question resumption.'},
        'architecture_decision': decision,
        'limitations': [
            'The historical progress rule is an emulation, not a checkout of the historical revision.',
            'Scripted providers return fixed answers and do not test model response to prompts.',
            'Scripted Check uses recorded verified readback evidence; the fixture oracle is reserved for outcome scoring.',
            'The ambiguous tactical case scripts both the model judgments and its subsequent actions; it cannot establish live recovery effectiveness.',
            'The combined fixture action removes inspection only for two scripted cases; it is not a generic simpler loop.',
            'The independent-read sequence is implemented only for bounded read-only actions, not a general simpler phase loop.',
            'The flat phase prototype uses real dispatch and outer verification but has no general replanning or question-resumption loop; tactical-only cases do not compare hierarchy.',
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
                      'prototype_limitations': report['prototype_limitations']}))
    # A completed comparison may include deliberate, independently recorded failures.
    return 0 if report['comparison_complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
