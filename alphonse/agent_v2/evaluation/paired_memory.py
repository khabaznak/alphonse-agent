"""Frozen flag-off/on boundary evaluation. Freeze first, then run that manifest.

Live runs use real Jev/System Two inference, a closed registry and fresh synthetic
stores. Unit-test providers never constitute measured rollout evidence.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from uuid import uuid5, NAMESPACE_URL
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / 'tests/fixtures/memory_replay'
PROTOCOL = ROOT / 'docs/v3-hierarchical-capd/13-decision-context-evaluation.md'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_boundaries():
    base_bytes = (FIXTURES / 'v1.json').read_bytes()
    base = json.loads(base_bytes)
    extension = json.loads((FIXTURES / 'paired_v3.json').read_text())
    if extension['synthetic'] is not True or extension['base_sha256'] != hashlib.sha256(base_bytes).hexdigest():
        raise ValueError('frozen_v1_corpus_changed')
    if len(base['cases']) != 22 or set(extension['original']) != {c['id'] for c in base['cases']}:
        raise ValueError('original_boundary_cases_changed')
    cases = []
    for original in base['cases']:
        case = deepcopy(original)
        supplement = extension['original'][case['id']]
        required_keys = {'delivery_oracles', 'necessary_by_purpose', 'state_oracle'}
        if not required_keys <= set(supplement) <= required_keys | {'zero_history'}:
            raise ValueError('original_expectations_cannot_be_overridden')
        for identity in original['expected']['necessary_sources']:
            content = next(s['content'] for s in original['sources'] if s['id'] == identity)
            if not any(o['source_id'] == identity and o['start_char'] == 0 and o['end_char'] == len(content)
                       and 'response_composition' in o['purposes'] for o in supplement['delivery_oracles']):
                raise ValueError('original_delivery_expectation_weakened')
        case.update(supplement)
        cases.append(case)
    cases.extend(deepcopy(extension['extensions']))
    if len({c['id'] for c in cases}) != len(cases):
        raise ValueError('duplicate_case_id')
    for case in cases:
        sources = {s['id']: s for s in case['sources']}
        if len(sources) != len(case['sources']):
            raise ValueError('duplicate_source_id')
        for oracle in case['delivery_oracles']:
            if not 0 <= oracle['start_char'] < oracle['end_char'] <= len(sources[oracle['source_id']]['content']):
                raise ValueError('invalid_source_range')
        if not set(case['expected']['necessary_sources']) <= {o['source_id'] for o in case['delivery_oracles']}:
            raise ValueError('missing_frozen_source_oracle')
    return {**{k: extension[k] for k in ('clock', 'timezone', 'rubric', 'version')}, 'synthetic': True, 'cases': cases}


def source_snapshot():
    # Include untracked new runtime/adapter files, prompts, registries and fixtures;
    # the byte inventory is stronger than HEAD + tracked dirty patch alone.
    paths = sorted(p for base in (ROOT / 'alphonse/agent_v2', FIXTURES) for p in base.rglob('*')
                   if p.is_file() and p.suffix in {'.py', '.json', '.md', '.txt', '.j2', '.jinja', '.jinja2', '.yaml', '.yml', '.toml'} and '__pycache__' not in p.parts)
    paths += [ROOT / 'pyproject.toml', PROTOCOL]
    return {'revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'dirty_patch_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=ROOT)).hexdigest(),
            'files': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}


def live_configuration():
    from .context_pilot import configured_jev
    from .context_pilot_live import CLI, configured_model
    jev = configured_jev()
    settings = jev.settings
    return {'provider': 'openai_codex', 'model': configured_model(),
            'jev_endpoint_sha256': hashlib.sha256(settings.api_url.encode()).hexdigest(),
            'jev': {f.name: getattr(settings, f.name) for f in fields(settings) if f.name in {'model', 'enabled'} or f.name.endswith('_threshold')},
            'transport_sha256': hashlib.sha256(Path(CLI).read_bytes()).hexdigest(),
            'python': platform.python_version(), 'platform': platform.platform(),
            'context_budget': int(os.environ.get('ALPHONSE_V3_CONTEXT_TOKEN_BUDGET', '24000')),
            'accounting': 'conservative_utf8_bytes; separate characters/4 context estimate; provider usage nullable',
            'environment': {k: v for k, v in os.environ.items() if k.startswith('ALPHONSE_') and not any(
                word in k for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')) and k != 'ALPHONSE_V3_CONTEXT_PROJECTIONS'}}


def trial_schedule(corpus, repetitions, smoke_cases=()):
    if smoke_cases:
        return [{'trial_id': f"{identity}:smoke:{arm}", 'case_id': identity, 'repetition': 1,
                 'warmup': False, 'arm': arm} for identity in smoke_cases for arm in ('legacy', 'candidate')]
    schedule = []
    for case in corpus['cases']:
        for repetition in range(repetitions + 1):
            for arm in (('legacy', 'candidate') if repetition % 2 == 0 else ('candidate', 'legacy')):
                schedule.append({'trial_id': f"{case['id']}:{repetition}:{arm}", 'case_id': case['id'],
                                 'repetition': repetition, 'warmup': repetition == 0, 'arm': arm})
    return schedule


def freeze(configuration, *, repetitions=20, smoke_cases=()):
    if repetitions < 1:
        raise ValueError('positive_repetitions_required')
    corpus = load_boundaries()
    if len(set(smoke_cases)) != len(smoke_cases) or not set(smoke_cases) <= {c['id'] for c in corpus['cases']}:
        raise ValueError('invalid_smoke_cases')
    schedule = trial_schedule(corpus, repetitions, smoke_cases)
    source = source_snapshot()
    arms = {name: {'projection_flag': candidate, 'revision': source['revision'],
                   'dirty_patch_sha256': source['dirty_patch_sha256'],
                   'source_files_sha256': digest(source['files']), 'fixture_inputs_sha256': digest(corpus)}
            for name, candidate in (('legacy', False), ('candidate', True))}
    manifest = {'schema_version': 1, 'protocol': 'decision-context-v2', 'adapter_version': 1,
        'source': source, 'arms': arms, 'configuration': configuration, 'corpus': corpus,
        'corpus_sha256': digest(corpus), 'rubric_sha256': digest(corpus['rubric']),
        'repetitions': repetitions, 'schedule': schedule, 'smoke_cases': list(smoke_cases),
        'experiment_kind': 'harness_smoke' if smoke_cases else 'boundary_comparison',
        'human_review_repetitions': [1, 10, 20], 'cache_conditions': 'fresh ephemeral transport/session and fixture state; remote cache unknown',
        'cost_boundary': 'admission through final response and fixture read-back; all model/tool calls including bootstrap, retries and errors',
        'historical_v1': 'Retained separately; no historical result rewritten. DC-4 context-estimate comparison remains unassessed.',
        'isolation': 'explicit temporary stores, closed native registry, allowlisted Bash; no daemon or durable extractor'}
    return {**manifest, 'manifest_sha256': digest(manifest)}


def validate_manifest(manifest, configuration):
    unsigned = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if digest(unsigned) != manifest['manifest_sha256']:
        raise ValueError('manifest_hash_mismatch')
    if manifest['source'] != source_snapshot() or manifest['configuration'] != configuration:
        raise ValueError('frozen_source_or_configuration_changed')
    if manifest['schedule'] != trial_schedule(manifest['corpus'], manifest['repetitions'], manifest.get('smoke_cases', ())):
        raise ValueError('frozen_schedule_changed')
    if manifest['corpus'] != load_boundaries():
        raise ValueError('frozen_corpus_changed')


def append_jsonl(path, row):
    with path.open('a') as stream:
        stream.write(json.dumps(row, ensure_ascii=False, default=str) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def report(manifest, rows, *, invalid=None):
    scheduled = {t['trial_id']: t for t in manifest['schedule']}
    measured = [r for r in rows if r.get('trial_id') in scheduled and not scheduled[r['trial_id']]['warmup']]
    complete = len(rows) == len(scheduled) and {r['trial_id'] for r in rows} == set(scheduled)
    contaminated = invalid or any(r.get('invalid_reason') for r in rows) or len({r['trial_id'] for r in rows}) != len(rows) or any(
        r.get('manifest_sha256') != manifest['manifest_sha256'] or r.get('trial_id') not in scheduled or any(r.get(k) != scheduled.get(r.get('trial_id'), {}).get(k) for k in ('case_id', 'repetition', 'warmup', 'arm')) for r in rows)
    failed = [r['trial_id'] for r in rows if r.get('status') != 'returned' or (r.get('runtime_error') and not r.get('expected_safe_failure')) or r.get('violations')
              or not r.get('delivery', {}).get('delivery_pass', r.get('delivery', {}).get('necessary_delivery_pass', False))
              or not r.get('outcome', {}).get('effect_pass', False)
              or not r.get('outcome', {}).get('response_diagnostic_pass', False)]
    instances = sum(r.get('delivery', {}).get('historical_instances', 0) for r in measured if r['arm'] == 'candidate')
    irrelevant = sum(r.get('delivery', {}).get('irrelevant_instances', 0) for r in measured if r['arm'] == 'candidate')
    hard_violation = any(r.get('violations') or any(
        r.get('outcome', {}).get('checks', {}).get(key) is False
        for key in ('no_duplicate_writes', 'no_unplanned_schedules', 'schedule_audience', 'no_duplicate_schedules', 'schedule_once')) for r in rows)
    candidate_failures = [identity for identity in failed if scheduled.get(identity, {}).get('arm') == 'candidate']
    gates = {'DC-0': 'invalid' if contaminated else 'pass', 'DC-1': 'fail' if hard_violation else 'unassessed',
             'DC-2': 'fail' if hard_violation else 'invalid' if contaminated or not complete else 'fail' if candidate_failures or (instances and irrelevant / instances > .05) else 'unassessed',
             'DC-3': 'unassessed', 'DC-4': 'unassessed', 'DC-5': 'unassessed'}
    if manifest.get('smoke_cases'):
        gates['DC-2'] = 'fail' if hard_violation else 'invalid' if contaminated or not complete else 'unassessed'
    return {'experiment_kind': manifest.get('experiment_kind', 'boundary_comparison'), 'gates': gates, 'invalid_reason': invalid, 'complete': complete, 'expected_arm_runs': len(scheduled),
            'recorded_arm_runs': len(rows), 'measured_arm_runs': len(measured), 'failures_including_warmup': failed, 'candidate_failures_including_warmup': candidate_failures,
            'historical_instances': instances, 'irrelevant_instances': irrelevant,
            'independent_human_review': 'unassessed', 'rollout_accepted': False,
            'usage': {'system_two_calls': sum(len(r.get('model_calls', [])) for r in measured),
                      'jev_calls': sum(len(r.get('jev_calls', [])) for r in measured),
                      'native_tool_calls': sum(len(r.get('tool_calls', [])) for r in measured)},
            'limits': ['DC-2 requires independent full-response/trace review; regex diagnostics cannot pass it.',
                       'DC-1 qualification belongs to #24; repeated comparisons and human review belong to #25.',
                       'Provider usage is retained per call; missing usage is unavailable, never zero.']}


def execute(manifest, directory, provider_factory, jev_factory, *, configuration):
    """One append-only attempt per scheduled trial; no retry-away or resume guessing."""
    from .paired_fixture import run_trial
    directory.mkdir(parents=True, exist_ok=False)
    (directory / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    rows = []
    try:
        validate_manifest(manifest, configuration)
    except ValueError as exc:
        (directory / 'gates.json').write_text(json.dumps(report(manifest, rows, invalid=str(exc)), indent=2) + '\n')
        raise
    corpus = manifest['corpus']
    cases = {c['id']: c for c in corpus['cases']}
    review = []
    for trial in manifest['schedule']:
        append_jsonl(directory / 'events.jsonl', {**trial, 'event': 'started'})
        row = {**trial, 'manifest_sha256': manifest['manifest_sha256'], 'status': 'aborted'}
        try:
            validate_manifest(manifest, configuration)
            with tempfile.TemporaryDirectory(prefix='paired-memory-') as temporary:
                counter = iter(range(100000))
                # Pair-stable historical event IDs; separate physical fixture roots.
                def identity():
                    return uuid5(NAMESPACE_URL, trial['case_id'] + ':' + str(next(counter)))
                provider, jev = provider_factory(), jev_factory()
                if configuration.get('provider') == 'openai_codex':
                    settings = jev.settings
                    actual_jev = {k: getattr(settings, k) for k in configuration['jev']}
                    if provider.model != configuration['model'] or actual_jev != configuration['jev'] or hashlib.sha256(settings.api_url.encode()).hexdigest() != configuration['jev_endpoint_sha256']:
                        raise ValueError('frozen_provider_changed')
                with patch('alphonse.agent_v2.conversations.uuid4', identity):
                    row.update(run_trial(Path(temporary), cases[trial['case_id']], corpus,
                                         provider, jev, candidate=trial['arm'] == 'candidate'))
            row['status'] = 'returned'
        except BaseException as exc:
            row.update(getattr(exc, 'evaluation_trace', {}))
            row.update(status='aborted', error=type(exc).__name__)
            if isinstance(exc, ValueError) and str(exc).startswith(('frozen_', 'manifest_')):
                row['invalid_reason'] = str(exc)
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            rows.append(row)
            append_jsonl(directory / 'trials.jsonl', row)
            gates = report(manifest, rows)
            (directory / 'gates.json').write_text(json.dumps(gates, indent=2) + '\n')
            flagged = row['trial_id'] in gates['failures_including_warmup']
            if (not trial['warmup'] and trial['repetition'] in manifest['human_review_repetitions']) or flagged:
                review.append({**trial, 'flagged': flagged, 'reviewer_identity': None, 'reviewer_role': None,
                               'decision': 'unassessed', 'evidence': 'trials.jsonl#' + row['trial_id'], 'disagreements': []})
                (directory / 'human-review.json').write_text(json.dumps(review, indent=2) + '\n')
        if row.get('invalid_reason'):
            break


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('freeze')
    prepare.add_argument('--output', type=Path, required=True)
    prepare.add_argument('--repetitions', type=int, default=20)
    prepare.add_argument('--smoke-case', action='append', default=[], help='Predeclared diagnostic subset; full corpus and oracles remain frozen.')
    run = commands.add_parser('run')
    run.add_argument('--manifest', type=Path, required=True)
    run.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    configuration = live_configuration()
    if args.command == 'freeze':
        manifest = freeze(configuration, repetitions=args.repetitions, smoke_cases=args.smoke_case)
        with args.output.open('x') as stream:
            stream.write(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    else:
        from .context_pilot import configured_jev
        from .household_live import HouseholdLiveProvider
        execute(json.loads(args.manifest.read_text()), args.output, HouseholdLiveProvider, configured_jev, configuration=configuration)


if __name__ == '__main__':
    main()
