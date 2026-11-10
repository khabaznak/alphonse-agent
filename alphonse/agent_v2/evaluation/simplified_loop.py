"""Offline paired evaluation contract for V3 and a candidate agent loop.

Adapters MUST be synthetic and effect-isolated. The harness gives each run a fresh
fixture directory, but is not an OS/network sandbox and cannot enforce that promise.
No real provider or household integration is invoked by this module.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any, Callable, Protocol


@dataclass
class Run:
    status: str
    response: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    reason: str | None = None
    measured_kinds: tuple[str, ...] = ()


class AdapterUnavailable(Exception):
    """The adapter cannot evaluate this case yet."""


class Adapter(Protocol):
    """An offline adapter; fixture effects must be confined to fixture_root.

    The callback records events as they happen, including before an exception.
    Events use kind=system_two|jev|tool|question; usage fields are optional.
    """
    revision: str
    offline: bool

    def __call__(self, case: dict[str, Any], fixture_root: Path,
                 emit: Callable[[dict[str, Any]], None]) -> Run: ...


def load_frozen(path: str | Path) -> tuple[dict[str, Any], str]:
    path = Path(path)
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != path.with_suffix('.sha256').read_text().split()[0]:
        raise ValueError('frozen_corpus_digest_mismatch')
    corpus = json.loads(raw)
    if corpus.get('schema_version') != 1 or not isinstance(corpus.get('cases'), list):
        raise ValueError('invalid_corpus')
    ids = [case['id'] for case in corpus['cases']]
    if len(ids) != len(set(ids)) or not ids:
        raise ValueError('invalid_case_ids')
    return corpus, digest


def _path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if path == root or root not in path.parents:
        raise ValueError('fixture_path_outside_root')
    return path


def _create_fixture(root: Path, case: dict[str, Any]) -> None:
    for name, value in case.get('fixture', {}).get('files', {}).items():
        path = _path(root, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value, encoding='utf-8')


def _snapshot(root: Path) -> dict[str, str]:
    result = {}
    for path in root.rglob('*'):
        if path.is_symlink():
            result[path.relative_to(root).as_posix()] = '<symlink>'
        elif path.is_file():
            raw = path.read_bytes()
            try:
                result[path.relative_to(root).as_posix()] = raw.decode('utf-8')
            except UnicodeDecodeError:
                result[path.relative_to(root).as_posix()] = 'sha256:' + hashlib.sha256(raw).hexdigest()
    return result


def _usage(events: list[dict[str, Any]], key: str, *, estimate: bool, complete: bool) -> int | None:
    if not complete:
        return None
    calls = [e for e in events if e.get('kind') in {'system_two', 'jev'}]
    if not calls:
        return 0
    field = 'estimated_usage' if estimate else 'provider_usage'
    values = [(e.get(field) or {}).get(key) for e in calls]
    # A partial sum is misleading: mark missing/invalid usage as unknown.
    if any(type(v) is not int or v < 0 for v in values):
        return None
    return sum(values)


def _metrics(events: list[dict[str, Any]], elapsed_ms: float, measured: tuple[str, ...]) -> dict[str, Any]:
    counts = Counter(e.get('kind') for e in events)
    fingerprints = [json.dumps([e.get('tool'), e.get('arguments')], sort_keys=True, ensure_ascii=False)
                    for e in events if e.get('kind') == 'tool']
    return {'system_two_calls': counts['system_two'] if 'system_two' in measured else None,
            'jev_calls': counts['jev'] if 'jev' in measured else None,
            'tool_calls': counts['tool'] if 'tool' in measured else None,
            'questions': counts['question'] if 'question' in measured else None,
            'repeated_actions': len(fingerprints) - len(set(fingerprints)) if 'tool' in measured else None,
            'provider_input_tokens': _usage(events, 'input_tokens', estimate=False, complete={'system_two', 'jev'} <= set(measured)),
            'provider_output_tokens': _usage(events, 'output_tokens', estimate=False, complete={'system_two', 'jev'} <= set(measured)),
            'estimated_input_tokens': _usage(events, 'input_tokens', estimate=True, complete={'system_two', 'jev'} <= set(measured)),
            'estimated_output_tokens': _usage(events, 'output_tokens', estimate=True, complete={'system_two', 'jev'} <= set(measured)),
            'elapsed_ms': round(elapsed_ms, 3)}


def _oracle(case: dict[str, Any], result: Run, events: list[dict[str, Any]],
            before: dict[str, str], after: dict[str, str]) -> dict[str, Any]:
    """Score using frozen expectations and observed fixture state, not agent claims."""
    expected = {**case['oracle'], **case['oracle'].get('by_status', {}).get(result.status, {})}
    actual_effects = {p: after.get(p) for p in set(before) | set(after) if before.get(p) != after.get(p)}
    expected_effects = expected.get('effects', {})
    status_ok = result.status in expected['allowed_statuses']
    response_ok = all(x.casefold() in result.response.casefold() for x in expected.get('response_contains', []))
    forbidden_response = any(x.casefold() in result.response.casefold() for x in expected.get('response_forbidden', []))
    forbidden_tools = {e.get('tool') for e in events if e.get('kind') == 'tool'} & set(expected.get('forbidden_tools', []))
    tools_ok = (not forbidden_tools) if 'tool' in result.measured_kinds else None
    forbidden_accesses = {e.get('source') for e in events if e.get('kind') == 'source_read'} & set(expected.get('forbidden_accesses', []))
    accesses_ok = (not forbidden_accesses) if 'source_read' in result.measured_kinds else None
    read_counts = Counter(e.get('source') for e in events if e.get('kind') == 'source_read')
    expected_reads = expected.get('source_read_counts', {})
    read_counts_ok = (all(read_counts[source] == count for source, count in expected_reads.items())
                      if 'source_read' in result.measured_kinds else None)
    expected_restarts = expected.get('restart_count')
    restarts_ok = (sum(e.get('kind') == 'restart' for e in events) == expected_restarts
                   if expected_restarts is not None and 'restart' in result.measured_kinds else None)
    reason_ok = all(x.casefold() in (result.reason or '').casefold() for x in expected.get('reason_contains', []))
    effects_ok = actual_effects == expected_effects
    max_questions = expected.get('max_questions')
    questions = sum(e.get('kind') == 'question' for e in events)
    questions_ok = (True if max_questions is None else
                    questions <= max_questions if 'question' in result.measured_kinds else None)
    checks = {'status': status_ok, 'response': response_ok and not forbidden_response,
              'effects': effects_ok, 'forbidden_tools': tools_ok if expected.get('forbidden_tools') else True,
              'forbidden_accesses': accesses_ok if expected.get('forbidden_accesses') else True,
              'source_read_counts': read_counts_ok if expected_reads else True,
              'restarts': restarts_ok if expected_restarts is not None else True,
              'reason': reason_ok, 'questions': questions_ok, 'adapter_error': result.error is None}
    # Claiming completed when the independent outcome fails is a false success.
    false_success = result.status == 'completed' and any(v is False for v in checks.values())
    return {'passed': all(checks.values()), 'checks': checks, 'false_success': false_success,
            'unnecessary_questions': max(0, questions - max_questions) if max_questions is not None and 'question' in result.measured_kinds else None,
            'observed_effects': actual_effects, 'forbidden_tools': sorted(forbidden_tools),
            'forbidden_accesses': sorted(forbidden_accesses), 'source_read_counts': dict(read_counts),
            'restart_count': sum(e.get('kind') == 'restart' for e in events) if 'restart' in result.measured_kinds else None}


def _run(case: dict[str, Any], adapter: Adapter | None) -> dict[str, Any]:
    if adapter is None:
        return {'availability': 'unavailable', 'passed': None, 'reason': 'adapter_not_implemented',
                'metrics': None, 'oracle': None, 'events': []}
    if getattr(adapter, 'offline', False) is not True:
        raise ValueError('adapter_must_explicitly_declare_offline')
    started, events = monotonic(), []
    with TemporaryDirectory(prefix='alphonse-loop-eval-') as directory:
        root = Path(directory).resolve()
        _create_fixture(root, case)
        before = _snapshot(root)
        try:
            # Never disclose the independent expected results to an engine.
            inputs = {key: deepcopy(value) for key, value in case.items() if key != 'oracle'}
            run = adapter(inputs, root, lambda event: events.append(deepcopy(event)))
            if not isinstance(run, Run):
                raise TypeError('adapter_must_return_Run')
            events.extend(deepcopy(run.events))
        except AdapterUnavailable as exc:
            # An adapter may only decline before doing work. Otherwise preserve
            # the attempted trajectory and independent fixture effects as a
            # failed run rather than hiding it behind an unavailable status.
            if not events and _snapshot(root) == before:
                return {'availability': 'unavailable', 'passed': None, 'reason': str(exc),
                        'metrics': None, 'oracle': None, 'events': events}
            run = Run(status='error', error='AdapterUnavailable after activity: ' + str(exc))
        except Exception as exc:
            run = Run(status='error', error=f'{type(exc).__name__}: {exc}')
        after = _snapshot(root)
        evaluation = _oracle(case, run, events, before, after)
    return {'availability': 'available', 'revision': str(getattr(adapter, 'revision', 'unknown')),
            'status': run.status, 'response': run.response, 'error': run.error, 'reason': run.reason,
            'passed': evaluation['passed'], 'oracle': evaluation,
            'metrics': _metrics(events, (monotonic() - started) * 1000, run.measured_kinds), 'events': events}


def compare(corpus_path: str | Path, *, baseline: Adapter | None, candidate: Adapter | None) -> dict[str, Any]:
    """Run paired cases in separate disposable roots; no missing run counts as a pass."""
    corpus, digest = load_frozen(corpus_path)
    rows = []
    for case in corpus['cases']:
        rows.append({'case_id': case['id'], 'baseline': _run(case, baseline),
                     'candidate': _run(case, candidate)})
    return {'schema_version': 1, 'simulation': True,
            'limitations': ['Offline adapter declaration is not a network sandbox.',
                            'Scripted providers do not measure autonomous model behavior.'],
            'corpus_sha256': digest, 'split': corpus['split'], 'cases': rows}


class ExistingV3ReplayAdapter:
    """Real V3 processor with the existing scripted offline replay adapter.

    Only explicitly mapped scenarios are supported. Its provider and tools are
    deterministic stand-ins, and its Jev calls are not fully instrumented.
    This is a runtime regression, not a live-model or Gmail integration test.
    """
    offline = True
    revision = 'working-tree:hierarchical-v3:deterministic-adapters'
    _supported = {'solar-project-completion'}

    def __call__(self, case: dict[str, Any], fixture_root: Path,
                 emit: Callable[[dict[str, Any]], None]) -> Run:
        from .deterministic_adapters import _DeterministicProvider, _ReplayJev, _RunState, _registry, _scenario
        from alphonse.agent_v2.core.core import CoreLoopContext
        from alphonse.agent_v2.core.inference import InferencePurpose, InferenceResult, InferenceRouter, ModelProfile
        from alphonse.agent_v2.core.intelligence.task_state import TaskState
        from alphonse.agent_v2.core.intelligence.v3 import HierarchicalCAPDProcessor
        from alphonse.agent_v2.core.messages import InMemoryMessageQueue
        scenario_id = case.get('fixture', {}).get('v3_replay_case')
        if scenario_id not in self._supported:
            raise AdapterUnavailable('no_frozen_v3_adapter_for_case')
        project = fixture_root / 'project'
        project.mkdir()
        for path in list(fixture_root.iterdir()):
            if path != project:
                path.rename(project / path.name)
        scenario = _scenario(scenario_id)
        state = _RunState(scenario=scenario, root=project, engine='hierarchical_v3')

        class Provider(_DeterministicProvider):
            def generate_json(self, request):
                if request.purpose == InferencePurpose.TASK_UNDERSTANDING:
                    return InferenceResult(json_value={
                        'requested_outcome': case['goal'], 'supplied_parameters': [],
                        'constraints': [], 'unresolved_dependencies': [],
                        'superseded_constraints': [],
                    })
                return super().generate_json(request)

        class Judge(_ReplayJev):
            # Track the mock calls, with no claim of real provider usage.
            def evaluate(self, **kwargs):
                emit({'kind': 'jev', 'purpose': 'check'})
                return super().evaluate(**kwargs)
            def curate_request_tools(self, **kwargs):
                emit({'kind': 'jev', 'purpose': 'tool_relevance'})
                return super().curate_request_tools(**kwargs)
            def select_plan_tools(self, **kwargs):
                emit({'kind': 'jev', 'purpose': 'legacy_tool_relevance'})
                return super().select_plan_tools(**kwargs)
            def triage_plan_messages(self, **kwargs):
                emit({'kind': 'jev', 'purpose': 'message_triage'})
                return super().triage_plan_messages(**kwargs)
            def judge_act(self, **kwargs):
                emit({'kind': 'jev', 'purpose': 'act'})
                return super().judge_act(**kwargs)
            def evaluate_tactical_progress(self, **kwargs):
                emit({'kind': 'jev', 'purpose': 'tactical_progress'})
                return super().evaluate_tactical_progress(**kwargs)

        def telemetry(event):
            if event.get('event_type') == 'inference':
                # The stand-in computes estimates, not real provider usage.
                emit({'kind': 'system_two', 'estimated_usage': {
                    'input_tokens': event.get('input_tokens_estimate'),
                    'output_tokens': event.get('output_tokens_estimate')}})
        router = InferenceRouter(provider=Provider(state),
            default_profile=ModelProfile(provider='deterministic', model='replay-v1', profile_id='offline-replay'),
            telemetry_sink=telemetry)
        context = CoreLoopContext(messages=InMemoryMessageQueue(), tools=_registry(state), inference=router,
            system_one=Judge({a.tool_id for a in scenario.actions if not a.read_only}), telemetry_sink=telemetry)
        task = TaskState(task_id='fixture-v3', message_id='fixture-message', user='evaluation-user',
            project_id='fixture-project', goal=case['goal'], intelligence_engine='hierarchical_v3',
            intelligence_schema_version=3)
        outcome = HierarchicalCAPDProcessor().process(task, context)
        final = TaskState.from_dict(outcome.snapshot.metadata.get('task_state') or task.to_dict())
        for path in list(project.iterdir()):
            path.rename(fixture_root / path.name)
        project.rmdir()
        return Run(status=final.status, response=str((final.outcome or {}).get('reason') or final.status),
                   error=str((final.outcome or {}).get('reason') or 'task failed') if final.status in {'failed', 'error'} else None,
                   measured_kinds=('system_two', 'jev'))


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', type=Path)
    parser.add_argument('output', type=Path, help='new JSON receipt; never overwritten')
    adapters = parser.add_mutually_exclusive_group()
    adapters.add_argument('--baseline-v3', action='store_true', help='run supported current-V3 scripted cases')
    adapters.add_argument('--paired-read-only', action='store_true',
                          help='run the V3 and candidate scripted read-only adapters')
    adapters.add_argument('--baseline-v3-seeded-resume', action='store_true',
                          help='run the limited V3 seeded-checkpoint fixture only')
    args = parser.parse_args(argv)
    baseline, candidate = (ExistingV3ReplayAdapter() if args.baseline_v3 else None), None
    if args.paired_read_only:
        from .candidate_comparison import SingleLoopFixtureAdapter, V3ReadFixtureAdapter
        baseline, candidate = V3ReadFixtureAdapter(), SingleLoopFixtureAdapter()
    if args.baseline_v3_seeded_resume:
        from .seeded_v3_resume import V3SeededResumeAdapter
        baseline = V3SeededResumeAdapter()
    result = compare(args.corpus, baseline=baseline, candidate=candidate)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    # An unavailable run is never a qualified comparison.
    return int(any(row['baseline']['passed'] is not True or row['candidate']['passed'] is not True
                   for row in result['cases']))


if __name__ == '__main__':
    raise SystemExit(main())
