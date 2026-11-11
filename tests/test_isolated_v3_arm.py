"""Process isolation of the synthetic flag-off/on V3 fixture arms."""
from pathlib import Path
from io import StringIO
import json
import subprocess

from alphonse.agent_v2.evaluation.isolated_v3_arm import IsolatedV3ReadArm
from alphonse.agent_v2.evaluation.simplified_loop import compare

CORPUS = Path(__file__).parent / 'fixtures/single_loop_comparison/read_only.json'
PRESCRIPTION = Path(__file__).parent / 'fixtures/single_loop_comparison/prescription_schedule_development_v1.json'


def test_flag_off_on_use_isolated_real_v3_processors(monkeypatch):
    monkeypatch.setenv('ALPHONSE_V3_CONTEXT_PROJECTIONS', 'unexpected-parent-value')
    report = compare(CORPUS, baseline=IsolatedV3ReadArm(projection_enabled=False),
                     candidate=IsolatedV3ReadArm(projection_enabled=True))
    for side in ('baseline', 'candidate'):
        assert report['cases'][0][side]['availability'] == 'available'
        assert report['cases'][0][side]['passed'] is True
        assert report['cases'][0][side]['metrics']['provider_input_tokens'] is None
        assert report['cases'][1][side]['availability'] == 'unavailable'
    assert 'projection-off' in report['cases'][0]['baseline']['revision']
    assert 'projection-on' in report['cases'][0]['candidate']['revision']


def test_unsupported_prescription_cases_are_not_scored_as_success():
    report = compare(PRESCRIPTION, baseline=IsolatedV3ReadArm(projection_enabled=False),
                     candidate=IsolatedV3ReadArm(projection_enabled=True))
    for row in report['cases']:
        for side in ('baseline', 'candidate'):
            assert row[side]['availability'] == 'unavailable'
            assert row[side]['passed'] is None


def test_worker_preserves_events_when_adapter_fails(monkeypatch, capsys):
    from alphonse.agent_v2.evaluation import isolated_v3_arm
    from alphonse.agent_v2.evaluation.candidate_comparison import V3ReadFixtureAdapter
    def fail(self, case, root, emit):
        emit({'kind': 'tool', 'tool': 'fixture.read'})
        raise RuntimeError('private provider detail')
    monkeypatch.setattr(V3ReadFixtureAdapter, '__call__', fail)
    monkeypatch.setattr('sys.stdin', StringIO(json.dumps({'case': {}, 'root': '/tmp'})))
    isolated_v3_arm._worker()
    payload = json.loads(capsys.readouterr().out)
    assert payload['events'] == [{'kind': 'tool', 'tool': 'fixture.read'}]
    assert payload['run']['status'] == 'error'
    assert payload['run']['error'] == 'RuntimeError'
    assert 'private provider detail' not in str(payload)


def test_timeout_and_partial_unavailable_are_failures_not_unavailable(monkeypatch):
    from alphonse.agent_v2.evaluation.simplified_loop import _run, load_frozen
    case = load_frozen(CORPUS)[0]['cases'][0]
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd='fixture', timeout=30)
    monkeypatch.setattr(subprocess, 'run', timeout)
    timed_out = _run(case, IsolatedV3ReadArm(projection_enabled=False))
    assert timed_out['availability'] == 'available' and timed_out['passed'] is False

    def partial(*args, **kwargs):
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps({
            'projection_enabled': False, 'events': [{'kind': 'tool', 'tool': 'fixture.read'}],
            'unavailable': 'after_started',
        }))
    monkeypatch.setattr(subprocess, 'run', partial)
    failed = _run(case, IsolatedV3ReadArm(projection_enabled=False))
    assert failed['availability'] == 'available' and failed['passed'] is False
    assert failed['events'] == [{'kind': 'tool', 'tool': 'fixture.read'}]

    def effect_then_decline(*args, **kwargs):
        request = json.loads(kwargs['input'])
        (Path(request['root']) / 'unexpected.txt').write_text('unexpected effect')
        return subprocess.CompletedProcess(args=[], returncode=0, stdout=json.dumps({
            'projection_enabled': False, 'events': [], 'unavailable': 'after_effect',
        }))
    monkeypatch.setattr(subprocess, 'run', effect_then_decline)
    failed = _run(case, IsolatedV3ReadArm(projection_enabled=False))
    assert failed['availability'] == 'available' and failed['passed'] is False
    assert failed['oracle']['observed_effects'] == {'unexpected.txt': 'unexpected effect'}
